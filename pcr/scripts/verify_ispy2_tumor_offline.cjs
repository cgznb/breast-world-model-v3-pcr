const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL, fileURLToPath} = require('node:url');
const {chromium} = require('playwright');

const root = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3]);
const policies = ['direct_t0', 'adjacent_real', 'rollout_generated_dce0_real_ser'];
const steps = [2, 10, 20, 50];

async function ready(page) {
  await page.waitForFunction(() => document.body.dataset.renderReady === 'true', {timeout: 60000});
  assert.equal(await page.locator('#error').innerText(), '');
}

async function checkCanvases(page, target, policy, counts) {
  await ready(page);
  const source = policy === 'direct_t0' ? 0 : target - 1;
  const kind = policy === 'rollout_generated_dce0_real_ser' && source > 0 ? 'generated' : 'real';
  assert.ok((await page.locator('#case-interval').innerText()).includes(`T${source} to T${target}`));
  for (const count of counts) {
    for (const model of ['biflow', 'symmflow']) {
      const panel = page.locator(`.step-row[data-steps="${count}"] .model.${model}`);
      assert.deepEqual(await panel.locator('figcaption').allTextContents(), [
        `Actual input: ${kind} T${source}`, `Generated target T${target}`, `Real target T${target}`,
      ]);
      const masks = await panel.locator('canvas').evaluateAll(cs => cs.map(c => c.dataset.mask));
      assert.equal(masks[1], '');
      assert.equal(Boolean(masks[0]), kind === 'real');
    }
  }
  // Pixel reads also verify that file:// canvas security does not block exports.
  const pixels = await page.locator('#cases canvas').evaluateAll(cs => cs.map(c => {
    const values = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let low = 255, high = 0;
    for (let i = 0; i < values.length; i += 4) {
      low = Math.min(low, values[i]);
      high = Math.max(high, values[i]);
    }
    return {low, high, ready: c.dataset.ready, image: c.dataset.image};
  }));
  assert.equal(pixels.length, 4 + 6 * counts.length);
  assert.ok(pixels.every(p => p.high > p.low && p.ready === 'true'));
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  assert.equal(await page.evaluate(() => imageCache.size <= 64), true);
}

async function main() {
  fs.mkdirSync(output, {recursive: true});
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'offline_manifest.json'), 'utf8'));
  assert.equal(manifest.image_references, 3120);
  // Default Chromium security settings; no local-file-access exception.
  const browser = await chromium.launch({headless: true});
  const errors = [], requests = [], views = [];
  try {
    for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}]) {
      const context = await browser.newContext({viewport, offline: true, acceptDownloads: true});
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      page.on('requestfailed', request => errors.push(request.url() + ': ' + request.failure().errorText));
      page.on('request', request => {
        if (!/^(file|data|blob):/.test(request.url())) requests.push(request.url());
      });
      await page.goto(pathToFileURL(path.join(root, 'index.html')).href);
      await ready(page);
      const study = await page.locator('#study-data').evaluate(node => JSON.parse(node.textContent));
      const patients = study.cases.map(c => c.patient_id);
      assert.equal(patients.length, 20);
      await checkCanvases(page, 3, 'adjacent_real', steps);
      await page.screenshot({path: path.join(output, `desktop-or-mobile-${viewport.width}.png`)});
      await page.locator('.step-row').first().screenshot({path: path.join(output, `comparison-${viewport.width}.png`)});

      for (const count of steps) {
        await page.selectOption('#case-steps', String(count));
        await checkCanvases(page, 3, 'adjacent_real', [count]);
      }
      await page.selectOption('#case-steps', 'all');
      await ready(page);
      const plane = await page.locator('#slice').inputValue();
      await page.locator('#slice-next').click();
      await ready(page);
      assert.notEqual(await page.locator('#slice').inputValue(), plane);
      await page.locator('#slice-prev').click();
      await ready(page);
      assert.equal(await page.locator('#slice').inputValue(), plane);
      const original = await page.locator('.step-row canvas').first().evaluate(c => c.toDataURL());
      await page.locator('#show-mask').uncheck();
      await ready(page);
      assert.notEqual(await page.locator('.step-row canvas').first().evaluate(c => c.toDataURL()), original);
      await page.locator('input[name="field"][value="full"]').check({force: true});
      await ready(page);
      await page.locator('input[name="field"][value="roi"]').check({force: true});
      await page.locator('#show-mask').check();
      await ready(page);
      assert.equal(await page.locator('.step-row canvas').first().evaluate(c => c.toDataURL()), original);
      await page.locator('#real-reference summary').click();
      await page.locator('#truth-reference').screenshot({path: path.join(output, `real-ser-${viewport.width}.png`)});
      await page.locator('#real-reference summary').click();
      await page.locator('#timeline summary').click();
      await page.waitForFunction(() => document.querySelectorAll('#timeline-images canvas').length === 8);
      await ready(page);
      await page.locator('#timeline summary').click();
      await page.waitForFunction(() => document.querySelectorAll('#timeline-images canvas').length === 0);
      await ready(page);
      const downloadEvent = page.waitForEvent('download');
      await page.locator('#download-png').click();
      const download = await downloadEvent;
      assert.equal(await download.failure(), null);
      const png = path.join(output, `offline-export-${viewport.width}.png`);
      await download.saveAs(png);
      const header = fs.readFileSync(png);
      assert.equal(header.subarray(1, 4).toString(), 'PNG');
      assert.equal(header.readUInt32BE(16), 1680);
      assert.equal(header.readUInt32BE(20), 1764);

      let filters = 0;
      for (const region of ['common_foreground', 'target_tumor', 'tumor_union_neighborhood']) {
        await page.selectOption('#region', region);
        await page.locator('#image-plot').evaluate(image => image.decode());
        for (const policy of policies) {
          await page.selectOption('#policy', policy);
          for (const depth of ['T0', 'T0-T1', 'T0-T2', 'T0-T3']) {
            await page.selectOption('#depth', depth);
            const actual = await page.locator('#metrics-body tr').evaluateAll(rows => rows.map(r => [...r.cells].map(c => c.textContent)));
            const images = study.cohort.images.filter(r => r.region === region && r.strategy === policy)
              .sort((a, b) => a.model.localeCompare(b.model) || a.steps - b.steps);
            assert.equal(actual.length, 9);
            for (let i = 0; i < images.length; i++) {
              const image = images[i];
              const pcr = study.cohort.pcr.find(r => r.model === image.model && r.steps === image.steps && r.strategy === policy && r.temporal_depth === depth);
              assert.deepEqual(actual[i].slice(2), [image.mae_unclipped_mean, image.ssim_3d_windowed_mean,
                image.psnr_windowed_db_mean, pcr.auroc_mean, pcr.prauc_mean].map(n => n.toFixed(5)));
            }
            filters++;
          }
        }
      }

      let caseViews = 0;
      for (const patient of patients) {
        await page.selectOption('#case-patient', patient);
        for (const policy of policies) {
          await page.selectOption('#case-policy', policy);
          for (const target of [1, 2, 3]) {
            await page.selectOption('#case-target', String(target));
            await checkCanvases(page, target, policy, steps);
            caseViews++;
          }
        }
        console.log(`${viewport.width}px: offline case ${caseViews / 9}/20 passed`);
      }
      for (const policy of policies) {
        await page.selectOption('#case-policy', policy);
        await ready(page);
        const links = await page.locator('a[href]').evaluateAll(as => as.map(a => a.href));
        for (const link of links) {
          assert.ok(link.startsWith('file:'), link);
          const filename = fileURLToPath(link);
          assert.ok(filename.startsWith(root + path.sep), link);
          assert.ok(fs.statSync(filename).isFile(), link);
        }
      }
      views.push({viewport, case_views: caseViews, metric_filters: filters, png_export: true,
        canvas_pixel_checks: true, controls: true, local_links: true});
      await context.close();
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(requests, []);
    const audit = {status: 'passed', created_utc: new Date().toISOString(), protocol: 'file:', offline: true,
      default_browser_security: true, image_references: manifest.image_references, views,
      browser_errors: errors, network_requests: requests};
    fs.writeFileSync(path.join(output, 'offline_browser_audit.json'), JSON.stringify(audit, null, 2) + '\n');
    console.log(JSON.stringify(audit));
  } finally {
    await browser.close();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
