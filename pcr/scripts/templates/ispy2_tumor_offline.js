// Classic local scripts keep canvas export usable under file:// without a server.
(() => {
  const pending = new Map();
  let current = null;

  function loadCase(patient) {
    if (current && current.patient === patient) return Promise.resolve(current);
    if (pending.has(patient)) return pending.get(patient).promise;
    const script = document.createElement('script');
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    const entry = {promise, resolve, reject, data: null};
    pending.set(patient, entry);
    const finish = () => { pending.delete(patient); script.remove(); };
    script.onload = () => {
      if (entry.data) {
        current = entry.data;
        resolve(current);
      } else reject(new Error('Offline image bundle is incomplete: ' + patient));
      finish();
    };
    script.onerror = () => {
      reject(new Error('Offline image bundle is unavailable: ' + patient));
      finish();
    };
    script.src = 'offline_assets/' + patient + '.js';
    document.head.append(script);
    return promise;
  }

  window.ispy2Offline = {
    register(patient, images, names) {
      const entry = pending.get(patient);
      if (!entry) throw new Error('Unexpected offline image bundle');
      entry.data = {patient, images, names};
    },
    async image(url) {
      const match = /^tumor_review\/cases\/(ISPY2-\d+)\/([^/]+\.png)$/.exec(url);
      if (!match) throw new Error('Unexpected offline image path: ' + url);
      const data = await loadCase(match[1]);
      const index = data.names[match[2]];
      if (!Number.isInteger(index) || !data.images[index]) {
        throw new Error('Offline image is missing: ' + match[2]);
      }
      return data.images[index];
    },
  };
})();
