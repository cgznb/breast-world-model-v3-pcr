import gzip
import re
import zipfile

import numpy as np
import nibabel as nib

_AQC = re.compile(r"BreastDCEDL_ISPY2/([^/]+)/dce/T(\d)/[^/]*_dce_aqc_(\d+)\.nii\.gz$")


class NiftiZip:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path)
        self.aqc = {}     # (pid, t) -> {acq_index: member_name}
        for n in self.zf.namelist():
            if n.startswith("__MACOSX") or n.endswith("/"):
                continue
            m = _AQC.search(n)
            if m:
                pid, t, idx = m.group(1), int(m.group(2)), int(m.group(3))
                self.aqc.setdefault((pid, t), {})[idx] = n

    def has(self, pid, t):
        return (pid, t) in self.aqc

    def timepoints(self, pid):
        return sorted(t for (p, t) in self.aqc if p == pid)

    def _read(self, name):
        raw = self.zf.read(name)
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        img = nib.Nifti1Image.from_bytes(raw)
        # dataobj is (slice, row, col); reorient to (row, col, slice) so the last axis is the slice
        # axis and the in-plane (row, col) crop is taken around (sraw, scol).
        return np.asarray(img.dataobj, dtype=np.float32).transpose(1, 2, 0)

    def phase_indices(self, pid, t):
        """BreastDCEDL I-SPY channels = [0, 2, min(last,6)], mapped to available acqs."""
        avail = sorted(self.aqc[(pid, t)])
        pre = 0 if 0 in avail else avail[0]
        early = 2 if 2 in avail else avail[min(1, len(avail) - 1)]
        target = min(max(avail), 6)
        late = target if target in avail else min(avail, key=lambda a: abs(a - target))
        return [pre, early, late]

    def rgb_volume(self, pid, t):
        """(3, H, W, Z) float32 — pre/early/late acquisitions stacked on axis 0."""
        idxs = self.phase_indices(pid, t)
        vols = [self._read(self.aqc[(pid, t)][i]) for i in idxs]
        return np.stack(vols, axis=0)
