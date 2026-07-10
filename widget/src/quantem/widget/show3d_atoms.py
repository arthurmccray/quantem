"""show3d_atoms: interactive 3D volume slice viewer with an atomic-site overlay.

Shows a single orthogonal slice (xy / xz / yz) through a 3D volume in one panel,
with a movable slice position, an intensity histogram with an adjustable color
range, and the traced atomic sites overlaid.  Marker size scales with site
intensity and marker opacity falls off with distance from the slice, so you can
judge tracing quality and pick thresholds slice by slice.

All coordinates are in voxel / array-index space so the sites register exactly
with the volume.

The full volume stays in the kernel; only the *current* 2D slice is streamed to
the frontend (plus the sites and a precomputed global histogram / color range).
This keeps every message well under Jupyter's 10 MiB websocket frame limit, so the
widget works at full resolution on a remote server without any config changes and
regardless of volume size (a 494^3 recon streams ~0.6 MB per slice, not ~300 MB).
"""

import pathlib

import anywidget
import numpy as np
import traitlets

_HIST_BINS = 96
_STAT_SAMPLE = 4_000_000  # cap voxels used for percentile/histogram (speed on big volumes)


class Show3DAtoms(anywidget.AnyWidget):
    """Interactive orthogonal-slice viewer with atomic-site overlay.

    Parameters
    ----------
    volume : ndarray or Dataset3d
        3D scalar volume ``(n0, n1, n2)``.
    sites : ndarray, optional
        ``(N, >=3)`` array of sites in voxel coordinates; columns are
        ``[a0, a1, a2, intensity, sigma]`` (intensity/sigma optional, padded
        with zeros).
    sampling : sequence of float, optional
        Voxel size per axis (for axis labels). Default ``(1, 1, 1)``.
    title : str, optional
        Title shown above the panel.
    cmap : str, default "gray"
        Colormap name.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3d_atoms.js"

    # Dimensions + sites (voxel/array-index space).
    n0 = traitlets.Int(0).tag(sync=True)
    n1 = traitlets.Int(0).tag(sync=True)
    n2 = traitlets.Int(0).tag(sync=True)
    num_sites = traitlets.Int(0).tag(sync=True)
    sites_bytes = traitlets.Bytes(b"").tag(sync=True)
    sampling = traitlets.List(traitlets.Float(), default_value=[1.0, 1.0, 1.0]).tag(sync=True)

    # Current 2D slice, streamed on (plane, slice_index) change.
    slice_bytes = traitlets.Bytes(b"").tag(sync=True)
    slice_rows = traitlets.Int(0).tag(sync=True)
    slice_cols = traitlets.Int(0).tag(sync=True)

    # Global intensity stats (computed once) for the color range + histogram.
    data_lo = traitlets.Float(0.0).tag(sync=True)
    data_hi = traitlets.Float(1.0).tag(sync=True)
    hist_bins = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)

    # Display / interaction state.
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("gray").tag(sync=True)
    plane = traitlets.Unicode("xy").tag(sync=True)  # "xy" | "xz" | "yz"
    slice_index = traitlets.Int(0).tag(sync=True)
    slice_thickness = traitlets.Float(3.0).tag(sync=True)  # voxels at full opacity
    opacity_falloff = traitlets.Float(3.0).tag(sync=True)  # extra voxels fading to 0
    vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vmax_pct = traitlets.Float(100.0).tag(sync=True)
    marker_scale = traitlets.Float(1.0).tag(sync=True)
    marker_linewidth = traitlets.Float(1.0).tag(sync=True)
    marker_filled = traitlets.Bool(True).tag(sync=True)
    show_sites = traitlets.Bool(True).tag(sync=True)
    show_slice = traitlets.Bool(True).tag(sync=True)
    canvas_size = traitlets.Int(520).tag(sync=True)

    def __init__(
        self,
        volume,
        sites=None,
        *,
        sampling=None,
        title="",
        cmap="gray",
        max_bytes=None,  # deprecated / ignored: slices stream, so no volume-size limit
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Full-resolution volume stays in the kernel; never synced wholesale.
        self._vol = np.ascontiguousarray(self._to_numpy3d(volume), dtype=np.float32)
        self.n0, self.n1, self.n2 = (int(s) for s in self._vol.shape)

        if sites is None:
            sites = np.zeros((0, 5), dtype=np.float32)
        sites = np.asarray(sites, dtype=np.float32)
        if sites.ndim != 2 or sites.shape[1] < 3:
            raise ValueError(
                f"sites must be (N, >=3) [a0,a1,a2,(intensity),(sigma)], got {sites.shape}"
            )
        if sites.shape[1] < 5:
            sites = np.pad(sites, ((0, 0), (0, 5 - sites.shape[1])), constant_values=0.0)
        self._sites = sites[:, :5]
        self.num_sites = int(sites.shape[0])
        self.sites_bytes = np.ascontiguousarray(self._sites, dtype=np.float32).tobytes()

        if sampling is not None:
            self.sampling = [float(s) for s in sampling]
        self.title = title
        self.cmap = cmap

        # Global color range + histogram (subsampled for speed), computed once.
        lo, hi, bins = self._global_stats(self._vol)
        self.data_lo, self.data_hi = lo, hi
        self.hist_bins = bins

        # xy plane fixes axis 2; start in the middle, then stream that slice.
        self.slice_index = int(self.n2 // 2)
        self._update_slice()

    @staticmethod
    def _global_stats(vol):
        flat = vol.reshape(-1)
        if flat.size > _STAT_SAMPLE:
            flat = flat[:: flat.size // _STAT_SAMPLE + 1]
        lo, hi = (float(x) for x in np.percentile(flat, [0.5, 99.5]))
        if hi <= lo:
            lo, hi = float(flat.min()), float(flat.max())
            if hi <= lo:
                hi = lo + 1.0
        rng = hi - lo if hi > lo else 1.0
        idx = np.clip(((flat - lo) / rng * _HIST_BINS).astype(np.int64), 0, _HIST_BINS - 1)
        counts = np.bincount(idx, minlength=_HIST_BINS).astype(np.float64)
        counts /= max(counts.max(), 1e-9)
        return lo, hi, [float(c) for c in counts]

    @traitlets.observe("plane", "slice_index")
    def _on_view_change(self, _change):
        self._update_slice()

    def _update_slice(self):
        vol = getattr(self, "_vol", None)
        if vol is None:
            return
        p = self.plane
        # match the frontend's plane convention (rows/cols = the two non-normal axes)
        if p == "xz":  # normal axis 1
            k = int(np.clip(self.slice_index, 0, self.n1 - 1))
            sl = vol[:, k, :]
        elif p == "yz":  # normal axis 0
            k = int(np.clip(self.slice_index, 0, self.n0 - 1))
            sl = vol[k, :, :]
        else:  # xy, normal axis 2
            k = int(np.clip(self.slice_index, 0, self.n2 - 1))
            sl = vol[:, :, k]
        sl = np.ascontiguousarray(sl, dtype=np.float32)
        self.slice_rows, self.slice_cols = int(sl.shape[0]), int(sl.shape[1])
        self.slice_bytes = sl.tobytes()

    @staticmethod
    def _to_numpy3d(volume):
        if isinstance(volume, np.ndarray):
            arr = volume
        else:
            arr = getattr(volume, "array", None)
            if arr is None:
                if hasattr(volume, "detach"):  # torch tensor
                    arr = volume.detach().cpu().numpy()
                elif hasattr(volume, "numpy"):  # Dataset
                    arr = volume.numpy()
                else:
                    arr = volume
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"volume must be 3D, got shape {arr.shape}")
        return arr

    def __repr__(self) -> str:
        return (
            f"Show3DAtoms(shape=({self.n0}, {self.n1}, {self.n2}), "
            f"{self.num_sites} sites, plane={self.plane})"
        )
