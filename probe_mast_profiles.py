"""Check that MAST kinetic profiles can be put on psi_n / rho_tor.

Reads one shot from the public FAIR-MAST store (anonymous S3, no login):
  - raw Thomson channels   level1 `ayc`  (te, ne + errors, radius per time)
  - raw CXRS channels      level1 `act`  (SS beam: carbon Ti, v_tor + errors)
  - offline EFIT++ map     level1 `efm`  (psirz, psi_axis, psi_boundary, q(psi_n))

Each channel's (R, Z=0) is mapped to psi_n with the converged EFIT slice nearest
in time, then psi_n -> rho_tor using q(psi_n). Values are converted to the
DIII-D contract units (docs/state_space.md): ne 1e19 m^-3, Te/Ti keV, rotation
krad/s. Channel Z is not stored in the archive; Z=0 (midplane chords) is assumed.

Prints a mapping sanity check (psi_n at the magnetic axis should be ~0, psi_n
along EFIT's LCFS outline should be ~1) and writes
plots/mast_profiles_psin_<shot>_<t>ms.png.

Level 1 `efm` is used because q(psi_n) only exists there. Its psirz is stored
NaN-padded (129 columns, 65 real); level 2 `equilibrium.psi` holds the same values.

Needs python>=3.11 with xarray, zarr>=3, s3fs, scipy, matplotlib.

    python probe_mast_profiles.py [shot] [time_ms]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(HERE, 'plots')
S3 = 'https://s3.echo.stfc.ac.uk/mast'

TS_COLOR, CX_COLOR = '#2a78d6', '#eb6834'
SURF, INK, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
# ACT fills missing fits with huge sentinels (|v| = 3e8 m/s, zero error)
MAX_T_EV, MAX_V_MS = 1e5, 1e6


def open_group(shot, group, level=1):
    return xr.open_zarr(f'{S3}/level{level}/shots/{shot}.zarr', group=group)


def load_efit(shot):
    efm = open_group(shot, 'efm')[['psirz', 'gridr', 'gridz', 'psi_axis', 'psi_boundary',
                                    'magnetic_axis_r', 'magnetic_axis_z', 'lcfs_r', 'lcfs_z',
                                    'qpsi_c']].load()
    psirz = efm.psirz.values
    cols = np.isfinite(psirz).any(axis=(0, 1))
    if cols.sum() != efm.gridr.size:
        raise ValueError(f'psirz has {cols.sum()} finite columns, gridr has {efm.gridr.size}')
    return efm, psirz[:, :, cols]


def psin_interp(efm, psirz, i):
    """Callable (R, Z) -> psi_n for EFIT slice i, or None if the slice did not converge."""
    pa, pb = efm.psi_axis.values[i], efm.psi_boundary.values[i]
    if not (np.isfinite(pa + pb) and np.isfinite(psirz[i]).all()):
        return None
    f = RegularGridInterpolator((efm.gridz.values, efm.gridr.values), psirz[i],
                                method='cubic', bounds_error=False)
    return lambda R, Z: (f(np.c_[np.broadcast_to(Z, np.shape(R)).ravel(), np.ravel(R)])
                         .reshape(np.shape(R)) - pa) / (pb - pa)


def rho_tor_of_psin(efm, i):
    """rho_tor = sqrt(Phi / Phi_edge) with Phi(psi_n) = integral of q dpsi_n."""
    psin = efm.psi_norm.values
    q = np.abs(efm.qpsi_c.values[i])
    phi = cumulative_trapezoid(q, psin, initial=0)
    return psin, np.sqrt(phi / phi[-1])


def mapping_check(efm, psirz, conv):
    at_axis, on_lcfs = [], []
    for i in conv:
        pn = psin_interp(efm, psirz, i)
        at_axis.append(pn(np.array([efm.magnetic_axis_r.values[i]]), efm.magnetic_axis_z.values[i])[0])
        lr, lz = efm.lcfs_r.values[i], efm.lcfs_z.values[i]
        ok = np.isfinite(lr) & np.isfinite(lz)
        on_lcfs.append(np.nanmedian(pn(lr[ok], lz[ok])))
    print(f'  psi_n mapping check over {len(conv)} converged EFIT slices:')
    print(f'    at magnetic axis   median {np.median(at_axis):+.4f}  (want 0)')
    print(f'    along EFIT LCFS    median {np.median(on_lcfs):.4f}  '
          f'p10-p90 {np.percentile(on_lcfs, 10):.3f}-{np.percentile(on_lcfs, 90):.3f}  (want 1)')


def nearest(times, t, has_data):
    idx = np.flatnonzero(has_data)
    return int(idx[np.argmin(np.abs(times[idx] - t))])


def main(shot=30421, time_ms=250):
    shot, t = int(shot), float(time_ms) / 1e3
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f'shot {shot} @ {time_ms} ms')

    efm, psirz = load_efit(shot)
    conv = [i for i in range(efm.time.size) if psin_interp(efm, psirz, i) is not None]
    mapping_check(efm, psirz, conv)

    ayc = open_group(shot, 'ayc')[['te', 'ne', 'te_error', 'ne_error', 'radius']].load()
    it = nearest(ayc.time.values, t, (np.isfinite(ayc.te.values) & (ayc.te.values > 0)).any(axis=1))
    act = open_group(shot, 'act')[['ss_temperature', 'ss_temperature_error',
                                   'ss_velocity', 'ss_velocity_error']].load()
    raw_ti = act.ss_temperature.values
    ic = nearest(act.time.values, t, ((raw_ti > 0) & (raw_ti < MAX_T_EV)).any(axis=1))

    ie = conv[int(np.argmin(np.abs(efm.time.values[conv] - ayc.time.values[it])))]
    pn = psin_interp(efm, psirz, ie)
    rho_x, rho_y = rho_tor_of_psin(efm, ie)
    r_axis, z_axis = efm.magnetic_axis_r.values[ie], efm.magnetic_axis_z.values[ie]

    R_ts = ayc.radius.values[it]
    ts = {}
    for key, err, scale in [('te', 'te_error', 1e-3), ('ne', 'ne_error', 1e-19)]:
        v, e = ayc[key].values[it], ayc[err].values[it]
        ok = np.isfinite(v) & np.isfinite(e) & np.isfinite(R_ts) & (v > 0) & (e > 0)
        ts[key] = (R_ts[ok], pn(R_ts[ok], 0.0), v[ok] * scale, e[ok] * scale)

    R_cx = act.majorradius.values
    cx = {}
    ti, ti_e = raw_ti[ic], act.ss_temperature_error.values[ic]
    ok = np.isfinite(ti) & np.isfinite(ti_e) & (ti > 0) & (ti < MAX_T_EV) & (ti_e > 0)
    cx['ti'] = (R_cx[ok], pn(R_cx[ok], 0.0), ti[ok] * 1e-3, ti_e[ok] * 1e-3)
    v, v_e = act.ss_velocity.values[ic], act.ss_velocity_error.values[ic]
    ok = np.isfinite(v) & np.isfinite(v_e) & (np.abs(v) < MAX_V_MS) & (v_e > 0)
    # linear m/s -> angular krad/s to match the DIII-D contract
    cx['omega'] = (R_cx[ok], pn(R_cx[ok], 0.0), v[ok] / R_cx[ok] * 1e-3, v_e[ok] / R_cx[ok] * 1e-3)

    n_sent = int(np.sum(np.isfinite(raw_ti) & ((raw_ti <= 0) | (raw_ti >= MAX_T_EV))))
    print(f'  times used: TS {ayc.time.values[it]*1e3:.1f} ms, CXRS {act.time.values[ic]*1e3:.1f} ms, '
          f'EFIT {efm.time.values[ie]*1e3:.1f} ms  (magnetic axis R={r_axis:.3f} m, Z={z_axis:+.3f} m)')
    for name, d in [('Te', ts['te']), ('ne', ts['ne']), ('Ti', cx['ti']), ('omega', cx['omega'])]:
        if d[0].size == 0:
            print(f'  {name:6s}   0 valid channels at this time')
            continue
        print(f'  {name:6s} {d[0].size:3d} valid channels, {np.sum(d[1] <= 1):3d} inside LCFS, '
              f'psi_n {np.nanmin(d[1]):.2f}-{np.nanmax(d[1]):.2f}')
    print(f'  CXRS sentinel Ti entries masked (whole shot): {n_sent}')
    print(f'  rho_tor at psi_n = 0.5: {np.interp(0.5, rho_x, rho_y):.3f}')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor(SURF)
    panels = [(axes[0, 0], ts['ne'], TS_COLOR, r'Thomson $n_e$  [$10^{19}\,m^{-3}$]'),
              (axes[0, 1], ts['te'], TS_COLOR, r'Thomson $T_e$  [keV]'),
              (axes[1, 0], cx['ti'], CX_COLOR, r'CXRS carbon $T_i$  [keV]'),
              (axes[1, 1], cx['omega'], CX_COLOR, r'CXRS $\omega_{tor} = v/R$  [krad/s]')]
    for ax, (R, p, val, err), color, title in panels:
        out = R >= r_axis
        ax.errorbar(p, val, yerr=err, fmt='none', ecolor=color, elinewidth=1, alpha=0.35)
        ax.plot(p[out], val[out], 'o', ms=6, color=color, mec=SURF, mew=1, label='outboard (R > R_axis)')
        ax.plot(p[~out], val[~out], 'o', ms=6, mfc='none', mec=color, mew=1.2, label='inboard')
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xlim(0, 1.6)
        ax.axvline(1.0, color=AXIS, lw=1)
        ax.set_xlabel(r'$\psi_n$', color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=MUTED)

    ax = axes[0, 2]
    ax.plot(rho_x, rho_y, color=TS_COLOR, lw=2, label=r'$\rho_{tor}$ from $q(\psi_n)$')
    ax.plot(rho_x, np.sqrt(rho_x), color=AXIS, lw=1, label=r'$\rho_{pol}=\sqrt{\psi_n}$ (reference)')
    ax.set_title(r'$\psi_n \to \rho_{tor}$ (EFIT slice)', color=INK, fontsize=11, loc='left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\psi_n$', color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower right')

    ax = axes[1, 2]
    ax.plot(ts['te'][0], ts['te'][1], 'o', ms=5, color=TS_COLOR, mec=SURF, mew=1)
    ax.axvline(r_axis, color=AXIS, lw=1)
    ax.set_title(r'Thomson channel R $\to \psi_n$ (chord at Z = 0)', color=INK, fontsize=11, loc='left')
    ax.text(0.02, 0.97, f'axis R = {r_axis:.2f} m, Z = {z_axis:+.2f} m\n'
            'core floor > 0 when the axis sits off Z = 0',
            transform=ax.transAxes, color=MUTED, fontsize=9, va='top')
    ax.set_xlabel('R  [m]', color=MUTED)
    ax.set_ylabel(r'$\psi_n$', color=MUTED)

    for ax in axes.ravel():
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(f'MAST {shot}: raw channels on psi_n with offline EFIT++ '
                 f'(TS {ayc.time.values[it]*1e3:.0f} ms, CXRS {act.time.values[ic]*1e3:.0f} ms; '
                 f'line at psi_n = 1 is the LCFS)', color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'mast_profiles_psin_{shot}_{int(float(time_ms))}ms.png')
    fig.savefig(out, dpi=130, facecolor=SURF)
    print(f'  wrote {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
