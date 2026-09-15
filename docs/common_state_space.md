# Common state space: DIII-D + MAST

Companion to `docs/state_space.md` (the DIII-D contract). This doc checks every
quantity in that contract against **MAST** (public FAIR-MAST archive,
https://mastapp.site) and proposes the **shared** state space for a cross-device
world model.

Checked on 2026-09-15 against the live archive (level 1 + level 2 zarr, S3) and
the ingestion mapping `ukaea/fair-mast-ingestion: mappings/level2/mast.yml`.
Proof script: `probe_mast_profiles.py` (see the end).

---

## TL;DR

- **Profiles: all 4 exist on MAST** (ne, Te, Ti, rotation), as raw channels with
  per-channel errors. The archive does **not** give them in psi_n / rho. We can
  compute both from the stored EFIT++ flux map. The mapping checks out
  (psi_n = 0.000 at the magnetic axis, 0.999 along EFIT's LCFS).
- **Scalars: all 7 exist.**
- **Actuators: most exist.** Missing on MAST: **ECH** (MAST had none).
  **NBI torque** is not stored, but we can compute it: per-beam power by energy
  part, beam energy per shot, tangency radius 0.7 m, direction from the sign of
  Ip (recipe below). **Gas** exists but as 20 per-valve voltages, not a
  DIII-D-style gasA.
- **The "real-time only" rule cannot hold on MAST.** The archive has no real-time
  EFIT. Every equilibrium quantity is offline EFIT++. For the shared model use
  **offline EFIT on both** (DIII-D: EFIT01, already stored).
- **Usable shots:** 4,168 with Thomson + CXRS + EFIT (M6-M9). 2,407 of those use
  the newer 130-channel Thomson.

---

## MAST in one paragraph

Spherical tokamak (Culham, UK), campaigns M5-M9, 11,573 shots in the archive.
Low aspect ratio: axis R ≈ 0.9-1.0 m, a ≈ 0.6 m. Vacuum field ≈ -0.42 T at
R = 1 m (median). Ip ≈ 0.74 MA (median). Short shots: plasma ends at 0.38 s
(median, range 0.1-0.7 s). Heating is NBI only: two deuterium beams, SS and SW,
up to 75 keV and 2.5 MW each, tangential, both aimed along normal Ip.

Normal shots run Ip > 0, Bt < 0. **384 shots run with both reversed**
(M5 13469-13696, M7 22376-22626, plus 4 strays). On those the beams push
**against** Ip (counter-NBI). 150 of them have CXRS.

Data access: anonymous S3, zarr v3, one store per shot.

```python
import xarray as xr
S3 = 'https://s3.echo.stfc.ac.uk/mast'
ts  = xr.open_zarr(f'{S3}/level1/shots/30421.zarr', group='ayc')          # raw Thomson
eq  = xr.open_zarr(f'{S3}/level2/shots/30421.zarr', group='equilibrium')  # IMAS-style
```

- **Level 1** = per-diagnostic analysed data, native channels, with errors.
  Group names are the 3-letter UDA source codes (`ayc`, `act`, `efm`, `anb`, ...).
- **Level 2** = IMAS-style groups (`thomson_scattering`, `charge_exchange`,
  `equilibrium`, `summary`, `pulse_schedule`, `gas_injection`, `tf`, ...).
  Profiles get **linearly regridded onto a 1 cm R grid, errors dropped**. Use
  level 2 for scalars/actuators, level 1 for profiles.
- **Shot table** (`https://mastapp.site/parquet/level2/shots`) = one row per shot,
  189 summary fields, including beam energy per beam.
- Needs Python >= 3.11 (zarr 3). The cluster default python 3.7 is too old.

### Shot coverage (from the archive source index)

| Campaign | All | Thomson + EFIT | Thomson + EFIT + CXRS |
|---|---|---|---|
| M5 | 1,960 | 1,721 | 0 |
| M6 | 2,697 | 2,551 | 686 |
| M7 | 3,514 | 2,926 | 1,668 |
| M8 | 2,009 | 1,773 | 1,003 |
| M9 | 1,101 | 966 | 603 |
| Unknown | 292 | 288 | 208 |
| **Total** | **11,573** | **10,225** | **4,168** |

- 4,000 of the 4,168 have NBI on.
- Thomson system changes at shot 22832: `atm` before (≈35 channels), `ayc` from
  22832 on (130 channels). Thomson + CXRS in the `ayc` era: **2,407 shots**.

---

## Checklist results (the `state_space.md` checklist, answered for MAST)

"RT?" = real-time available. MAST's archive only has post-shot analysed
diagnostics and offline EFIT++. PCS request signals (`xdc`) are the only
genuinely real-time signals.

### Profiles

| Quantity | On MAST? | MAST units | RT? | Source (level 1) | Caveats |
|---|---|---|---|---|---|
| ne | yes | m^-3 | no | `ayc/ne` + `ne_error`, `ayc/radius` | ÷1e19 for contract. Pre-22832 shots: `atm` |
| Te | yes | eV | no | `ayc/te` + `te_error` | ÷1e3 for keV |
| Ti | yes (4,168 shots) | eV | no | `act/ss_temperature` + `_error` | carbon CXRS, only while SS beam is on; ACT also has `sw_*` and `pla_*` views (not checked yet) |
| Rotation | yes (same shots) | **linear m/s** | no | `act/ss_velocity` + `_error` | convert ω = v / R (R = channel `majorradius`). Lab-frame sign, see torque section |
| Uncertainty | yes, level 1 only | absolute, same units as value | – | `*_error` | medians: Te 8-14 %, ne 5-6 %, Ti 3-5 %. Level 2 has none |
| (R, Z) → psi_n | yes, offline | – | **no** | `efm/psirz`, `gridr`, `gridz`, `psi_axis`, `psi_boundary` | no RT EFIT archived. Channel Z not stored (see below) |
| rho_tor | computable | – | no | `efm/qpsi_c` on `psi_norm` | not stored; rho_tor = sqrt(Φ/Φ_edge), Φ = ∫ q dψ_n |

Time bases: Thomson 4.2 ms (240 Hz laser bursts, some slots empty), CXRS 5 ms,
EFIT ≈ 5 ms.

### Scalars

| Quantity | On MAST? | Source (level 2) | RT? | Caveats |
|---|---|---|---|---|
| betaN | yes | `equilibrium/beta_tor_normal` | no | archive labels units "T"; it is the usual %·m·T/MA |
| n̄e | yes | `summary/line_average_n_e` (m^-3) | no | also raw line-integral `interferometer/n_e_line` (m^-2) |
| li | yes | `equilibrium/li` | no | |
| q0 | yes | `equilibrium/q_axis` | no | |
| q95 | yes | `equilibrium/q95` | no | |
| Vloop | yes | `equilibrium/vloop_static`, `vloop_dynamic` | no | two definitions; pick one |
| Wmhd | yes | `equilibrium/wmhd` (also `plasma_energy`) | no | ~20× smaller than DIII-D; normalise per device |

### Actuators

| Quantity | On MAST? | Source | RT? | Caveats |
|---|---|---|---|---|
| NBI power (total) | yes | `summary/power_nbi` (W); per beam `anb/ss_sum_power`, `sw_sum_power` (MW) | – | |
| NBI torque | **computable** | `anb/{ss,sw}_{full,half,third}_power` + shot table `nbi_energy_{ss,sw}_max_power` + R_tan 0.7 m + sign(Ip) | – | injected torque; recipe below |
| Ip target | yes | `pulse_schedule/i_plasma` (PCS `xdc` ipref) | yes | |
| Ip measured | yes | `summary/ip` | – | |
| Bt magnitude + sign | yes | `equilibrium/bvac_val` at `bvac_r` (T); `tf/coil_current` (A) | – | negative on normal shots, positive on the 384 reversed shots |
| Gas command | **different form** | `gas_injection/valve_voltage` (20 ch), `valve_target_voltage` (12 ch) | – | per-valve hardware. Density request also exists: `pulse_schedule/n_e_line` |
| Shape: a, δ_top, δ_bot, κ, R_axis, Z_axis | yes | `equilibrium/minor_radius`, `triangularity_upper/lower`, `elongation`, `magnetic_axis_r/z` | no (offline) | |
| ECH (all) | **no** | – | – | MAST had no ECH heating |

Extra MAST actuators we could keep as device-specific: per-beam NBI power (SS/SW),
`pulse_schedule/z_ref`, `controllers/zip_proxy`, PF coil currents (`pf_active`).

---

## Proposed common state space

### Profiles (core, shared)

| # | Quantity | Units | DIII-D | MAST |
|---|---|---|---|---|
| 0 | ne | 1e19 m^-3 | Thomson | `ayc`/`atm` |
| 1 | Te | keV | Thomson | `ayc`/`atm` |
| 2 | Ti | keV | CER (carbon) | `act` SS (carbon) |
| 3 | ω_tor | krad/s | CER | `act` SS, v/R |

- Raw point cloud per timestep: value, uncertainty, mask, psi_n, rho_tor, (R, Z).
  Same as `state_space.md`.
- **psi_n is the primary axis.** Both machines give it straight from the flux
  map. rho_tor is stored alongside (on MAST it needs the q integral).
- **Equilibrium for the shared model: offline EFIT on both** (DIII-D EFIT01,
  MAST EFIT++). Keep EFITRT1 for a DIII-D-only real-time variant.
- **Rotation sign:** store it relative to Ip on both machines. MAST CXRS gives a
  lab-frame sign that does not flip on reversed-Ip shots.

### Scalars (shared)

betaN, n̄e, li, q0, q95, Vloop, Wmhd. All 7 exist on both machines.

### Actuators

| Group | Shared | DIII-D only (masked on MAST) | MAST only (optional) |
|---|---|---|---|
| Heating | total NBI power, NBI torque (injected, signed vs Ip) | all ECH | per-beam NBI power |
| Current | Ip target, Ip measured | | |
| Field | \|Bt\|, sign(Bt) | | |
| Shape | a, δ_top, δ_bot, κ, R_axis, Z_axis | | z_ref |
| Fueling | *open decision, see below* | gasA | valve voltages, density request |

Suggest also a per-shot **device context** (device id, R0, a, B0) so the model
can tell the machines apart. Magnitudes differ a lot (MAST Bt ≈ 0.4 T vs DIII-D
≈ 2 T; Wmhd ≈ 20× smaller).

---

## Changes this forces on `state_space.md`

1. **Real-time rule.** It cannot apply to the shared model: MAST has no RT
   equilibrium in the archive. Shared model = offline EFIT on both machines.
2. **NBI torque** stays shared, computed on MAST (recipe below). Confirm DIII-D
   `\TINJ` is also *injected* torque before mixing the two.
3. **ECH** leaves the shared set (DIII-D-only, masked on MAST).
4. **Gas.** DIII-D gasA (one valve command) and MAST's 20 valve voltages are
   not the same quantity. Needs a decision (below).
5. **Rotation.** MAST stores linear m/s. Convert to krad/s with channel R, and
   sign it relative to Ip.

## Open decisions

1. **Fueling in the shared set.** (a) keep gas device-specific and masked on the
   other machine, or (b) build a per-device "total fueling" signal. Suggest (a)
   for now.
2. **Time step.** MAST shots last ≈ 0.4 s. At 20 ms that is ≈ 19 steps per shot
   (DIII-D ≈ 250). MAST diagnostics support 5 ms. 20 ms keeps one step size
   across machines. 5-10 ms gives MAST more steps.

---

## Recipe: MAST profiles → psi_n / rho_tor

1. Load `efm` (level 1). `psirz` is stored **NaN-padded**: 129 columns, only 65
   real. Keep the finite columns; they line up with `gridr` (65) × `gridz` (65).
   Skip slices where `psi_axis` is NaN (not converged; ≈ 30 % of slices, mostly
   ramp-up/down).
2. For each Thomson / CXRS time, take the nearest converged EFIT slice.
3. psi_n = (psi(R, Z) − psi_axis) / (psi_boundary − psi_axis), with Z = 0.
4. rho_tor: Φ(ψ_n) = ∫ |q| dψ_n from `qpsi_c`; rho_tor = sqrt(Φ / Φ(1)).
5. Mask bad values:
   - Thomson: finite, value > 0, error > 0.
   - CXRS: `act` writes sentinels for failed fits (v = −3e8 m/s with error 0,
     junk Ti). Keep 0 < Ti < 1e5 eV, |v| < 1e6 m/s, error > 0. Sentinels are
     10-23 % of CXRS entries on the checked shots.

### Validation (shots 30421, 27493, 29387)

- psi_n at the EFIT magnetic axis: 0.000 (median). Along EFIT's LCFS outline:
  0.999 (p10-p90 0.998-1.000).
- Inboard and outboard Thomson and CXRS points overlap on the same psi_n profile
  (`plots/mast_profiles_psin_29387_250ms.png`, `..._30421_250ms.png`; PNGs are
  gitignored, rerun the script to make them).
- Level 1 `psirz` and level 2 `equilibrium.psi` agree (same axis/boundary flux).
- rho_tor at psi_n = 0.5 is 0.51-0.55 (sensible for a spherical tokamak).
- Do **not** sanity-check against EFIT's `rpsi95_out` / `rpsi100_out` at Z_axis:
  they come out 0.03-0.06 low there, while the LCFS points land at 1.

### Caveats

- **Channel Z is not in the archive.** Z = 0 assumed (midplane chords). A 2 cm
  error moves psi_n by ≈ 0.02.
- **The magnetic axis often sits off the midplane** (30421: Z_axis ≈ −0.22 m).
  Then the Z = 0 chords never cross the axis, so core coverage starts at
  psi_n ≈ 0.15 on those shots. Real geometry, not a bug.
- CXRS data exists only while the SS beam is on (45-76 % of time slices on the
  checked shots).

---

## Recipe: MAST NBI torque

Injected torque, per time sample:

```
T(t) = s · Σ_beam Σ_{j=1,2,3} P_j(t) · R_tan · sqrt(2 m_D / (E / j))
```

| Piece | Where | Notes |
|---|---|---|
| P_j(t) | level 1 `anb/ss_full_power`, `ss_half_power`, `ss_third_power` (MW), same for `sw_*` | time traces. j = 1, 2, 3 for full, half, third energy. Some shots have no `sw_*` variables: skip that beam |
| E | shot table `nbi_energy_ss_max_power`, `nbi_energy_sw_max_power` | keV (units undocumented; values fit: SS median 65, max 75; SW median 52). One value per shot, taken at peak power |
| m_D | constant | deuteron mass (both beams inject deuterium) |
| R_tan | literature | 0.7 m for both beams. Not in the archive |
| s | sign of `summary/ip` | beams are fixed hardware aimed along normal Ip. s = +1 co (normal), −1 counter (reversed shots) |

### Checks

- 30421: E_ss 70.9 keV, E_sw 61.4 keV, peak torque 2.25 N·m at ≈ 3.9 MW.
  29387: peak 1.24 N·m.
- Mean power split while on (30421): SS 89 / 9 / 2 %, SW 58 / 26 / 16 %
  (full / half / third).
- The reversed-Ip shots are real, not a sign glitch: the PCS Ip request is
  negative (22376: −0.40 MA, 22626: −0.71 MA; neighbours 22370, 22640: +0.6,
  +0.7 MA), TF current flips sign, and operator logs say "switch to cntr-NBI
  conditioning" and "optimal EFC settings with reversed Ip and TF".
- Lab-frame core rotation keeps its sign on reversed shots (+42 to +85 km/s vs
  +35 km/s on normal shots). The plasma spins with the beams, so the beams run
  counter to Ip there.

### Caveats

- **Injected, not absorbed.** No shine-through, no fast-ion loss. Both are large
  on MAST at low density, and counter-NBI loses more fast ions.
- R_tan is a literature value. Beam energy is one number per shot.

Sources: [On the Neutral Beam Heating of MAST (IAEA)](https://www-pub.iaea.org/mtcd/publications/pdf/csp_008c/pdf/exp4_19.pdf)
(R_tan 0.7 m); [Jones et al., arXiv:1401.6864](https://arxiv.org/pdf/1401.6864)
(SS + SW deuterium beams, 75 keV, 2.5 MW each).

---

## Try it

```bash
~/miniconda3/bin/python -m venv ~/mastenv
~/mastenv/bin/pip install xarray zarr s3fs scipy matplotlib pandas pyarrow
~/mastenv/bin/python probe_mast_profiles.py 29387 250
```

Prints the mapping check + channel counts, writes
`plots/mast_profiles_psin_<shot>_<t>ms.png`.
