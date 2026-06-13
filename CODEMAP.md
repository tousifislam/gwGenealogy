# gwGenealogy — Code Map

> Agent-oriented index of every public class/function, plus the import/data-flow graph.
> Generated 2026-06-11. Regenerate when signatures change:
> `grep -nE '^(class |def |    def )' gwGenealogy/**/*.py`

`gwGenealogy` models BBH population phenomenology with one through-line: gravitational
**recoil kicks** decide whether a merger remnant is **retained** in its host and can
undergo **hierarchical** (repeated) mergers to grow toward IMBH/SMBH scales.

Layer roles: `utils` (math primitives) → `binaries` (BBH objects + remnant physics) ·
`stellar` (builds 1G BH pool) · `hosts` (environments / escape velocities) →
`core` (the science: hierarchical mergers, seed growth, retention grids).

---

## Dependency graph

```
core/hierarchical ─┬─> binaries.BBHRemnant
                   ├─> binaries.sample_spin_angles
                   ├─> stellar.sample_1g_bh_masses_from_stellar_collapse
                   ├─> hosts.Mcl_rh_to_vescape
                   └─> utils.distributions (uniform, powerlaw, beta)

core/seed_growth ──┬─> binaries.BBHRemnant
                   ├─> binaries.sample_spin_angles
                   ├─> stellar.sample_1g_bh_masses_from_stellar_collapse
                   └─> utils.distributions (uniform)

core/retention ────┬─> binaries.BBHRemnant
                   ├─> binaries.{sample_spin_magnitudes, sample_spin_angles}
                   └─> utils.distributions (uniform, beta, powerlaw)

binaries/bbh ──────> utils.{conversions, coordinates}
binaries/bbh_remnant ─> gwModels.remnants (HBR/UIB/HLZ/IW2025) + surfinBH [optional]
                      + utils.coordinates
binaries/bbh_spins ─> utils.{coordinates, distributions}
binaries/bbh_masses ─> utils.distributions
binaries/smbbh ────> utils.distributions + binaries.sample_spin_angles
binaries/bbh_gwtc ─> scipy.stats + binaries/data/*.h5,*.json   (self-contained)
binaries/bbh_redshifts ─> scipy   (self-contained; Planck-2018 ΛCDM)

stellar/stellar_evolution ─> stellar.collapse
stellar/collapse ──> stellar/data/stellar_evolution_data.h5  (RAPSTER F12d / SEVN grids)
stellar/natal_kick ─> utils.distributions (maxwellian)

hosts/plummer ─────> scipy.special (self-contained)
hosts/star_clusters ─> (self-contained)
hosts/host_retention ─> scipy.stats (self-contained)

utils/* ───────────> numpy/scipy only (leaf layer)
```

External hard dep: **gwModels** (`gwModels.remnants`, incl. `IW2025_kick_precessing`).
Optional dep: **surfinBH** (NR surrogate remnants; lazy-imported in `bbh_remnant.py`).

---

## Symbol index

### `utils/` — math primitives (leaf layer, numpy/scipy only)

**conversions.py**
- `m1_m2_to_mchirp(m1, m2)` · `m1_m2_to_q(m1, m2)` (→ q≥1) · `m1_m2_to_eta(m1, m2)`
- `source_frame_to_detector_frame_mass(m_source, z)` · `detector_frame_to_source_frame_mass(m_detector, z)`
- `chi_eff(q, chi1z, chi2z)` · `chi_p(q, chi1_perp, chi2_perp)`
- `delta_parallel/delta_perp/chi_tilde_parallel/chi_tilde_perp(q, chi1, chi2, theta1, theta2[, delta_phi])`

**coordinates.py**
- `polar_to_cartesian(r, theta, phi)` · `cartesian_to_polar(x, y, z)` (θ∈[0,π], φ∈[0,2π))
- `spins_polar_to_cartesian_vectors(a1, a2, theta1, theta2, phi1, phi2)` → two (N,3) arrays
- `spins_cartesian_vectors_to_polar(chi1_vec, chi2_vec)` → 6-tuple

**distributions.py** — all `(n_samples, ..., seed=None, plot=False, bins=50)`
- `sample_uniform_1d(low, high)` · `sample_loguniform_1d(low, high, base=10)`
- `sample_gaussian_1d(mean, std)` · `sample_lognormal_1d(mean, sigma)`
- `sample_powerlaw_1d(beta, xmin, xmax)` (∝x^β, inverse-CDF; β=-1 → log) · `sample_beta_1d(a=1.4, b=3.6)`
- `sample_maxwellian_1d(sigma)` (f(v)∝v²e^(-v²/2σ²), mean=σ√(8/π))

**statistics.py**
- `compute_kullback_leibler_divergence(a, b, n_bins=100)` (asymmetric)
- `compute_jensen_shannon_divergence(a, b, n_bins=100)` (symmetric, bounded [0,ln2])

**rcparams.py** — `set_rcparams()` (matplotlib publication style)

---

### `binaries/` — BBH objects + remnant physics

**bbh.py** — `class BBHs`
- `__init__(m1=None, m2=None, M=None, q=None, small_q=None, a1=0, a2=0, theta1=0, theta2=0, phi1=0, phi2=0, z=None)`
- Flexible mass spec: `(m1,m2)` or `(M,q)`. Computes `M,q,eta,mchirp`, spin vectors,
  `chi_eff, chi_p, chi1z, chi2z, delta_*, chi_tilde_*`. Pure container (no sampling).

**bbh_masses.py**
- `sample_masses(n_samples, m_min=5, m_max=50, m1_distribution='uniform', pairing='random', alpha=-2.5, beta=6.7, seed=None)` → (m1, m2), m1≥m2
  - m1: `uniform | loguniform | powerlaw(∝m^alpha)`
  - pairing: `random | secondary_mass_power_law(∝m2^beta) | total_mass_power_law(∝(m1+m2)^4)`

**bbh_spins.py**
- `sample_spins(n_samples, chi_min=0, chi_max=1, spin_magnitude='uniform', spin_angles='isotropic', beta_a=1.4, beta_b=3.6, tilt_beta_a=None, tilt_beta_b=None, seed=None)` → chi1, chi2 (N,3)
- `sample_spin_magnitudes(n_samples, chi_min, chi_max, spin_magnitude, beta_a, beta_b, seed)` → a1,a2 (`uniform|beta`)
- `sample_spin_angles(n_samples, spin_angles='isotropic', ...)` → theta1,theta2,phi1,phi2 (`isotropic|uniform|beta`)

**bbh_redshifts.py** — Planck-2018 flat ΛCDM (H0=67.4, Om0=0.315). p(z)∝R(z)·dVc/dz/(1+z)
- `sample_redshift(n, z_min=0, z_max=10, seed)` (R=const)
- `sample_redshift_powerlaw(n, lamb, ...)` (R∝(1+z)^λ)
- `sample_redshift_madau_dickinson(n, gamma, kappa, z_peak, ...)`
- helpers: `E, comoving_volume_element, luminosity_distance, *_pdf`

**bbh_remnant.py** — `class BBHRemnant` (THE physics engine: remnant mass/spin/kick)
- `__init__(bbh=None, m1=…, m2=…, a1, a2, theta1, theta2, phi1, phi2, precessing=True, mass_spin_model=None, kick_model=None)`
- Outputs attrs: `Mf`, `af`, `vkick`
- Valid models (auto-defaults in **bold**):
  - precessing mass/spin: **hbr**, sur7dq4remnant, sur7dq4emri
  - precessing kick: **gwmodel** (IW2025 flow), hlz, sur7dq4remnant
  - nonprec mass/spin: **uib**, hbr, sur7dq4remnant, sur7dq4emri, sur3dq8remnant
  - nonprec kick: **gwmodel_kick_q200**, hlz, sur3dq8remnant
- `sur*` models require surfinBH; everything else uses gwModels formulas.

**bbh_gwtc.py** — LVK official population models (self-contained, ships data/)
- `sample_gwtc_population(n_samples, catalog='gwtc5', source='posterior', mode='ppd', data_dir=None, n_hyper_draws=1000, gwtc3_hyper_samples=None, seed=None)` → dict
  - catalogs: `gwtc3` (PowerLaw+Peak, Beta spin) · `gwtc4`/`gwtc5` (Broken PL + 2 peaks, Gaussian spin) · `gwtc5_var4` · `gwtc5_madau_dickinson`
  - source: `posterior|prior`; mode: `ppd|mean|<int hyper-draw idx>`
- `available_catalogs()` → list
- data: `binaries/data/{gwtc3_default.json, gwtc4_default.h5, gwtc5_default_var1.h5, gwtc5_default_var4.h5, gwtc5_default_madau_dickinson.h5}`

**smbbh.py**
- `sample_smbbh(n_samples, accretion='hot', m_total_min=1e5, m_total_max=1e11, seed=None)` → dict
  - accretion channels: `agnostic | hot | cold | dry` (set spin mag/tilt distributions). M_total log-uniform.

---

### `stellar/` — builds the first-generation (1G) BH pool

**stellar_evolution.py**
- `IMF_kroupa(m, alpha3=-2.3)` (broken PL, breaks 0.08/0.5/1.0 Msun)
- `sample_kroupa_masses(n, m_min=0.08, m_max=150, alpha3=-2.3, seed)`
- `sample_zams_masses(n, m_zams_min=10, m_zams_max=150, imf='salpeter'|'kroupa'|'uniform', seed)`
- `evolve_stars(M_ZAMS, Z, model='Fryer12_delayed'|'SEVN_delayed')` → remnant masses (gap-filtered)
- `sample_1g_bh_masses_from_stellar_collapse(n, Z=0.0002, model, m_zams_min, m_zams_max, imf, seed)`
  — **main entry used by core/**: IMF → ZAMS → collapse → keep only non-gap BHs.

**collapse.py** — RAPSTER interpolation tables, lazy-loaded from `data/stellar_evolution_data.h5`
- `compute_Mrem_Fryer12_delayed_rapster(M, Z, mass_gap_low=45, mass_gap_high=120)`
- `compute_Mrem_SEVN_delayed_rapster(M, Z, mass_gap_low=55, mass_gap_high=120)`
- 2D `RegularGridInterpolator` + Heaviside masking of pair-instability gap.

**natal_kick.py**
- `sample_maxwellian_kick(sigma, n_samples=1, seed)` — σ default ~265 km/s (Hobbs+2005).

---

### `hosts/` — environments & escape velocities

**plummer.py** — `class PlummerCluster(Mcl, r_h, cluster_type=None, epsilon, tau_gyr, rt_over_rh=5)`
- attrs: scale radius `a=r_h/1.305`, `v_esc`, `v_core≈0.54 v_esc`, `r_t`
- `.potential(r)` · `.density(r)` · `.sigma(r)` · `.merger_analysis(v_kick, M_bh)` → retained mask, r_max, t_DF, P_ret/P_core/P_hier
- standalone: `plummer_scale_radius, plummer_potential, plummer_density, plummer_velocity_dispersion,`
  `plummer_escape_speed, plummer_core_speed, plummer_tidal_radius, plummer_apocentre,`
  `chandrasekhar_F, orbit_shape_factor, dynamical_friction_time(... ln_lambda=2.5), retained_mask`

**star_clusters.py**
- `Mcl_rh_to_vescape(Mcl, r_h)` (virial; **used by core/hierarchical**) · `Mcl_rho_to_vescape(Mcl, rho)`
- `sample_star_clusters_mapelli2021(n, cluster_type='GC'|'NSC'|'YSC', ...)` → dict (log-normal M, ρ)
- `sample_star_clusters(n, M_cluster_min, M_cluster_max, r_h_min, r_h_max, Z_min, Z_max, seed)`

**host_retention.py** — env escape-speed distributions: GC, NSC, EG, DG, MW, M31
- `sample_escape_speed(n, environment, seed)` · `sample_multi_escape_speed(n, environments, seed)`
- `escape_speed_cdf(v, environment)` · `retention_curve(v_kick, v_esc_array)` (empirical CDF)
- `compute_environment_retention(v_kick, environment)` → 1-CDF(v_kick)
- `compute_multi_environment_retention(v_kick, environments=None)` → dict per env
- `compute_environment_cumulative_retention(v_kick, environment, method='kde'|'mc', vmax=5000, ngrid=5000)`
- `compute_multi_environment_cumulative_retention(...)`

---

### `core/` — the headline science (returns dicts + ships plotting methods)

**hierarchical.py**
- `class HierarchicalMergersInCluster` — single cluster, fixed or evolving v_esc
  - `__init__(Mcl=None, rh=None, v_esc=None, Z=None, stellar_model=None, n_pool=5000, m_pool=None, m_min=None, m_max=None, imf=None, imf_gamma=-2.5, chi_max=0.2, spin_dist='uniform', beta_a=1.4, beta_b=3.6, evolve_v_esc=False, pairing='random', pairing_beta=None, max_gen=5, n_samples=None, precessing=True, mass_spin_model=None, kick_model=None, seed=None)`
  - escape vel: provide `(Mcl, rh)` OR `v_esc`; `evolve_v_esc=True` requires `(Mcl, rh)`.
  - 1G mass pool priority: `m_pool` → `(m_min, m_max, imf)` → `Z` (stellar collapse).
  - `.simulate(verbose=False)` → dict keyed by generation · `.plot_generations(data, compare_data=None)`
- `class HierarchicalMergersInClusterPopulation` — ensemble; v_esc sampled per merger
  - `__init__(n_samples=5000, chi_max=0.2, m_min=3, m_max=60, imf='uniform', imf_gamma=-2.5, spin_dist='uniform', beta_a=1.4, beta_b=3.6, v_esc_min=1, v_esc_max=300, v_esc_dist='uniform', v_esc_mean=150, v_esc_sigma=45, pairing='random', pairing_beta=None, evolve_v_esc=False, v_esc_decay_index=-0.35, max_gen=5, precessing=True, mass_spin_model=None, kick_model=None, seed=None)`
  - `.simulate()` · `.plot_generations()`
- module helpers: `_get_pair`, `_compute_remnants`, `_scatter_panel`, `_plot_generations_{single,compare}`
- generation convention: `remnant_gen = max(parent_gen1, parent_gen2) + 1`; only kick<v_esc survive.

**seed_growth.py**
- `class MonteCarloBHSeedGrowth` — one seed grows by repeated mergers; chain ends on ejection / target / max_gen
  - `__init__(v_esc=100, Z=0.005, chi_max=0.2, m_seed=10, m_targets=None(→[100,250]), beta=4.0, max_generations=20, precessing=True, mass_spin_model=None, kick_model=None, evolve_v_esc=False, v_esc_decay_index=-0.35, m_pool=None, n_pool=5000, seed=None)`
  - pairing weight P∝(m_seed+m_partner)^beta
  - `.simulate(n_experiments=10000, store_history=False, verbose=False)` → dict (final_masses/spins/gens, retained, reached_target, P_ret, P_target)
  - `.simulate_grid(v_esc_values, n_experiments=10000, verbose=False)` → P_ret(v_esc), reuses pool
- `class IMBHFormationProbability` — 3D grid (seed_mass × chi_max × v_esc) of P_imbh / P_retention
  - `__init__(seed_mass_values, chi_max_values, v_esc_values, m_target=100, n_experiments=10000, **mc_kwargs)`
  - `.compute(verbose=False)` · `.plot_heatmap(v_esc, axes='seed_chi', ...)` · `.plot_heatmap_all_vesc(...)`

**retention.py**
- `class BBHRetentionProbabilityOverChiq` — P_ret grid over (q × chi_max × v_esc) per kick model
  - `__init__(q_values, chi_max_values, v_esc_values, kick_models=None, n_samples=10000, spin_dist='uniform', beta_a=1.4, beta_b=3.6, precessing=True, seed=None)`
  - defaults kick_models: `['hlz','gwmodel']` (prec) / `['hlz','gwmodel_kick_q200']` (nonprec)
  - `.compute()` · `.plot_heatmap(model, v_esc, ...)` · `.plot_heatmap_all_vesc(...)`
- `class BBHRetentionProbabilities` — population-level kicks + P_ret(v_esc) curves
  - `__init__(v_esc_values, kick_models=None, n_samples=5000, q_min=1, q_max=10, q_dist='uniform', q_power=-1, q_beta_a=2, q_beta_b=5, a_min=0, a_max=1, spin_dist='uniform', spin_beta_a=1.4, spin_beta_b=3.6, spin_angles='isotropic', tilt_beta_a=None, tilt_beta_b=None, precessing=True, seed=None)`
  - `.compute()` · `.plot_kicks(bins=50, log=True)` · `.plot_retention()`

---

## End-to-end data flow

```
IMF (Kroupa/Salpeter)
  → ZAMS masses
  → stellar collapse (Fryer12 / SEVN, gap-masked)        [stellar/]
  → 1G BH mass pool
  → pair & build BBHs (+ sampled spins)                  [binaries/bbh, bbh_spins]
  → BBHRemnant → (Mf, af, vkick)                          [binaries/bbh_remnant → gwModels]
  → compare vkick vs host v_esc                           [hosts/]
  → retained remnants re-enter the pool as next-gen BHs   [core/]
  → repeat → hierarchical growth / IMBH formation / retention statistics
```
