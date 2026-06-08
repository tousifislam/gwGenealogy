# gwGenealogy

A Python toolkit for binary black hole (BBH) phenomenology: sampling BBH populations, computing remnant properties, modeling hierarchical mergers in dense star clusters, and analyzing retention across astrophysical environments.

This package is a culmination of my works into understanding hierarchical mergers with my amazing collaborators Digvijay Wadekar and Konstantinos Kritos. Collaborations are always welcome!

## Package structure

```
gwGenealogy/
├── utils/      — Distribution samplers, coordinate transforms, conversions, statistics
├── binaries/   — BBH population sampling (masses, spins, redshifts, GWTC models), BBHRemnant
├── stellar/    — Stellar evolution (Fryer12, SEVN), Kroupa IMF, natal kicks
├── hosts/      — Plummer clusters, star cluster sampling, escape velocities, environment retention
├── core/       — Hierarchical mergers, BH seed growth, IMBH formation probability, retention grids
```

## Installation

```bash
git clone https://github.com/tousifislam/gwGenealogy.git
cd gwGenealogy
pip install -e .
```

### Dependencies

- `numpy`, `scipy`, `matplotlib`, `h5py`
- [`gwModels`](https://github.com/tousifislam/gwModels) — remnant mass, spin, and kick models (required)
- [`surfinBH`](https://github.com/vijayvarma392/surfinBH) — NR surrogate remnant models (optional)

## Quick start

### Sample a BBH population and compute remnants

```python
from gwGenealogy.binaries import BBHs, BBHRemnant, sample_masses, sample_spins

m1, m2 = sample_masses(5000, m_min=5, m_max=50, m1_distribution='powerlaw', alpha=-2.35, seed=42)
chi1, chi2 = sample_spins(5000, chi_max=1.0, spin_magnitude='beta', spin_angles='isotropic', seed=43)

bbh = BBHs(m1=m1, m2=m2, ...)
rem = BBHRemnant(bbh=bbh, precessing=True)
# rem.Mf, rem.af, rem.vkick
```

### Hierarchical mergers across generations

```python
from gwGenealogy.core import HierarchicalMergersInClusters

sim = HierarchicalMergersInClusters(n_samples=50000, chi_max=0.2, m_min=3, m_max=60,
                                     kick_model='gwmodel', seed=42)
data = sim.simulate(verbose=True)
fig, axes = sim.plot_generations(data)
```

### BH seed growth and IMBH formation

```python
from gwGenealogy.core import MonteCarloBHSeedGrowth

mc = MonteCarloBHSeedGrowth(v_esc=300, Z=0.005, chi_max=0.2, m_seed=10.0,
                             m_targets=[250], kick_model='gwmodel', seed=42)
result = mc.simulate(n_experiments=10000, verbose=True)
```

### Retention probability grids

```python
from gwGenealogy.core import BBHRetentionProbability1G1G
import numpy as np

grid = BBHRetentionProbability1G1G(
    q_values=np.linspace(1, 10, 50),
    chi_max_values=np.linspace(0.01, 1, 50),
    v_esc_values=[50, 100, 200, 500],
    kick_models=['hlz', 'gwmodel'], seed=42)
grid.compute(verbose=True)
fig, axes = grid.plot_heatmap_all_vesc()
```

### GWTC population sampling

```python
from gwGenealogy.binaries import sample_gwtc_population, available_catalogs

available_catalogs()
pop = sample_gwtc_population(50000, catalog='gwtc5', source='posterior', seed=42)
# pop['mass_1'], pop['a_1'], pop['chi_eff'], pop['redshift'], ...
```

### Environment retention

```python
from gwGenealogy.hosts import compute_multi_environment_retention, ENVIRONMENTS

ret = compute_multi_environment_retention(v_kick_array)
# ret['GC'], ret['NSC'], ret['EG'], ret['DG'], ret['MW'], ret['M31']
```

## Tutorials

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Distributions](tutorials/01_distributions.ipynb) | Core distribution samplers and JSD |
| 02 | [Stellar evolution](tutorials/02_stellar_evolution.ipynb) | Kroupa IMF, Fryer12/SEVN remnants, natal kicks |
| 03 | [BBH nonprecessing](tutorials/03_bbh_nonprecessing.ipynb) | BBHs + BBHRemnant with nonprecessing models |
| 04 | [BBH precessing](tutorials/04_bbh_precessing.ipynb) | BBHs + BBHRemnant with precessing and surrogate models |
| 05 | [GWTC populations](tutorials/05_gwtc_populations.ipynb) | GWTC-3/4/5 population sampling, posterior vs prior |
| 06 | [Retention probability](tutorials/06_retention_probability.ipynb) | 1G+1G retention heatmaps across kick models |
| 07 | [Host environments](tutorials/07_host_environments.ipynb) | Plummer clusters, escape velocities, environment retention |
| 08 | [Hierarchical mergers](tutorials/08_hierarchical_mergers.ipynb) | Multi-generation mergers, pairing models, evolving v_esc |
| 09 | [Seed growth](tutorials/09_seed_growth.ipynb) | BH seed growth chains, v_esc sweeps, evolving v_esc |

## Citations

If you use this package, please cite the relevant papers:

```bibtex
@article{Islam:2025drw,
    author = "Islam, Tousif and Wadekar, Digvijay",
    title = "{Accurate models for recoil velocity distribution in black hole
              mergers with comparable to extreme mass-ratios and their
              astrophysical implications}",
    eprint = "2511.11536",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/4jvv-qg4h",
    journal = "Phys. Rev. D",
    volume = "113",
    number = "10",
    pages = "104017",
    year = "2026"
}

@article{Islam:2026yxx,
    author = "Islam, Tousif and Wadekar, Digvijay and Kritos, Konstantinos",
    title = "{Kick matters: The impact of a new recoil model on the retention
              of hierarchical black-hole remnants in globular clusters}",
    eprint = "2603.10170",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "3",
    year = "2026"
}

@article{Islam:2026iyn,
    author = "Islam, Tousif",
    title = "{Inference of recoil kicks from binary black hole mergers up to
              GWTC--4 and their astrophysical implications}",
    eprint = "2604.04546",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "4",
    year = "2026"
}

@article{Islam:2026sjl,
    author = "Islam, Tousif and Venumadhav, Tejaswi and Wadekar, Digvijay",
    title = "{Progenitor of the recoiling super-massive black hole RBH-1
              identified using HST/JWST imaging}",
    eprint = "2601.18986",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "1",
    year = "2026"
}
```

## Contact

For questions or issues, please reach out to tousifislam24@gmail.com.

## License

MIT
