gwGenealogy
===========

**gwGenealogy** is a Python toolkit for binary black hole (BBH) phenomenology:
sampling BBH populations, computing remnant properties, modeling hierarchical
mergers in dense star clusters, and analyzing retention across astrophysical
environments.

This package is a culmination of my works into understanding hierarchical
mergers with my amazing collaborators Digvijay Wadekar and Konstantinos Kritos.
Collaborations are always welcome!

Features
--------

* **Distribution samplers** — power-law, beta, uniform, Gaussian, log-uniform, Maxwellian, log-normal via :mod:`gwGenealogy.utils`
* **BBH population sampling** — masses, spins, redshifts, GWTC-3/4/5 models via :mod:`gwGenealogy.binaries`
* **Remnant properties** — mass, spin, and kick velocity for precessing and nonprecessing BBH mergers via :class:`~gwGenealogy.binaries.BBHRemnant`
* **Stellar evolution** — Kroupa IMF, Fryer12/SEVN delayed remnant models, natal kicks via :mod:`gwGenealogy.stellar`
* **Host environments** — Plummer clusters, escape velocities, environment-marginalised retention via :mod:`gwGenealogy.hosts`
* **Hierarchical mergers** — multi-generation merger simulations via :class:`~gwGenealogy.core.HierarchicalMergersInClusters`
* **BH seed growth** — Monte Carlo seed growth chains and IMBH formation probability via :class:`~gwGenealogy.core.MonteCarloBHSeedGrowth`
* **Retention grids** — 1G+1G retention probability heatmaps via :class:`~gwGenealogy.core.BBHRetentionProbability1G1G`

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api/index
   notebooks/index
   citation
