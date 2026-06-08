Quick Start
===========

This guide shows the basic usage of ``gwGenealogy`` for BBH population
studies and hierarchical merger simulations.

Sample a BBH population
------------------------

.. code-block:: python

   from gwGenealogy.binaries import BBHs, BBHRemnant, sample_masses, sample_spins
   import numpy as np

   # Power-law primary masses, isotropic Beta-distributed spins
   m1, m2 = sample_masses(5000, m_min=5, m_max=50,
                           m1_distribution='powerlaw', alpha=-2.35, seed=42)
   chi1, chi2 = sample_spins(5000, chi_max=1.0, spin_magnitude='beta',
                              spin_angles='isotropic', seed=43)

   # Build BBH container from Cartesian spin vectors
   a1 = np.linalg.norm(chi1, axis=1)
   a2 = np.linalg.norm(chi2, axis=1)
   bbh = BBHs(m1=m1, m2=m2, a1=a1, a2=a2,
              theta1=np.arccos(chi1[:, 2] / a1),
              theta2=np.arccos(chi2[:, 2] / a2),
              phi1=np.arctan2(chi1[:, 1], chi1[:, 0]) % (2 * np.pi),
              phi2=np.arctan2(chi2[:, 1], chi2[:, 0]) % (2 * np.pi))
   print(bbh)

Compute remnant properties
--------------------------

.. code-block:: python

   # Precessing remnants (default: HBR mass/spin + gwmodel kick)
   rem = BBHRemnant(bbh=bbh, precessing=True)
   print(rem)
   # rem.Mf — remnant mass
   # rem.af — remnant spin
   # rem.vkick — kick velocity [km/s]

   # Nonprecessing remnants
   rem_np = BBHRemnant(bbh=bbh, precessing=False)

Hierarchical mergers
--------------------

.. code-block:: python

   from gwGenealogy.core import HierarchicalMergersInClusters

   sim = HierarchicalMergersInClusters(
       n_samples=50000, chi_max=0.2, m_min=3, m_max=60,
       kick_model='gwmodel', seed=42)
   data = sim.simulate(verbose=True)

   # Paper-style mass-spin scatter + mass histogram
   fig, axes = sim.plot_generations(data)

BH seed growth
--------------

.. code-block:: python

   from gwGenealogy.core import MonteCarloBHSeedGrowth

   mc = MonteCarloBHSeedGrowth(
       v_esc=300, Z=0.005, chi_max=0.2, m_seed=10.0,
       m_targets=[250], kick_model='gwmodel', seed=42)
   result = mc.simulate(n_experiments=10000, verbose=True)
   print(f"P_ret = {result['P_ret']:.3f}")

   # Sweep over escape velocities
   import numpy as np
   grid = mc.simulate_grid(np.linspace(50, 500, 20), n_experiments=1000)

Retention probability grids
---------------------------

.. code-block:: python

   from gwGenealogy.core import BBHRetentionProbability1G1G
   import numpy as np

   grid = BBHRetentionProbability1G1G(
       q_values=np.linspace(1, 10, 50),
       chi_max_values=np.linspace(0.01, 1, 50),
       v_esc_values=[50, 100, 200, 500],
       kick_models=['hlz', 'gwmodel'], seed=42)
   grid.compute(verbose=True)
   fig, axes = grid.plot_heatmap_all_vesc()

GWTC population sampling
-------------------------

.. code-block:: python

   from gwGenealogy.binaries import sample_gwtc_population, available_catalogs

   available_catalogs()
   pop = sample_gwtc_population(50000, catalog='gwtc5',
                                 source='posterior', seed=42)
   # pop['mass_1'], pop['a_1'], pop['chi_eff'], pop['redshift'], ...

Host environment retention
--------------------------

.. code-block:: python

   from gwGenealogy.hosts import (compute_multi_environment_retention,
                                   ENVIRONMENTS, PlummerCluster)

   # Environment-marginalised retention
   ret = compute_multi_environment_retention(v_kick_array)
   # ret['GC'], ret['NSC'], ret['EG'], ret['DG'], ret['MW'], ret['M31']

   # Plummer cluster model
   cluster = PlummerCluster(Mcl=1e6, r_h=3.0, cluster_type='GC')
   cluster.merger_analysis(v_kick, M_bh=30.0)
   print(f"P_ret = {cluster.P_ret:.3f}")
