Installation
============

From source
-----------

.. code-block:: bash

   git clone https://github.com/tousifislam/gwGenealogy.git
   cd gwGenealogy
   pip install -e .

Dependencies
------------

Required:

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `h5py <https://www.h5py.org/>`_
* `gwModels <https://github.com/tousifislam/gwModels>`_ — remnant mass, spin, and kick models

Optional:

* `surfinBH <https://github.com/vijayvarma392/surfinBH>`_ — NR surrogate remnant models (NRSur7dq4Remnant, NRSur3dq8Remnant, NRSur7dq4EmriRemnant)

Building the documentation
--------------------------

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme nbsphinx
   cd docs
   make html

The built documentation will be in ``docs/build/html/``.
