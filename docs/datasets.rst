Datasets
========

HIF-Datasets
------------

The HAT.datasets module and ability to load Hypergraph Interchange Format (HIF) files allows HAT to work with a wide array of higher-order datasets.
Several datasets are automatically downloaded from the `HIF-datasets GitHub repository <https://github.com/Jpickard1/HIF-datasets>`_ and cached locally.

Loading Datasets
"""""""""""""""""

The primary method for accessing datasets is through the ``datasets.load()`` function:

.. code-block:: python

    from hat.datasets import load
    
    # Load a dataset - automatically downloads if not cached
    H = load("BioCarta_2013")
    
    # Load with custom data directory
    H = load("KEGG_2018", datapath="./my_datasets")

HAT Dataset Module
------------------

.. automodule:: HAT.datasets
   :members:
   :undoc-members:
   :show-inheritance: