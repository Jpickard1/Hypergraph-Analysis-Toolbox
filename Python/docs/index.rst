.. Hypergraph Analysis Toolbox documentation master file, created by
   sphinx-quickstart on Mon Oct  3 14:52:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Hypergraph Analysis Toolbox's documentation!
=======================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   properties.rst
   visualization.rst
   decompositions.rst
   similarity.rst
   control.rst

Introduction
============
Hypergraph Analysis Tolbox (HAT) is a software suite for analyzing higher order structures. HAT is implemented
in Python and Matlab. There are 5 main modules of HAT including:

1. Visualization
2. Hypergraph Properties
3. Controlability
4. Similarity Measures
5. Decompositions

Motivation
**********
HAT is intended as a general purpose, multiplatform software for studying hypergraphs.

Python
******
The Python implementation of HAT may be installed through pip:
.. code-block:: Python

   >> pip install HypergraphAnalysisPackage

HAT may be imported to the Python invironment with the command:

.. code-block:: Python

   import HAT              # Import package
   import HAT.hypergraph   # hypergraph class
   import HAT.graph        # graph class
   import HAT.plot         # visualization tools

Development
***********
All implementations of HAT are managed through a common git repository available at: https://github.com/Jpickard1/Hypergraph-Analysis-Toolbox

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
