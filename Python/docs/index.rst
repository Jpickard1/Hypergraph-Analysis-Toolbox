.. Hypergraph Analysis Toolbox documentation master file, created by
   sphinx-quickstart on Mon Oct  3 14:52:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hypergraph Analysis Toolbox
===========================

.. image:: _static/index_dyadic_decomp.png
    :align: center

*Important Note*
****************
The software for HAT is complete, but the online documentation is a work in progress. Currently, the software is only accessable via git and the Mathworks file exchange, but it will be published via PiPy shortly.

Introduction
============
Hypergraph Analysis Toolbox (HAT) is a software suite for the analysis and visualization of hypergraphs and
higher order structures. Motivated to investigate Pore-C data, HAT is intended as a general prupose, versatile
software for hypergraph construction, visualization, and analysis. HAT addresses the following hypergraph
problems:

1. Construction
2. Visualization
3. Expansion and numeric representation
4. Structral Properties
5. Controllability
6. Similarity Measures

The capabilities and use cases of HAT are outlined in `this short notice <https://drive.google.com/file/d/1Mx8ifUtjR05ufhTXc5QgYKlwmfQjfItv/view?usp=share_link>`_. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials.rst
   HAT.rst
   ref.rst

Installation
============

MATLAB
******
The MATLAB distribution of HAT can be installed through either the `MATLAB Central <https://www.mathworks.com/matlabcentral/fileexchange/121013-hypergraph-analysis-toolbox>`_. A MathWorks :code:`.mltbx` file can be downloaded from the site,
and installed through the add on manager in the MATLAB Home environment. Once installed as a toolbox, you will have access to all HAT functionality.

The MATLAB distribution has the following dependencies:

1. `TenEig — Tensor Eigenpairs Solver <https://users.math.msu.edu/users/chenlipi/teneig.html>`_

Python
******
The Python distribution of HAT may be installed through pip:

.. code-block:: Python

   >> pip install HypergraphAnalysisToolbox

Information on the PiPy distribution is available `here <https://pypi.org/project/HypergraphAnalysisToolbox/>`_. Once installed, HAT may be imported into the Python invironment with the command:

.. code-block:: Python

   import HAT              # Import package
   import HAT.Hypergraph   # Hypergraph class
   import HAT.plot         # Visualization tools

The Python distribution has the following dependencies:

1. numpy
2. scipy
3. matplotlib
4. itertools

Development Distribution
************************
All implementations of HAT are managed through a `common git repository <https://github.com/Jpickard1/Hypergraph-Analysis-Toolbox>`_. This is public, so it may be
cloned and modified. If interested in modifying or contributing to HAT, please contact Joshua Pickard at jpic@umich.edu

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
