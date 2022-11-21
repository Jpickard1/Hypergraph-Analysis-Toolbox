.. Hypergraph Analysis Toolbox documentation master file, created by
   sphinx-quickstart on Mon Oct  3 14:52:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hypergraph Analysis Toolbox
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   properties.rst
   visualization.rst
   decompositions.rst
   similarity.rst
   control.rst
   HAT.rst
   modules.rst

*Important Note*
****************
The software for HAT is complete, but the online documentation is a work in progress.

Introduction
============
Hypergraph Analysis Tolbox (HAT) is a software suite for the analysis and visualization of hypergraphs and
higher order structures. Motivated to investigate Pore-C data, HAT is intended as a general prupose, versatile
software for hypergraph construction, visualization, and analysis. HAT addresses the following hypergraph
problems:

1. Construction
2. Visualization
3. Expansion and numeric representation
4. Structral Properties
5. Controllability
6. Similarity Measures

Instillation
============

MATLAB
******
The MATLAB distribution of HAT can be installed through either the `MathWorks file exchange <https://www.mathworks.com/matlabcentral/fileexchange/>`
or this `link here <https://rajapakse.lab.medicine.umich.edu/software>`. Both links provide a MathWorks .mltbx which is installed through the add on manager
in the MATLAB Home environment. Once installed as a toolbox, you will have access to all HAT functionality.

The MATLAB distribution has the following dependencies:
1. 

Python
******
The Python distribution of HAT may be installed through pip:

.. code-block:: Python

   >> pip install HypergraphAnalysisToolbox

Once installed, HAT may be imported to the Python invironment with the command:

.. code-block:: Python

   import HAT              # Import package
   import HAT.Hypergraph   # Hypergraph class
   import HAT.plot         # Visualization tools

Development
***********
All implementations of HAT are managed through a common git repository available at: https://github.com/Jpickard1/Hypergraph-Analysis-Toolbox

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
