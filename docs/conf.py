import os
import sys
sys.path.insert(0, os.path.abspath('../HAT/'))
sys.path.insert(0, os.path.abspath('..'))

import mock
 
MOCK_MODULES = ['numpy',
                'scipy',
                'scipy.io',
                'matplotlib',
                'matplotlib.pyplot',
                'scipy.linalg',
                'networkx',
                'pandas',
                'pd',
                'np'
               ]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# import HAT.HAT
# import HAT.Hypergraph
# import HAT.multilinalg
# import HAT.draw
# import Hypergraph
# from HAT.HAT import *

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Hypergraph Analysis Toolbox'
copyright = '2022, Joshua Pickard'
author = 'Joshua Pickard'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_member_order = 'bysource'   # Order documentation by order of code in file

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
