import os
import sys
sys.path.insert(0, os.path.abspath('../HAT/'))
sys.path.insert(0, os.path.abspath('../Python/HAT/'))
sys.path.insert(0, os.path.abspath('../Python/'))
sys.path.insert(0, os.path.abspath('..'))
print(f"{sys.path=}")

import mock
 
MOCK_MODULES = ['numpy',
                'scipy',
                'scipy.io',
                'scipy.spatial'
                'matplotlib',
                'matplotlib.pyplot',
                'scipy.linalg',
                'networkx',
                'pandas',
                'pd',
                'np',
                'sp',
                'mock'
               ]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()



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

# Napoleon settings for NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_member_order = 'bysource'   # Order documentation by order of code in file

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
