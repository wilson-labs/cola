project = 'CoLA'
copyright = '2023, Wilson-Labs'
author = 'Marc Finzi and Andres Potapczynski'
language = "en"

import os
import sys
import re
sys.path.insert(0, os.path.abspath('..'))

RE_VERSION = re.compile(r'^__version__ \= \'(\d+\.\d+\.\d+(?:\w+\d+)?)\'$', re.MULTILINE)
PROJECTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTDIR)

def get_release():
    with open(os.path.join(PROJECTDIR, 'cola', '__init__.py')) as f:
        version = re.search(RE_VERSION, f.read())
    assert version is not None, "can't parse __version__ from __init__.py"
    return version.group(1)

release = get_release()


sys.path.append(os.path.abspath('sphinxext'))

extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax',
    'recommonmark',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosectionlabel',
    # 'sphinx_rtd_dark_mode',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', 'notebooks/.ipynb_checkpoints', 'notebooks/colabs/**'
, '**/_**.**']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'objax': ('https://objax.readthedocs.io/en/latest/', None),
}

source_suffix = ['.rst', '.md']
main_doc = 'index'

html_theme = 'sphinx_rtd_theme'
autodoc_default_options = {'autosummary': True}
autodoc_member_order = 'bysource'
# html_theme = 'sphinx_rtd_theme_dark_mode'
# html_static_path = ['_static']
autosummary_generate = True
napolean_use_rtype = False
nbsphinx_execute = 'never'
nbsphinx_codecell_lexer = 'ipython3'
nbsphinx_prolog = r"""
{% set docname = 'docs/notebooks/colabs/' + env.doc2path(env.docname, base=None).split('/')[-1] %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::
        Interactive online version:
        :raw-html:`<a href="https://colab.research.google.com/github/wilson-labs/cola/blob/master/{{ docname }}"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>`

"""