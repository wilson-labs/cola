project = 'CoLA'
copyright = '2023, Wilson-Labs'
author = 'Marc Finzi and Andres Potapczynski'
language = "en"

import io
import os
import sys
import re


sys.path.insert(0, os.path.abspath('..'))

RE_VERSION = re.compile(r'^__version__ \= \'(\d+\.\d+\.\d+(?:\w+\d+)?)\'$', re.MULTILINE)
PROJECTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECTDIR)


# Get version from setuptools_scm file
def find_version(*file_paths):
    try:
        with io.open(os.path.join(PROJECTDIR, 'cola', *file_paths), encoding="utf8") as fp:
            version_file = fp.read()
            pattern = r"^__version__ = version = ['\"]([^'\"]*)['\"]"
        version_match = re.search(pattern, version_file, re.M)
        return version_match.group(1)
    except Exception:
        return None


release = find_version()

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
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

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
