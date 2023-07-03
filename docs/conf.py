project = 'CoLA'
copyright = '2023, Wilson-Labs'
author = 'Marc Finzi and Andres Potapczynski'
release = '0.1.0'
language = "en"
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'recommonmark',
    'sphinx.ext.intersphinx',
    # 'sphinx_rtd_dark_mode',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', 'notebooks/.ipynb_checkpoints'
]

source_suffix = ['.rst', '.ipynb', '.md']

html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_rtd_theme_dark_mode'
html_static_path = ['_static']
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