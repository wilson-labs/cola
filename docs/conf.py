project = 'CoLA'
copyright = '2023, Wilson-Labs'
author = 'Marc Finzi and Andres Potapczynski'
release = '0.1.0'
language = "en"
extensions = [
    'sphinx.ext.autodoc',
    'nbsphinx',
    'recommonmark',
    # 'sphinx_rtd_dark_mode',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', 'notebooks/.ipynb_checkpoints'
]

source_suffix = ['.rst', '.md']

html_theme = 'sphinx_rtd_theme'
# html_theme = 'sphinx_rtd_theme_dark_mode'
html_static_path = ['_static']
