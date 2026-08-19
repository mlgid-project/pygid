import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'pygid'
author = 'Ainur Abukaev'
release = '0.2.15'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    # 'nbsphinx',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

tutorial_dir = os.path.join(os.path.dirname(__file__), 'tutorials')
tutorials = sorted(f for f in os.listdir(tutorial_dir) if f.endswith('.ipynb'))
tutorial_names = [os.path.splitext(f)[0] for f in tutorials]

toctree_path = os.path.join(os.path.dirname(__file__), 'tutorials_toctree.rst')
with open(toctree_path, 'w', encoding='utf-8') as f:
    f.write("Tutorials\n=========\n\n.. toctree::\n   :maxdepth: 2\n\n")
    for name in tutorial_names:
        f.write(f"   tutorials/{name}\n")
