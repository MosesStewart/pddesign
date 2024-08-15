from datetime import datetime

# General information about the project.
project = u'RDDesign'
copyright = u'2024, Moses Stewart'
author = u'Moses Stewart'
year = datetime.now().year
source_suffix = '.rst'

version = u'0.0.0'
release = u'0.0.0'
language = 'english'

# -----------------------------------------------------------------------------
# Extensions
# -----------------------------------------------------------------------------
extensions = [
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_design',
]

# -----------------------------------------------------------------------------
# Numpy Theme
# -----------------------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo_dark.svg",
    },
    "github_url": "https://github.com/MosesStewart/pddesign",
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "version-switcher",
        "navbar-icon-links"
    ],
    "navbar_persistent": [],
    "show_version_warning_banner": True,
}

html_title = "%s v%s Manual" % (project, version)
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'
html_css_files = ["design.css"]
html_context = {"default_mode": "light"}
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

htmlhelp_basename = 'rddesign'


mathjax_path = "scipy-mathjax/MathJax.js?config=scipy-mathjax"

plot_html_show_formats = False
plot_html_show_source_link = False

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True

# -----------------------------------------------------------------------------
# Bibliography
# -----------------------------------------------------------------------------
bibtex_bibfiles = ['refs.bib']
