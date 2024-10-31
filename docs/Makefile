# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    ?= -W
SPHINXBUILD   ?= sphinx-build -E
PAPER         ?=
DOXYGEN       ?= doxygen
BUILDDIR      = build

LATEX-BW      = $(BUILDDIR)/latex-bw

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
#ALLSPHINXOPTS   = -T --keep-going -d build/doctrees $(PAPEROPT_$(PAPER)) \
                  $(SPHINXOPTS)
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
FILES=

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter

.PHONY: help clean html web htmlhelp latex changes linkcheck \
	dist dist-build gitwash-update version-check html-build latex-build \
	merge-doc show docenv

default: html

all: html pdf pdf-bw

clean:
	-rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html

view:
	open build/html/index.html

pdf:
	$(SPHINXBUILD) -b latex $(SCREENOPTS) $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	sed -i .bak -f process-latex $(BUILDDIR)/latex/$(LATEX-NAME).tex
	rsync -a latex/ $(BUILDDIR)/latex/
	$(MAKE) -C $(BUILDDIR)/latex pdf

pdf-bw:
	$(SPHINXBUILD) -b latex -t black_and_white $(PAPEROPTS) $(ALLSPHINXOPTS) $(LATEX-BW)
	sed -i .bak -f process-latex $(LATEX-BW)/$(LATEX-NAME).tex
	sed -i .bak '/\\setcounter{page}{1}/d' $(LATEX-BW)/sphinxmanual.cls
	rsync -a latex/ $(LATEX-BW)/
	$(MAKE) -C $(LATEX-BW)/ pdf
