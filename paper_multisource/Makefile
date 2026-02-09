.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib figures/*.png figures/*.pdf
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
