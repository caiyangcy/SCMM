.PHONY: all clean
.SECONDARY: report.pdf

all:	pre-build report.pdf

pre-build:
	cd figs && make

report.pdf:	report.bbl report.tex report.brf
	pdflatex report.tex
	pdflatex report.tex

report.bbl:	report.aux
	bibtex report.aux

report.brf:	report.bbl figs
	pdflatex report.tex

report.aux:	report.bib figs
	pdflatex report.tex

count:	
	detex report.tex | wc

clean:	
	cd figs && make clean
	rm *.aux *.bbl *.blg *.brf *.idx *.lof *.log *.lot *.out *.toc


