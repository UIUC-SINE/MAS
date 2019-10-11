.ONESHELL:
.SILENT:
SHELL := /bin/bash
# .SHELLFLAGS := -ec

TEX_SOURCE := ICASSP.tex
BIB_SOURCE := foo

all:
	pdflatex -halt-on-error ${TEX_SOURCE}

	evince $(shell basename ${TEX_SOURCE} .tex).pdf &

	while true;
	do
		inotifywait -e modify ${TEX_SOURCE}
		pdflatex -halt-on-error ${TEX_SOURCE}
	done
