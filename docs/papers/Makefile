.ONESHELL:

all:
	latex csbs
	bibtex csbs
	latex csbs
	pdflatex csbs.tex

watch:
	while :;
	do
		inotifywait -r -e modify *
		echo "Detected modification.  Rebuilding..."
		latex csbs
		bibtex csbs
		latex csbs
		pdflatex csbs.tex
	done
