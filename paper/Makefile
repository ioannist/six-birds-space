pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	latexpand -o main.flattened.tex main.tex

clean:
	latexmk -C
