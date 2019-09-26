# Evan Widloski - 2019-03-04
# makefile for building/testing python packages

# run all lines in target in single shell, quit on error
.ONESHELL:
.SHELLFLAGS = -ec

version := $(shell python -c "exec(open('version.py').read());print(__version__)")

.PHONY: dist
dist:
	python setup.py sdist

.PHONY: pypi
pypi: dist man
	twine upload dist/mas-$(version).tar.gz
