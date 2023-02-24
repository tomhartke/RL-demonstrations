pc:
	pre-commit run --all

test:
	pytest

build:
	python3 -m build

deploy: build
	twine upload --skip-existing dist/*
