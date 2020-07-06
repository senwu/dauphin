dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

check:
	isort -c bin/image
	isort -c bin/text
	isort -c dauphin/
	black bin/ --check
	black dauphin/ --check
	flake8 bin/
	flake8 dauphin/

clean:
	pip uninstall -y dauphin
	rm -rf dauphin.egg-info pip-wheel-metadata

format:
	isort -rc bin/image
	isort -rc bin/text
	isort -rc dauphin/
	black bin/
	black dauphin/
	flake8 bin/
	flake8 dauphin/

.PHONY: dev check clean