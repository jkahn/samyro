#!/bin/bash
check-manifest --ignore tox.ini,tests*
python setup.py check -m -r -s
flake8 --exclude=.tox,*.egg,build,data .
py.test tests
