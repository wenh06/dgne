#!/bin/sh
black ./python/ --extend-exclude .ipynb -v
flake8 ./python/ --count --ignore="E501 W503 E203 F841" --show-source --statistics --exclude=./.*,build,dist
