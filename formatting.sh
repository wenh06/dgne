#!/bin/sh
black ./python/ --extend-exclude .ipynb -v
flake8 ./python/ --count --ignore="C901 E501 W503 E203 E231 E266 F403 F841 W293 W291" --show-source --statistics --exclude=./.*,build,dist
