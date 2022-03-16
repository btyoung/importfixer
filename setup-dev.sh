#! /usr/bin/env sh
pip install --upgrade pip
pip install -r requirements-dev.txt
pre-commit install
pip install -e ./
