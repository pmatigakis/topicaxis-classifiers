#!/bin/bash
set -e

python train/categories/basic.py cat__programming ./data/categories/dataset.json
python train/categories/basic.py cat__politics ./data/categories/dataset.json
python train/categories/basic.py cat__business ./data/categories/dataset.json
python train/categories/basic.py cat__science ./data/categories/dataset.json
python train/categories/basic.py cat__technology ./data/categories/dataset.json

python train/categories/bert.py ./data/categories/dataset.json "cat__programming,cat__science,cat__business,cat__politics,cat__technology"
