#!/usr/bin/env bash
#!/bin/bash

## From dataset description at https://www.isprs.org/education/benchmarks/UrbanSemLab/semantic-labeling.aspx
# Six categories/classes have been defined:
#  Impervious surfaces (RGB: 255, 255, 255)
#  Building (RGB: 0, 0, 255)
#  Low vegetation (RGB: 0, 255, 255)
#  Tree (RGB: 0, 255, 0)
#  Car (RGB: 255, 255, 0)
#  Clutter/background (RGB: 255, 0, 0)
python train_on_vaihingen.py --amp --classes 6 --epochs 5 --batch-size 64
#python train_on_vaihingen.py --amp --classes 6 --epochs 1 --batch-size 64
