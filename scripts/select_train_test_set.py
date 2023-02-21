# From https://www.isprs.org/education/benchmarks/UrbanSemLab/img/table_label_contest_image_overview.png
from pathlib import Path

ground_truth_areas = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]

dir_top = Path('./data/vaihingen/top/')
dir_ground_truth = Path('./data/vaihingen/ground_truth/')

dir_train_img = Path('./data/vaihingen/imgs/')
dir_train_mask = Path('./data/vaihingen/masks/')

dir_test_img = Path('./data/vaihingen/testset/imgs/')
dir_test_mask = Path('./data/vaihingen/testset/masks/')
