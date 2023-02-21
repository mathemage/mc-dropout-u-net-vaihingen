# From https://www.isprs.org/education/benchmarks/UrbanSemLab/img/table_label_contest_image_overview.png
import logging
import os
from pathlib import Path
import shutil

logging_level = logging.INFO
# logging_level = logging.DEBUG
# logging_level = logging.CRITICAL
logging.basicConfig(level=logging_level, format='[%(levelname)s] %(message)s')

ground_truth_areas = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
allowed_areas = tuple([str(area) for area in ground_truth_areas])
logging.debug(f"ground_truth_areas: {ground_truth_areas}")
logging.info(f"len(ground_truth_areas): {len(ground_truth_areas)}")

logging.debug(f"ls: {os.listdir()}")

dir_top = "../data/vaihingen/top/"
logging.info(f"dir_top == {dir_top}")
dir_ground_truth = Path('./data/vaihingen/ground_truth/')

dir_train_img = f"../data/vaihingen/imgs/"
for filename in os.listdir(dir_top):
    name_without_extension = os.path.splitext(filename)[0]
    logging.debug(f"name_without_extension == {name_without_extension}")
    if name_without_extension.endswith(allowed_areas):
        filepath = f"{dir_top}/{filename}"
        logging.info(f"filepath ==  {filepath}")
        shutil.copy(filepath, dir_train_img)

dir_train_mask = Path('./data/vaihingen/masks/')

dir_test_img = Path('./data/vaihingen/testset/imgs/')
dir_test_mask = Path('./data/vaihingen/testset/masks/')
