# From https://www.isprs.org/education/benchmarks/UrbanSemLab/img/table_label_contest_image_overview.png
import logging
import os
from pathlib import Path
import shutil

trunk_name = "top_mosaic_09cm_area"

logging_level = logging.INFO
# logging_level = logging.DEBUG
# logging_level = logging.CRITICAL
logging.basicConfig(level=logging_level, format='[%(levelname)s] %(message)s')

logging.debug(f"ls: {os.listdir()}")

ground_truth_areas = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
logging.info(f"ground_truth_areas: {ground_truth_areas}")
logging.info(f"len(ground_truth_areas): {len(ground_truth_areas)}")

test_areas = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]
logging.info(f"test_areas: {test_areas}")
logging.info(f"len(test_areas): {len(test_areas)}")

# exit(1)

dir_top = "../data/vaihingen/top/"
logging.info(f"dir_top == {dir_top}:\n{' '.join(os.listdir(dir_top))}")
dir_ground_truth = "../data/vaihingen/ground_truth/"
logging.info(f"dir_ground_truth == {dir_ground_truth}:\n{' '.join(os.listdir(dir_top))}")

dir_size = min(len(os.listdir(dir_top)), len(os.listdir(dir_ground_truth)))
all_areas = range(1, dir_size + 1)
logging.info(f"Total of {dir_size} areas.")

dir_train_img = f"../data/vaihingen/imgs/"
for area in all_areas:
    filename = f"{trunk_name}{area}.tif"
    filepath = f"{dir_top}/{filename}"
    logging.info(f"filepath ==  {filepath}")
    # shutil.copy(filepath, dir_train_img)

dir_train_mask = Path('./data/vaihingen/masks/')

dir_test_img = Path('./data/vaihingen/testset/imgs/')
dir_test_mask = Path('./data/vaihingen/testset/masks/')
