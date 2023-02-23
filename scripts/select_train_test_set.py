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

dir_top = "../data/vaihingen/top/"
logging.info(f"dir_top == {dir_top}:\n{' '.join(os.listdir(dir_top))}")
dir_ground_truth = "../data/vaihingen/ground_truth/"
logging.info(f"dir_ground_truth == {dir_ground_truth}:\n{' '.join(os.listdir(dir_top))}")

named_areas = {"trainset": ground_truth_areas, "testset": test_areas}
for set_name, area_list in named_areas.items():
    logging.info(f"set_name: {set_name}")
    logging.info(f"area_list: {area_list}")

    for area in area_list:
        dir_inputs = f"../data/vaihingen/{set_name}/imgs/"
        filename = f"{trunk_name}{area}.tif"
        filepath = f"{dir_top}/{filename}"
        logging.debug(f"filepath == {filepath}")
        shutil.copy(filepath, dir_inputs)

        dir_targets = f"../data/vaihingen/{set_name}/masks/"
        source_filepath = f"{dir_ground_truth}/{filename}"
        logging.debug(f"source_filepath == {source_filepath}")
        destination_filename = f"{trunk_name}{area}_mask.tif"
        destination_filepath = f"{dir_targets}/{destination_filename}"
        logging.debug(f"destination_filename == {destination_filename}")
        shutil.copy(source_filepath, destination_filepath)

    logging.info(f"dir_inputs == {dir_inputs}:\n{' '.join(os.listdir(dir_inputs))}")
    logging.info(f"dir_targets == {dir_targets}:\n{' '.join(os.listdir(dir_targets))}")
