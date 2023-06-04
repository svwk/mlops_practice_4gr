#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# %% Импорт
import argparse
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler

import message_constants as mc
from data_methods import transforms
from plot_data import plot_data

# %% Задание пути для сохранения файлов
file_path = os.getcwd()
dataset_name = 'moons'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir')
    parser.add_argument('-t', '--type')
    namespace = parser.parse_args()

    if namespace.dir:
        file_path = namespace.dir

    if namespace.type:
        dataset_name = namespace.type

if not file_path.endswith("/"):
    file_path = f"{file_path}/"

# %% Если пути не существует, скрипт прерывает свою работу
if (not os.path.isdir(file_path)) or \
        (not os.path.isdir(file_path + 'train/')) or \
        (not os.path.isdir(file_path + 'test/')):
    print(mc.NO_PATH)
    sys.exit(1)

show_plots = False

for filename in os.listdir(file_path + "train/"):
    if filename.endswith("_source.csv"):
        full_filename = f"{file_path}train/{filename}"
        scaler_filename = f"{file_path}{filename.replace('_source.csv', '_scaler.pkl')}"
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        dataset_data = transforms(full_filename, scaler, True)
        if dataset_data is None:
            sys.exit(3)
        joblib.dump(scaler, scaler_filename)

        if show_plots:
            plot_data(dataset_data, dataset_name + "_train")

for filename in os.listdir(file_path + "test/"):
    if filename.endswith("_source.csv"):
        full_filename = f"{file_path}test/{filename}"
        scaler_filename = f"{file_path}{filename.replace('_source.csv', '_scaler.pkl')}"
        scaler = joblib.load(scaler_filename)
        dataset_data = transforms(full_filename, scaler, False)
        if dataset_data is None:
            sys.exit(4)

        if show_plots:
            plot_data(dataset_data, dataset_name + "_test")
