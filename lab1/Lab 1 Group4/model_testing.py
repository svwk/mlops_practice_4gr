#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import pickle
import argparse
import os
import sys

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
        (not os.path.isdir(file_path + 'test/')):
    print("There is no such path")
    sys.exit(1)

for filename in os.listdir(file_path + "test/"):
    if filename.endswith("_stand.csv"):
        full_filename = f"{file_path}test/{filename}"
        df_test = pd.read_csv(full_filename)

        X_test = df_test.drop('z', axis=1)

        model_filename = filename.replace('_stand.csv', '_model.pkl')
        model_file_path = f"{file_path}{model_filename}"
        try:
            with open(model_file_path, 'rb') as model_storage:
                model = pickle.load(model_storage)

            accuracy = model.score(X_test.values, df_test['z'].values)
            print(f'Model test accuracy is: {accuracy:.3f}')
        except Exception as inst:
            print(f"Ошибка чтения модели {model_filename} {inst.args}")
            sys.exit(1)



