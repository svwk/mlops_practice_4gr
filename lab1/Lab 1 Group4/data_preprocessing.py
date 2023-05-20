#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# %% Импорт
import sys
import argparse
import sklearn
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

scaler = StandardScaler()
def process_file(path, file_name):
    """
    Обработка данных и сохранение их в новый файл
    """
    file_path = ""
    try:
        file_name = os.path.basename(path)
        scaler = StandardScaler()
        with open(path, 'r') as f:
            scaler.fit_transform(f.read())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_path = file_name + 'trans'+'.csv'
        path.to_csv(file_path, index=False)
        return path

    except:
        print(f"Ошибка сохранения файла {file_path}")
        return None
