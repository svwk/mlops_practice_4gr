"""
Методы работы с данными
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import message_constants as mc
from detail_params import l2_nom, l1_nom, godn, brak, godn_name, brak_name


def print_error(ex, file_path):
    """
    Вывод сообщения об ошибке сохранения файла
    :param ex: объект исключения
    :param file_path:  полный путь файла
    """
    print(f"{mc.FILE_SAVE_ERROR} {file_path}: {ex.args}")


def make_data(count, method='line', noises=0.15, random_state=42):
    """
    Создание набора данных для бинарной классификации
    :param count: количество точек
    :param method: тип набора данных
    :param noises: коэффициент ~ сила шума
    :param random_state: фиксированный сид случайных чисел (для повторяемости)
    :return: три numpy массива значений x, y и класса
    """
    if random_state:
        np.random.RandomState(seed=random_state)

    data_class = np.array([])
    xy = np.array([])

    if method == 'line' or method is None:
        # если выбран метод line (или никакой не задан),
        # то мы создаем набор данных с использованием функции make_classification
        # это будут линейно-разделимые данные
        xy, data_class = make_classification(n_samples=count,  # количество точек
                                             n_features=2,  # количество признаков
                                             n_redundant=0,  # количество бесполезных признаков
                                             n_informative=2,  # количество информативных признаков
                                             n_clusters_per_class=1,  # количество групп точек на класс
                                             class_sep=2,  # количество классов
                                             random_state=random_state  # фиксированный сид случайных чисел
                                             )

    elif method == 'moons':
        # если выбран метод moons,
        # то мы создаем набор данных с использованием функции make_moons
        # это будут данные похожие на Инь-Ян
        xy, data_class = make_moons(n_samples=count,  # количество точек
                                    noise=noises,  # уровень шума
                                    random_state=random_state  # фиксированный сид случайных чисел
                                    )

    elif method == 'circles':
        # если выбран метод circles,
        # то мы создаем набор данных с использованием функции make_circles
        # это будут данные в виде концентрических окружностей
        xy, data_class = make_circles(n_samples=count,  # количество точек
                                      noise=noises,  # уровень шума
                                      factor=0.5,  # соотношение радиусов внутренней и внешней окружности
                                      random_state=random_state  # фиксированный сид случайных чисел
                                      )

    data_class = np.where(data_class == godn, godn_name, brak_name)
    xy[:, 1] = xy[:, 1] * 2 / 10 + l2_nom
    xy[:, 0] = xy[:, 0] * 3 / 10 + l1_nom

    indexes = np.arange(xy.shape[0])
    index_nan_l1 = np.random.choice(indexes, int(xy.shape[0] * 0.03))
    xy[index_nan_l1, 0] = np.nan

    index_nan_l2 = np.random.choice(indexes, int(xy.shape[0] * 0.04))
    xy[index_nan_l2, 1] = np.nan

    index_nan_status = np.random.choice(indexes, int(data_class.shape[0] * 0.02))
    data_class[index_nan_status] = ""

    if np.size(xy) > 2 and xy.shape[1] > 1 and np.size(data_class) > 2:
        return xy[:, 0], xy[:, 1], data_class

    return None


def devide_save(x, y, data_cls, data_path, name, random_state=42):
    """
     Разделение на обучающую и тестовую выборки и
     сохранение данных в файлы
    :param x: numpy массив значений x
    :param y: numpy массив значений y
    :param data_cls:  numpy массив значений класса
    :param data_path:  адрес директории для сохранений файлов
    :param name: название файла для сохранения
    :param random_state: фиксированный сид случайных чисел (для повторяемости)
    :return: Два дата-фрейма с обучающими и тесовыми данными
    """

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, data_cls, test_size=0.3,
                                                                         random_state=random_state)

    df_train = pd.DataFrame({'l1': x_train, 'l2': y_train, 'status': z_train})
    df_test = pd.DataFrame({'l1': x_test, 'l2': y_test, 'status': z_test})

    file_path = ""
    try:
        file_path = data_path + 'train/' + name + '_source.csv'
        df_train.to_csv(file_path, index=False)

    except Exception as ex:
        print_error(ex, file_path)
        return None

    try:
        file_path = data_path + 'test/' + name + '_source.csv'
        df_test.to_csv(file_path, index=False)
        print(f"{mc.SOURCE_SAVED}: {file_path}")

    except Exception as ex:
        print_error(ex, file_path)
        return None

    return df_train, df_test


def transforms(file_name, scaler, to_fit_scaler):
    """
    Предобработка данных и сохранение их в новый файл
    :param file_name: полный путь и имя обрабатываемого файла
    :param scaler: объект StandardScaler, выполняющий стандартизацию
    :param to_fit_scaler: флаг нужно ли обучать об]ект scaler
    """

    df = pd.read_csv(file_name, sep=',')
    pre_count = df.shape[0]

    df = df.drop_duplicates()
    df = df.drop(df[(df.l1.isnull()) | (df.l1 == 0)].index)
    df = df.drop(df[(df.l2.isnull()) | (df.l2 == 0)].index)
    df = df.drop(df[(df.status.isnull()) | (~df.status.isin([godn_name, brak_name]))].index)

    df = df.reset_index(drop=True)

    df['x'] = df['l2'] - l2_nom
    df['y'] = df['l1'] - l1_nom

    num_columns = ['x', 'y']

    if to_fit_scaler:
        scaler.fit(df[num_columns])

    scaled = scaler.transform(df[num_columns])
    scaled = to_polynom(scaled, order=3)

    num_columns = np.hstack([num_columns, ['x2', 'y2', 'x3', 'y3']])

    df_standard = pd.DataFrame(scaled, columns=num_columns)
    df_standard['z'] = df['status'].apply(lambda x: godn if x == godn_name else brak)

    try:
        file_name = file_name.replace("_source", "_stand")
        df_standard.to_csv(file_name, index=False)

        post_count = df.shape[0]
        print(f"{mc.STAND_SAVED}: {file_name}")
        print(f"{mc.STRING_COUNT_PRE}: {pre_count}")
        print(f"{mc.STRING_COUNT_POST}: {post_count}")

        return df_standard

    except Exception as ex:
        print_error(ex, file_name)
        return None


# %% Загрузим, обучим и сохраним модель
def train_and_save_model(x_train, y_train, file_path, file_name):
    """
    Обучение и сохранение модели
    :param x_train: x, y params of dataset
    :param y_train: target
    :param file_path: path to content folder
    :param file_name: name of opened file
    """
    model = LogisticRegression(max_iter=100_000).fit(x_train, y_train)
    file_name = file_name.replace("_stand.csv", "_model.pkl")
    file_path = file_path + file_name

    try:
        pickle.dump(model, open(file_path, 'wb'))
        print(f"{mc.MODEL_SAVED}: {file_path}")
        return model

    except Exception as ex:
        print_error(ex, file_name)
        return None


def to_polynom(x, order=2):
    """
    Преобразование к полиному
    :param x: исходные данные
    :param order: степень полинома
    """
    order_range = range(2, order + 1, 1)
    out = np.copy(x)
    for i in order_range:
        out = np.hstack([out, np.power(x, i)])
    return out
