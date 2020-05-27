import os
import cv2
import pickle
import argparse
import numpy as np

DATA_FOLDER = './data/'
LABEL_FOLDER = './labels/'
FEATURE_FOLDER = './features/'


def save_data(data, path):
    """
    将数据存储为 pickle 文件。
    :param data:
    :param path:
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)

    print('finish writing:', path)
    return


def load_data(path):
    """
    读取 pickle 文件。
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print('finish loading:', path)
    return data


def load_file(path, by_line=False):
    """
    读取 txt 文件。
    :param path:
    :param by_line:
    :return:
    """
    with open(path, 'r') as f:
        if by_line:
            data = f.readlines()
        else:
            data = f.read()

    return data
