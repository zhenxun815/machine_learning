#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: file_utils.py
# @Project: machine_learning
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 9/8/2019 12:09
import os


def get_filename(file_path, parser=None):
    """
    get a specify name from a file path
    :param file_path:
    :param parser: function to deal with the file name
    :return:
    """
    base_name = os.path.basename(file_path).split(sep='.')[0]
    return base_name if parser is None else parser(base_name)


if __name__ == '__main__':
    print(f'{get_filename("tt.txt")}')
