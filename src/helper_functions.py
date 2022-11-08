#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Helper functions used across scripts.
"""

import os
import pandas as pd

def check_dir_exists(path):
    """Checks if folder directory already exists, else makes directory.
    Args:
        path (str): folder path for saving.
    """
    is_exist = os.path.exists(path)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Creating {path} folder")
    else:
        print(f"Folder exists: {path}")

def load_n_samples(data_dir, task, split, n_entries):
    """Loads first n entries of dataset split.

    Args:
        data_dir (str): Directory with data.
        task (str): Task name e.g. abuse
        split (str): Split from [train, test, dev] to be sampled from.
        n_entries (int): Number of entries to sample.

    Returns:
        pd.DataFrame: Dataset of n rows.
    """
    df = pd.read_csv(f'{data_dir}/{task}/clean_data/{task}_{split}.csv', nrows = n_entries)
    return df
