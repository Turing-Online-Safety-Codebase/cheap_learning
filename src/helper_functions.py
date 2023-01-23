#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Helper functions used across scripts.
"""

import os
import pandas as pd
from sklearn.utils import shuffle

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
    if n_entries == -1:
        # load all entries
        df = pd.read_csv(f'{data_dir}/{task}/clean_data/{task}_{split}.csv')
    else:
        # load n_entries
        df = pd.read_csv(f'{data_dir}/{task}/clean_data/{task}_{split}.csv', nrows = n_entries)
    return df

def load_balanced_n_samples(data_dir, task, split, n_entries):
    """Loads balanced first n entries of training dataset split across 2 classes.

    Args:
        data_dir (str): Directory with data.
        task (str): Task name e.g. abuse
        split (str): Split from [train, test, dev] to be sampled from.
        n_entries (int): Number of entries to sample in total.
    Returns:
        pd.DataFrame: Dataset of n rows.
    """
    SEED = 123
    balanced_n = int(n_entries/2)
    df = pd.read_csv(f'{data_dir}/{task}/clean_data/{task}_{split}.csv')
    df_pos = df[df['label'] == True].head(balanced_n)
    df_neg = df[df['label'] == False].head(balanced_n)
    df_concat = pd.concat([df_pos, df_neg])
    shuffled_df = shuffle(df_concat, random_state = SEED)
    return shuffled_df

def convert_labels(df):
    """Converts string or boolean labels to integers.

    Args:
        df (pd.Dataframe): Input dataframe.

    Returns:
        pd.DataFrame: Output dataframe with label columns.
        int: Number of unique classes.
    """
    # Save label_str value
    df['label_str'] = df['label']
    # Replace label column with int values
    df['label'] = pd.Categorical(df['label_str']).codes
    n_classes = len(df['label'].unique())
    return df, n_classes
