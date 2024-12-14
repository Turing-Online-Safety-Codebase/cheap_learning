#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Helper functions used across scripts.
"""

import os
import random
import pandas as pd
from sklearn.utils import shuffle


def reduce_path(path, target_dir):
    # Reduce path so that target_dir is the last term
    p = path
    while p.split("/")[-1] != target_dir:
        if "/" not in p:
            raise Exception(f"Dir {target_dir} not found in")
        # cut off last term
        p = "/".join(p.split("/")[:-1])

    return p


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


def load_n_samples(data_dir, task, split, n_entries, seed: int | None = None):
    """Loads first n entries of dataset split.

    Args:
        data_dir (str): Directory with data.
        task (str): Task name e.g. abuse
        split (str): Split from [train, test, dev] to be sampled from.
        n_entries (int): Number of entries to sample.

    Returns:
        pd.DataFrame: Dataset of n rows.
    """
    filename = f"{data_dir}/{task}/clean_data/{task}_{split}.csv"
    if n_entries == -1:
        # load all entries
        df = pd.read_csv(filename)
        df = df[~df.text.isna()]
    else:
        if seed is None:
            # load n_entries
            df = pd.read_csv(filename, nrows=n_entries)
            df = df[~df.text.isna()]
            i = 0
            while len(df) != n_entries and i < 100:
                missing = n_entries - len(df)
                df_dummy = pd.read_csv(filename, skiprows=len(df), nrows=missing)
                df = pd.concat([df, df_dummy])
                df = df[~df.text.isna()]
                i += 1
        else:
            random.seed(seed)
            n = sum(1 for line in open(filename)) - 1
            skip = sorted(random.sample(range(1, n + 1), n - n_entries))
            df = pd.read_csv(filename, skiprows=skip)
            df = df[~df.text.isna()]
            i = 0
            while len(df) != n_entries and i < 100:
                missing = n_entries - len(df)
                df_dummy = pd.read_csv(filename, skiprows=len(df), nrows=missing)
                df = pd.concat([df, df_dummy])
                df = df[~df.text.isna()]
                i += 1
    return df


def load_balanced_n_samples(data_dir, task, split, n_entries, seed: int = 123):
    """Loads balanced first n entries of training dataset split across 2 classes.

    Args:
        data_dir (str): Directory with data.
        task (str): Task name e.g. abuse
        split (str): Split from [train, test, dev] to be sampled from.
        n_entries (int): Number of entries to sample in total.
        SEED (int): Seed for random state
    Returns:
        pd.DataFrame: Dataset of n rows.
    """
    filename = f"{data_dir}/{task}/clean_data/{task}_{split}.csv"
    balanced_n = int(n_entries / 2)
    df = pd.read_csv(filename)
    df = df[~df.text.isna()]
    df_pos = df[df["label"] == True].head(balanced_n)
    df_neg = df[df["label"] == False].head(balanced_n)
    df_concat = pd.concat([df_pos, df_neg])
    shuffled_df = shuffle(df_concat, random_state=seed)
    return shuffled_df


def load_custom_n_samples(
    seed, data_dir, task, split, n_entries, label_balance: int = 0.5
):
    """Loads first n entries of training dataset split across 2 classes, balanced according to passed proportion.

    Args:
        seed (float): Random seed to use when shuffling dataset
        data_dir (str): Directory with data.
        task (str): Task name e.g. abuse
        split (str): Split from [train, test, dev] to be sampled from.
        n_entries (int): Number of entries to sample in total.
        label_balance (float, optional): Number between 0 and 1 representing the proportion of positive labelled entries to include in the sample.
    Returns:
        pd.DataFrame: Dataset of n rows.
    """
    df = pd.read_csv(f"{data_dir}/{task}/clean_data/{task}_{split}.csv")

    # make df out of first n rows, matching passed label balance
    dfn = pd.concat(
        [
            df[df["label"] == True].head(int(n_entries * label_balance)),
            df[df["label"] == False].head(n_entries - int(n_entries * label_balance)),
        ]
    ).reset_index(drop=True)

    # return shuffled result
    return shuffle(dfn, random_state=seed)


def convert_labels(df):
    """Converts string or boolean labels to integers.

    Args:
        df (pd.Dataframe): Input dataframe.

    Returns:
        pd.DataFrame: Output dataframe with label columns.
        int: Number of unique classes.
    """
    # Save label_str value
    df["label_str"] = df["label"]
    # Replace label column with int values
    df["label"] = pd.Categorical(df["label_str"]).codes
    n_classes = len(df["label"].unique())
    return df, n_classes
