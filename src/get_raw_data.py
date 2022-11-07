#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Loads raw data from Vicomtech/hate-speech-dataset.
"""

import os
import requests
import pandas as pd
from helper_functions import check_dir_exists

def load_data(url_metadata):
    """Loads raw data from Github URL download.

    Args:
        url_metadata (str): URL to github data.

    Returns:
        pd.DataFrame: metadata details of each entry.
    """
    metadata_df = pd.read_csv(url_metadata, sep=",")
    metadata_df['label'] = metadata_df['label'].map({'noHate': False, 'hate': True})
    metadata_df = metadata_df.rename(columns={"label": "hate_binary"})
    print(f'Number of entries: {len(metadata_df)}')
    return metadata_df

def get_text(metadata_df, file_url_train, file_url_all):
    """Appends text for each metadata entry.

    Args:
        metadata_df (pd.DataFrame): dataframe of metadata.
        file_url_train (str): URL to folder with train texts.
        file_url_all (str): URL to folder with all texts.

    Returns:
        pd.DataFrame: metadata details of each entry with appended text.
    """
    train_test_bool  = []
    text_list = []
    for i, item in enumerate(metadata_df['file_id']):
        if i%1000 == 0:
            print(f"{i}/{len(metadata_df)} completed")
        # checking if they are in train
        file_url_train = file_url_train + str(item) + ".txt"
        request_train = requests.get(file_url_train)
        if request_train.ok:
            train = "True"
            train_test_bool.append(train)
        else:
            not_train = "False"
            train_test_bool.append(not_train)
        # get text
        file_url =  file_url_all + str(item) + ".txt"
        r = requests.get(file_url)
        text = r.text
        text_list.append(text)
    # append new columns
    metadata_df['train_binary'] = train_test_bool
    metadata_df['text'] = text_list
    return metadata_df

def main():
    base_url = "https://raw.githubusercontent.com/Vicomtech/hate-speech-dataset/master/"
    url_metadata = f"{base_url}/annotations_metadata.csv"
    file_url_train = f"{base_url}/sampled_train/"
    file_url_all = f"{base_url}/all_files/"
    # load data
    metadata_df = load_data(url_metadata)
    # append texts
    metadata_text_df = get_text(metadata_df, file_url_train, file_url_all)
    # save
    path = os.getcwd()
    DIR, _ = os.path.split(path)
    save_path = f'{DIR}/data/raw_data'
    # check save dir exists
    check_dir_exists(save_path)
    metadata_text_df.to_csv(f"{save_path}/metadata_w_text.csv", index=False)

if __name__ == '__main__':
    main()
