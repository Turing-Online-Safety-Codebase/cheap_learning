#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Loads Wulczyn et al., (2017) data.
Sourced from https://github.com/ewulczyn/wiki-detox
See https://github.com/ewulczyn/wiki-detox/blob/master/src/figshare/Wikipedia%20Talk%20Data%20-%20Getting%20Started.ipynb
"""

import os
import urllib.request
import pandas as pd
from sklearn.utils import shuffle
from helper_functions import check_dir_exists
from cleaning_functions import clean_text

def download_file(download_url, file_path, file_name, sep = '\t', index_col=None):
    """Loads dataframe from URL download.

    Args:
        download_url (str): URL location of dataset.
        file_path (str): Directory to store dataset.
        file_name (str): File name for dataset.
        sep (str, optional): Separator for dataset. Defaults to '\t'.
        index_col (int, optional): Index column. Defaults to None.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    urllib.request.urlretrieve(download_url, f'{file_path}/{file_name}')
    read_df = pd.read_csv(f'{file_path}/{file_name}', sep = sep, index_col = index_col)
    return read_df


def main():
    SEED = 123
    # set directories
    task = 'binary_abuse'
    path = os.getcwd()
    main_dir = os.path.split(path)[0]
    print(f'Current working directory is: {main_dir}')
    data_dir = f'{main_dir}/data/{task}/raw_data'
    check_dir_exists(data_dir)
    # download data from URL
    comments_url = 'https://ndownloader.figshare.com/files/7554634'
    annotations_url = 'https://ndownloader.figshare.com/files/7554637'
    comments = download_file(comments_url, data_dir, 'attack_annotated_comments.tsv',
                            sep = '\t', index_col = 0)
    annotations = download_file(annotations_url, data_dir, 'attack_annotations.tsv',
                            sep = '\t')
    print(f"Number of entries: {len(annotations['rev_id'].unique())}")
    # labels a comment as an atack if the majority of annoatators did so
    labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
    # join labels and comments
    comments['label'] = labels
    df = comments.copy()
    # remove newline, tab tokens and ==
    df['comment'] = df['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    df['comment'] = df['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
    df['comment'] = df['comment'].apply(lambda x: x.replace("==", ""))
    # rename columns
    df = df.rename(columns = {'comment': 'text'})
    # clean text
    df['text'] = df['text'].apply(clean_text)
    # save main df
    save_dir = f'{main_dir}/data/{task}/clean_data'
    check_dir_exists(save_dir)
    keep_cols = ['text', 'label', 'split']
    save_df = df[keep_cols]
    save_df.to_csv(f'{save_dir}/{task}.csv')
    # save splits
    splits = ['train', 'test', 'dev']
    for s in splits:
        subset_df = save_df[save_df['split']==s]
        print(f"Number of {s} entries: {len(subset_df)}")
        subset_df = shuffle(subset_df, random_state = SEED)
        subset_df.to_csv(f'{save_dir}/{task}_{s}.csv')

if __name__ == '__main__':
    main()
