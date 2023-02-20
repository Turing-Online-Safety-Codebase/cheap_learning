#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Loads full dev set from file and creates a stratified 10% sample for technique development.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# def stratified_sample_df(df, col, n_samples, SEED):
#     """Takes stratified sample of dataframe.

#     Args:
#         df (pd.DataFrame): input dataframe.
#         col (str): column of labels.
#         n_samples (int): number of entries to sample.

#     Returns:
#         pd.DataFrame: output dataframe of sampled entries.
#     """
#     n = min(n_samples, df[col].value_counts().min())
#     df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state = SEED))
#     df_.index = df_.index.droplevel(0)
#     return df_

def main():
    task = sys.argv[1] # binary_abuse or binary_movie_sentiment
    if task=="binary_abuse": x_cols = ['rev_id','text','split']
    elif task=="binary_movie_sentiment": x_cols = ['text','split']
    else: raise Exception(f"Invalid task {task}")

    SEED = 123

    # set directories
    main_dir = os.getcwd()
    print(f'Current working directory is: {main_dir}')
    data_dir = f'{main_dir}/data'

    # Load dev data
    orig_dev = pd.read_csv(f'{data_dir}/{task}/clean_data/{task}_dev.csv')
    print(f"Original dev set size: {len(orig_dev)}")
    print(f"Original value counts:\n{orig_dev['label'].value_counts(normalize = True)}")

    # Take 10% sample stratified by class labels
    X = orig_dev[x_cols]
    Y = np.array(orig_dev['label'])
    unsampled_X, sample_X, unsampled_Y, sample_Y = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = SEED)
    
    # Recombine as dataframe
    sample_dev = sample_X.copy()
    sample_dev['label'] = sample_Y.tolist()
    print(f"Sampled dev set size: {len(sample_dev)}")
    print(f"Sampled value counts:\n{sample_dev['label'].value_counts(normalize = True)}")
    
    # Save
    save_dir = f"{data_dir}/{task}/clean_data"
    sample_dev.to_csv(f'{save_dir}/{task}_dev_sample.csv', encoding = 'utf-8', index = True)


if __name__ == '__main__':
    main()
