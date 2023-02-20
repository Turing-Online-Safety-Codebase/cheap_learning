#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Loads Maas et al., (2011) data.
Sourced from https://huggingface.co/datasets/imdb
"""
import os
import pandas as pd
import datasets
from sklearn.utils import shuffle
from helper_functions import check_dir_exists
from cleaning_functions import clean_text, drop_nans

def main():
    SEED = 123 
    task = 'binary_movie_sentiment'

    path = os.getcwd()
    main_dir = os.path.split(path)[0]
    print(f'Current working directory is: {main_dir}')
    task_dir = f"{main_dir}/data/{task}"
    check_dir_exists(task_dir)

    # download data
    ds = datasets.load_dataset("imdb")

    check_dir_exists(f"{task_dir}/raw_data")
    ds['train'].to_pandas().to_csv(f"{task_dir}/raw_data/train_dataset.csv", encoding='utf-8', index=False)
    ds['test'].to_pandas().to_csv(f"{task_dir}/raw_data/test_dataset.csv", encoding='utf-8', index=False)
    ds['unsupervised'].to_pandas().to_csv(f"{task_dir}/raw_data/unsupervised_dataset.csv", encoding='utf-8', index=False)

    # get train set as df
    ds_train = ds['train'].to_pandas()
    ds_train['split'] = 'train'

    # get test set as df, shuffle, extract stratified 50% for dev set
    ds_test = ds['test'].to_pandas()
    ds_test = shuffle(ds_test, random_state=SEED).reset_index(drop=True)

    ds_dev_0 = ds_test[ds_test.label==0].sample(frac=0.5, random_state=SEED)
    ds_dev_1 = ds_test[ds_test.label==1].sample(frac=0.5, random_state=SEED)
    ds_dev = shuffle(pd.concat([ds_dev_0, ds_dev_1]), random_state=SEED)
    ds_dev['split'] = 'dev'

    ds_test = shuffle(ds_test.drop(ds_dev.index), random_state=SEED)
    ds_test['split'] = 'test'

    # get unsupervised set as df
    ds_unsup = ds['unsupervised'].to_pandas()
    ds_unsup['split'] = 'unsupervised'

    # combine splits
    df = pd.concat([ds_train, ds_test, ds_dev, ds_unsup]).sort_values(by=['split','label']).reset_index(drop=True)

    # map label to bool
    df['label'] = df.label.astype(bool)

    # clean text
    df['text'] = df['text'].apply(clean_text)

    # save main df
    check_dir_exists(f"{task_dir}/clean_data")
    df.to_csv(f'{task_dir}/clean_data/{task}.csv', encoding='utf-8', index=True)

    # save splits
    splits = ['train', 'test', 'dev']
    for s in splits:
        subset_df = df[df['split']==s]
        print(f"Number of {s} entries: {len(subset_df)}")
        subset_df = shuffle(subset_df, random_state=SEED)
        subset_df.to_csv(f'{task_dir}/clean_data/{task}_{s}.csv', encoding = 'utf-8', index = True)


if __name__ == '__main__':
    main()