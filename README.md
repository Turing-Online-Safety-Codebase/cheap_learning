# Cheap Learning

This repository follows the research presented in the article "NAME OF THE ARTICLE".

In it, we test different NLP techniques and models to classify text in a context of scarce data. We train and test a Naive Bayes model, a Weak Supervision technique, a Prompt Engineering and a Transfer Learning techniques. 

For each one of these techniques, we train and test on two different datasets: the [IMDb movie reviews](https://huggingface.co/datasets/imdb) and the [Wikipedia Detox](https://github.com/ewulczyn/wiki-detox).

For more details, please refer to the paper. 

### Index
1. [Installation](#1-installation)
2. [Training and testing data sets](#2-data-sets)
    1. [IMDb movie review sentiment](#21-imdb-movie-review-sentiment)
    3. [TMDb movie review sentiment](#211-tmdb-data-set) 
    2. [Wiki Personal Attacks](#22-wikipedia-detox)
3. [Techniques](#3-techniques)
    1. Weak Supervision 
    2. Transfer Learning
    3. Prompt engineering
    4. Naive Bayes
    5. Zero-shot GPT
4. [Results](#4-results)
5. [Analysis](#5-analysis)
6. [Contact](#6-contact)


## 1. Installation<a name="1-installation">

The package is written in Python (minimal version: 3.10). We recommend that the installation is made inside a virtual environment and to do this, one can use either `conda` (recommended in order to control the Python version).


### Using conda

The tool `conda`, which comes bundled with Anaconda has the advantage that it lets us specify the version of Python that we want to use. Python>=3.10 is required.

After locating in the local github folder, like `cd $PATH$` e.g. `Documents/Local_Github/cheap_learning`, a new environment can be created with

```bash
$ conda env create -f environment.yaml
```

The environment's name will be `cheap_learning`.

## 2. Data sets<a name="2-data-sets">

### 2.1 IMDB movie review sentiment<a name="21-imdb-movie-review-sentiment">

First introduced in THIS ARTICLE, the data set contains 50,000 movie reviews from IMDb labelled according to whether they have a positive sentiment or negative sentiment (0: negative sentiment (50%), 1: positive sentiment (50%)) 

The entire data set is located in the subfolder [`data/binary_movie_sentiment`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_movie_sentiment). In it, we distinguish between:
- [The clean data splits](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_movie_sentiment/clean_data)
- [The raw data](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_movie_sentiment/raw_data)
- [An unbalanced version](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_movie_sentiment/unbalanced_data) of the data splits with a ratio of 12% Positive vs 88% Negative reviews

#### 2.1.1 TMDb data set

Given that we cannot be sure that the IMDb movie review data set is part of the training data set of the GPT-3.5 and GPT-4.0 models, we collected and tested an analogous dataset of movie reviews from TMDb. This data set containg 855 movie reviews published after October 2021 (passed the GPT training date cut) with a ratio of 73.3% of positive reviews and 26.7% of negative reviews.

The data set can be found in [`data/tbdb`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/tmdb)


### 2.2 Wikipedia Detox

First introduced in THIS ARTICLE, containing 115,864comments from the English language Wikipedia labelled according to whether they contain a personal attack (0: no personal attack (88.3%), 1: contains personal attack (11.7%)). 

The entire data set is located in the subfolder [`data/binary_abuse`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_abuse). In it, we distinguish between:
- [The clean data splits](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_abuse/clean_data)
- [The raw data](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_abuse/raw_data)


## 3. Techniques

## 4. Results

A collection of `csv` files with all the results can be found in [`results`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/results)

## 5. Analysis

Analysis is done via the `jupyter notebooks [results_analysis.ipynb](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/results_analysis.ipynb)`

## 6. Contact

Contact us!

In alphabetical order:
- Jonathan Bright - jbright@turing.ac.uk
- Leonardo Castro-Gonzalez - lmcastrogonzalez@turing.ac.uk
- Yi-Ling Chung - ychung@turing.ac.uk
- Pica Johansson - 
- Hannah R. Kirk - 
- Angus R. Williams - awilliams@turing.ac.uk

```
.
├── src
│   ├── dataset sampling script
│   ├── generic preprocessing script
│   ├── generic evaluation script
│   ├── transfer learning script
│   ├── weak supervision script
│   └── prompt engineering script     
├── data                   
│   ├── samples/
│   │   └── csv per X size sample?
│   ├── results (all contents on gitignore)
│   │   └── results/predictions per experiment
├── environment.yaml
├── .gitignore
└── README.md

```

### Creating Environment

Step 1: Locate local github folder `cd $PATH$` e.g. `Documents/Local_Github/cheap_learning`

Step 2: Create conda environment `conda env create -f environment.yaml`


### Training data with a certain technique 

If training with Naive Bayes: `bash ./src/naive_bayes_train_script.sh`

If training with Weak Supervision: `bash ./src/weak_supervision_script.sh`

If training with Transfer Learning: `bash ./src/transfer_learning_train_script.sh`

If training with Prompt Engineering: `bash ./src/prompt_engineering_train_script.sh`

### Plotting results

Check [plot_manuscript_figures.ipynb](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/blob/main/src/plot_manuscript_figures.ipynb) for data analysis