# Cheap Learning

This repository follows the research presented in the article "NAME OF THE ARTICLE".

We test different NLP techniques and models to classify text in a context of scarce data. We train and test a Naive Bayes model, a Weak Supervision technique, a Prompt Engineering and a Transfer Learning techniques.

For each one of these techniques, we train and test on two different datasets: the [IMDb movie reviews](https://huggingface.co/datasets/imdb) and the [Wikipedia Detox](https://github.com/ewulczyn/wiki-detox).

For more details, please refer to the paper. 

## Index
1. [Installation](#1-installation)
2. [Training and testing data sets](#2-data-sets)
    1. [IMDb movie review sentiment](#21-imdb-movie-review-sentiment)
    3. [TMDb movie review sentiment](#211-tmdb-data-set) 
    2. [Wiki Personal Attacks](#22-wikipedia-detox)
3. [Techniques](#3-techniques)
    1. [Naive Bayes](#31-naive-bayes)
    2. [Weak Supervision](#32-weak-supervision) 
    3. [Transfer Learning](#33-transfer-learning)
    4. [Prompt engineering](#34-prompt-engineering)
    5. [Zero-shot GPT](#35-zero-shot-classifier-using-gpt)
4. [Results](#4-results)
5. [Analysis](#5-analysis)
6. [Contact](#6-contact)


## 1. Installation<a name="1-installation">

The package is written in Python (version: 3.8). We recommend that the installation is made inside a virtual environment and to do this, one can use `conda` (recommended in order to control the Python version).


### Using conda

The tool `conda`, which comes bundled with Anaconda has the advantage that it lets us specify the version of Python that we want to use. Python=3.8 is required.

After locating in the local github folder, like `cd $PATH$` e.g. `Documents/Local_Github/cheap_learning`, a new environment can be created with

```bash
$ conda env create -f environment.yaml
```

The environment's name will be `cheap_learning`.

## 2. Data sets<a name="2-data-sets">

### 2.1 IMDB movie review sentiment<a name="21-imdb-movie-review-sentiment">

First introduced in ([Maas et al, 2011](https://aclanthology.org/P11-1015/)), the data set contains 50,000 movie reviews from IMDb labelled according to whether they have a positive sentiment or negative sentiment (0: negative sentiment (50%), 1: positive sentiment (50%)) 

The entire data set is located in the subfolder [`data/binary_movie_sentiment`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_movie_sentiment). In it, we distinguish between:
- [The clean data splits](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_movie_sentiment/clean_data)
- [The raw data](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_movie_sentiment/raw_data)
- [An unbalanced version](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_movie_sentiment/unbalanced_data) of the data splits with a ratio of 12% Positive vs 88% Negative reviews

#### 2.1.1 TMDb data set

Given that we cannot be sure that the IMDb movie reviews data set is part of the training data set of the GPT-3.5 and GPT-4.0 models, we collected and tested an analogous dataset of movie reviews from TMDb. This data set contains 855 movie reviews published after October 2021 (passed the GPT training date cut) with a ratio of 73.3% of positive reviews and 26.7% of negative reviews.

The data set can be found in [`data/tmdb`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/tmdb). The scraper script is found in [`src/tmdb-database.py`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/tmdb-database.py)


### 2.2 Wikipedia Detox

First introduced in ([Wulczyn et al, 2017](https://dl.acm.org/doi/10.1145/3038912.3052591)), the data set contains 115,864 comments from the English language Wikipedia labelled according to whether they contain a personal attack or not (0: no personal attack (88.3%), 1: contains personal attack (11.7%)). 

The entire data set is located in the subfolder [`data/binary_abuse`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_abuse). In it, we distinguish between:
- [The clean data splits](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_abuse/clean_data)
- [The raw data](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_abuse/raw_data)


## 3. Techniques

Each one the techniques, with the exception of the the zero-shot Prompt Engineering classification using GPT-3.0, 3.5, 4.0 have a `bash` script that deploys the training of each technique.

### 3.1 Naive Bayes

To deploy the training with Naive Bayes, please run `bash ./src/naive_bayes_train_script.sh`.

The `bash` script calls [`src/naive_bayes_classifier.py`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/naive_bayes_classifier.py).

### 3.2 Weak Supervision

To deploy the training with Weak Supervision, please run `bash ./src/weak_supervision_script.sh`.

The `bash` script calls [`src/weak_supervision.py`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/weak_supervision.py) and the dictionary of labeling functions, found in [`src/labeling_functions.py`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/labeling_functions.py).

In particular, for the binary abuse task, Weak Supervision also uses the annotated keywords in [`data/binary_abuse/misc`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/data/binary_abuse/misc).

We train the model defined by the authors of [Weak Supervision](https://ojs.aaai.org/index.php/AAAI/article/view/4403):
- LabelModel

### 3.3 Transfer Learning

To deploy the training with Transfer Learning, please run `bash ./src/transfer_learning_train_script.sh`.

The `bash` script calls [`src/transfer_learning.py`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/weak_supervision.py).

We train two models:
- DistilBERT
- DeBERTa-v3

### 3.4 Prompt Engineering

To deploy the training with Prompt Engineering, please run `bash ./src/prompt_engineering_train_script.sh`.

The `bash` script calls [`src/prompt_engineering.py`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/prompt_engineering.py).

We use three different prompts for each of the two datasets:
    
For IMDB movie review sentiment:
- "Is this text negative?"
- "Does this text contain negative sentiment?"
- "It was? Negative or not negative?"

For Wikipedia Detox:
- "Is this text abusive?",
- "Does this text contain abuse?"
- "It was? Abusive or Not Abusive"

We train two models:
- DistilBERT
- GPT-2.0

### 3.5 Zero-shot classifier using GPT

We also perform a zero-shot classification exercise (no training) with the out-of-the-box OpenAI LLMs, GPT-3.0, 3.5 and 4.0.

The script of the zero-shot exercise can be found in the `jupyter notebook` [`open_ai_prompt_engineering.ipynb`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/openai_prompt_engineering.ipynb).

We use three different prompts for each of the two datasets:

For IMDB movie review sentiment:
- "Using one word, classify the sentiment of the movie review using 'Positive' or 'Negative'."
- "Using one word, does the movie review contain negative sentiment, Yes or No?"
- "You are a researcher who needs to classify movie reviews as containing negative sentiment or not containing negative sentiment. Using one word, does the movie review contain negative sentiment, Yes or No?"

For Wikipedia Detox:
- "Using one word, does the internet comment contain toxic language, Yes or No?"
- "Using one word, is this internet comment using toxic language, Yes or No?"
- "You are a researcher who needs to classify comments on the internet as containing abusive language or not containing abusive language. Using one word, does the internet comment contain abusive language, Yes or No?"

## 4. Results

A collection of `csv` files with all the results can be found in [`results`](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/results). The full results are stored in [task_results_final.csv](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/blob/main/results/task_results_final.csv).

## 5. Analysis

Analysis is done via the `jupyter notebooks` [results_analysis.ipynb](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/tree/main/src/results_analysis.ipynb)` and [plot_manuscript_figures.ipynb](https://github.com/Turing-Online-Safety-Codebase/cheap_learning/blob/main/src/plot_manuscript_figures.ipynb)

## 6. Contact

In alphabetical order:
- Jonathan Bright - jbright@turing.ac.uk
- Leonardo Castro-Gonzalez - lmcastrogonzalez@turing.ac.uk
- Yi-Ling Chung - ychung@turing.ac.uk
- John Francis - jfrancis@turing.ac.uk
- Pica Johansson - 
- Hannah R. Kirk - 
- Angus R. Williams - awilliams@turing.ac.uk
