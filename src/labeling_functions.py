import pandas as pd
import datetime as dt
import re

from snorkel.labeling import LabelingFunction, labeling_function
from snorkel.preprocess import preprocessor
from textblob import TextBlob


### Global variables ###
ABUSE = 1
NOT_ABUSE = 0
ABSTAIN = -1

lfs = dict()

### NLP labeling functions ###

### Preprocessors
@preprocessor(memoize=True)
def textblob_polarity(x):
    scores = TextBlob(x.text)
    x.polarity = scores.polarity
    return x

@preprocessor(memoize=True)
def textblob_subjectivity(x):
    scores = TextBlob(x.text)
    x.subjectivity = scores.subjectivity
    return x

### NLP labeling functions
def lfg_polarity(bound: float,
                          operator: str,
                          CONSTANT: int,
                          name: str = 'polarityLabelingFunction',
                          ABSTAIN: int = -1
                        ):
    def lf_polarity(x):
        condition = eval(f"{x.polarity}"+operator+f"{bound}")
        return CONSTANT if condition else ABSTAIN
    return LabelingFunction(name=name, f=lf_polarity, pre=[textblob_polarity])
    

### Heuristic labeling functions ###

### length labeling function
def lfg_lengthtext(min_characters: int,
                            operator: str,
                            CONSTANT: int,
                            name: str = 'lengthtextLabelingFunction',
                            ABSTAIN: int = -1):
    def lf_lengthtext(x):
        condition = eval("len(x.text)"+operator+f"{min_characters}")
        return CONSTANT if condition else ABSTAIN

    return LabelingFunction(name=name, f=lf_lengthtext)

### Keyword labeling function
def lfg_keywords(keywords: list, 
                          CONSTANT: int, 
                          name: str = 'keywordsLabelingFunction',
                          ABSTAIN: int = -1):
    @labeling_function()
    def func(x):
        return CONSTANT if any(word in x.text.lower() for word in keywords) else ABSTAIN
    func.name = name
    return func

### Regex labeling function
def lfg_regex(regex, 
                       CONSTANT: int, 
                       name: str = 'regexLabelingFunction',
                       ABSTAIN: int = -1):
    @labeling_function()
    def func(x):
        return CONSTANT if re.search(regex, x.text.lower(), flags=re.I) else ABSTAIN
    func.name = name
    return func

### Annotators labeling functions ###

def worker_lf(x, worker_dict):
    return worker_dict.get(x.rev_id, ABSTAIN)

def lfg_worker(worker_id, worker_dicts):
    worker_dict = worker_dicts[worker_id]
    name = f"worker_{worker_id}"
    return LabelingFunction(name, f=worker_lf, resources={"worker_dict": worker_dict})


#######################################


def load_annotations(path: str) -> pd.DataFrame:
    annotations = pd.read_table(path)
    return annotations

def load_keywords(path: str) -> pd.DataFrame:
    keywords = pd.read_csv(path).keyword.values
    return keywords

def get_worker_dicts(annotations: pd.DataFrame) -> dict:

    labels_by_annotator = annotations.groupby("worker_id")
    worker_dicts = {}
    for worker_id in labels_by_annotator.groups:
        worker_df = labels_by_annotator.get_group(worker_id)[["rev_id", "attack"]]
        worker_dicts[worker_id] = dict(zip(worker_df.rev_id, worker_df.attack))
    return worker_dicts

def get_lfs(path_keywords: str, path_annotations: str) -> dict:
    lfs = dict()

    lfs['polarity_negative'] = lfg_polarity(-0.25, '<', ABUSE, name='polarity_negative')
    lfs['polarity_positive'] = lfg_polarity(0.1, '>=', NOT_ABUSE, name='polarity_positive')

    lfs['length_text'] = lfg_lengthtext(9500, ">", ABUSE)

    badwords = load_keywords(path_keywords)
    lfs['badwords'] = lfg_keywords(badwords, ABUSE)

    annotations = load_annotations(path_annotations)
    worker_dicts = get_worker_dicts(annotations)
    for worker_id in worker_dicts:
        name = f"worker_{worker_id}"
        lfs[name] = lfg_worker(worker_id, worker_dicts)

    return lfs
