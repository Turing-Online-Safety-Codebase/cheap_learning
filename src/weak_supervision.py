import os
import logging
import datetime as dt
import argparse

from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model.label_model import LabelModel

from helper_functions import check_dir_exists, convert_labels, load_balanced_n_samples, load_n_samples
from evaluation import get_results_dict, save_results
from labeling_functions import get_lfs

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Weak Supervision")
    parser.add_argument('--n_train', type=int, default=16, help='num of training entries')
    parser.add_argument('--n_test', type=int, default=100, help='num of testing entries. If -1 is given, then whole set is used as input')
    parser.add_argument('--n_dev', type=int, default=100, help='num of dev entries. If -1 is given, then whole set is used as input')
    parser.add_argument('--balanced_train', action=argparse.BooleanOptionalAction, default=False, help='if training entries are balanced by class label')
    parser.add_argument('--dev', action=argparse.BooleanOptionalAction, default=True, help='dev split included?')
    parser.add_argument('--model_name', type=str, default='LabelModel', help='name of the model')
    parser.add_argument('--task', type=str, help='target task')
    parser.add_argument('--filename_annotations', type=str, help='filename of annotations csv')
    parser.add_argument('--filename_keywords', type=str, help='filename of keywords csv')
    parser.add_argument('--tie_break_policy', type=str, default='random', help='Tie break policy for predicting labels (random vs abstain)')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

map_bool = lambda x: 1 if x else 0

def prepare_dataset(raw_data: dict, 
                    columns = ['text', 'label', 'rev_id'],
                    splits = ['train', 'test', 'dev'],
                    ) -> dict:
    """
    This function only works for binary abuse dataset for now
    """
    dataset = dict()
    for split in splits:
         df = raw_data[split]
         df = df[columns]

         df.loc[:, ['label']] = df['label'].apply(map_bool)
         dataset[split] = df

    return dataset
         

def main(
        TASK: str,
        TECH: str,
        seed: int,
        model_name: str, 
        data_dir: str,
        output_dir: str,
        splits: list,
        n_train: int, 
        n_test: int,
        n_dev: int,
        balanced_train: bool,
        path_annotations: str,
        path_keywords: str, 
        tie_break_policy: str,
        ):

    ### Time 
    datetime_str = str(dt.datetime.now())

    ### Setup logging
    logger.setLevel(logging.DEBUG)
    log_dir = f'{output_dir}/logs'
    check_dir_exists(log_dir)
    handler = logging.FileHandler(f"{log_dir}/{datetime_str}.log")
    # format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    # handler.setFormatter(format)
    logger.addHandler(handler)

    logger.info(f"--Start: {datetime_str}--")

    ### Load data
    raw_data = dict()
    split = 'train'
    if balanced_train:
        raw_data[split], n_classes_train = convert_labels(load_balanced_n_samples(data_dir, TASK, split, n_train))
    else:
        raw_data[split], n_classes_train = convert_labels(load_n_samples(data_dir, TASK, split, n_train)) 

    split = 'test'
    raw_data[split], n_classes_test = convert_labels(load_n_samples(data_dir, TASK, split, n_test))

    if 'dev' in splits:
        split = 'dev'
        raw_data[split], n_classes_dev = convert_labels(load_n_samples(data_dir, TASK, split, n_dev))

    if n_classes_train == n_classes_dev == n_classes_test:
        n_classes = n_classes_train
    else:
        print("Error: train, test or dev have different number of classes")

    for split in splits:
        logger.info(f'--{len(raw_data[split])} examples in {split} set--\n')
        logger.info(f"--label distribution for {split} set--\n{raw_data[split]['label'].value_counts()}")
    
    ### Preprocess data
    dataset = prepare_dataset(raw_data)

    ### Labeling functions
    labeling_functions = get_lfs(path_keywords, path_annotations)

    ### Train model
    logger.info("--Model Training--")
    lfs = [item[1] for item in labeling_functions.items()]

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=dataset['train'])
    L_test = applier.apply(df=dataset['test'])

    label_model = LabelModel(cardinality=n_classes_train, verbose=False)
    Y_test = dataset['test'].label.values

    if 'dev' in splits:
        L_dev = applier.apply(df=dataset['dev'])
        Y_dev = dataset['dev'].label.values
        start = dt.datetime.now()
        label_model.fit(L_train=L_train, Y_dev=Y_dev, n_epochs=100, log_freq=10, seed=seed)
    else:
        start = dt.datetime.now()
        label_model.fit(L_train=L_train, n_epochs=100, log_freq=10, seed=seed)
    end = dt.datetime.now()

    ### Model evaluation
    logger.info("--Model Evaluation--")
    runtime = str(end-start)
    test_preds = label_model.predict(L_test, tie_break_policy=tie_break_policy)
    dev_preds = label_model.predict(L_dev, tie_break_policy=tie_break_policy)
    results_dict = get_results_dict(TASK, TECH, model_name, runtime,
                    Y_test, test_preds,
                    Y_dev, dev_preds,
                    n_train, n_dev, n_test, balanced_train, seed, datetime_str)

    ### Save output
    save_results(output_dir, datetime_str, results_dict)


###
if __name__ == '__main__':
    args = parse_args()

    # Set global vars
    TECH = 'weak_supervision'
    TASK = args.task

    # Set dirs
    path = os.getcwd()
    main_dir = os.path.split(path)[0]
    data_dir = f"{main_dir}/data"
    raw_data_dir = f"{main_dir}/data/{TASK}/raw_data/"
    misc_data_dir = f"{main_dir}/data/{TASK}/misc/"
    output_dir = f'{main_dir}/results/{TASK}/{TECH}'

    if args.dev:
        splits = ['train', 'test', 'dev']
    else:
        splits = ['train', 'test']

    path_annotations = raw_data_dir + args.filename_annotations
    path_keywords = misc_data_dir + args.filename_keywords
    # Run for multiple seeds
    for SEED in [1,2,3]:
        main(TASK, TECH, SEED, args.model_name, data_dir, output_dir, splits, 
                args.n_train, args.n_test, args.n_dev, args.balanced_train,
                path_annotations, path_keywords, args.tie_break_policy)
