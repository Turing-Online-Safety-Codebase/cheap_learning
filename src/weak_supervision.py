import os
import logging
import datetime as dt
import argparse

from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model.label_model import LabelModel

from helper_functions import (
    check_dir_exists,
    convert_labels,
    load_balanced_n_samples,
    load_n_samples,
)
from evaluation import get_results_dict, save_results
from labeling_functions import get_lfs, get_lfs_imdb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Weak Supervision")
    parser.add_argument(
        "--n_train", type=int, default=16, help="num of training entries"
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=-1,
        help="num of eval entries. Set to -1 to take all entries.",
    )
    parser.add_argument(
        "--eval_set", type=str, default="dev_sample", help="name of eval set"
    )
    # parser.add_argument('--balanced_train', type=bool, action=argparse.BooleanOptionalAction, help='if training entries are balanced by class label')
    parser.add_argument("--balanced_train", action="store_true")
    parser.add_argument(
        "--no-balanced_train", dest="balanced_train", action="store_false"
    )
    parser.set_defaults(balanced_train=True)
    parser.set_defaults(balanced_eval=True)

    parser.add_argument(
        "--model_name", type=str, default="LabelModel", help="name of the model"
    )
    parser.add_argument("--task", type=str, help="target task")
    parser.add_argument(
        "--filename_annotations",
        default="",
        type=str,
        help="filename of annotations csv",
    )
    parser.add_argument(
        "--filename_keywords", default="", type=str, help="filename of keywords csv"
    )
    parser.add_argument(
        "--tie_break_policy",
        type=str,
        default="random",
        help="Tie break policy for predicting labels (random vs abstain)",
    )
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args


map_bool = lambda x: 1 if x else 0


def prepare_dataset(
    raw_data: dict,
    columns=["text", "label", "id"],
    splits=["train", "eval"],
) -> dict:
    """
    This function only works for binary abuse dataset for now
    """
    dataset = dict()
    for split in splits:
        df = raw_data[split]
        df = df[columns]

        df.loc[:, ("label")] = df.label.apply(map_bool)
        dataset[split] = df

    return dataset


def main(
    TASK: str,
    TECH: str,
    seed: int,
    model_name: str,
    data_dir: str,
    output_dir: str,
    n_train: int,
    n_eval: int,
    eval_set: str,
    balanced_train: bool,
    path_annotations: str,
    path_keywords: str,
    tie_break_policy: str,
):
    ### Time
    datetime_str = str(dt.datetime.now())

    ### Setup logging
    logger.setLevel(logging.DEBUG)
    log_dir = f"{output_dir}/logs"
    check_dir_exists(log_dir)
    handler = logging.FileHandler(f"{log_dir}/{datetime_str}.log")
    # format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    # handler.setFormatter(format)
    logger.addHandler(handler)

    logger.info(f"--Start: {datetime_str}--")

    ### Load data
    raw_data = dict()
    split = "train"
    if balanced_train:
        raw_data[split], n_classes_train = convert_labels(
            load_balanced_n_samples(data_dir, TASK, split, n_train)
        )
    else:
        raw_data[split], n_classes_train = convert_labels(
            load_n_samples(data_dir, TASK, split, n_train)
        )

    split = "eval"
    raw_data[split], n_classes_eval = convert_labels(
        load_n_samples(data_dir, TASK, eval_set, n_eval)
    )
    # Is the eval set balanced or unbalanced?

    if "unbalanced" in eval_set:
        balanced_eval = False
    elif "balanced" in eval_set:
        balanced_eval = True
    else:
        balanced_eval = "NA"
        print("No information about the balancing of the eval set")

    if n_classes_train == n_classes_eval:
        n_classes = n_classes_train
    else:
        print("Error: train and eval have different number of classes")

    for split in ["train", "eval"]:
        logger.info(f"--{len(raw_data[split])} examples in {split} set--\n")
        logger.info(
            f"--label distribution for {split} set--\n{raw_data[split]['label'].value_counts()}"
        )

    ### Preprocess data
    if TASK == "binary_abuse":
        dataset = prepare_dataset(raw_data)
    else:
        dataset = raw_data.copy()

    ### Labeling functions
    if TASK == "binary_abuse":
        labeling_functions = get_lfs(path_keywords, path_annotations)
    else:
        labeling_functions = get_lfs_imdb()

    ### Train model
    logger.info("--Model Training--")
    lfs = [item[1] for item in labeling_functions.items()]

    applier = PandasLFApplier(lfs=lfs)

    start = dt.datetime.now()
    L_train = applier.apply(df=dataset["train"])

    label_model = LabelModel(cardinality=n_classes_train, verbose=False)
    Y_eval = dataset["eval"].label.values

    label_model.fit(L_train=L_train, n_epochs=100, log_freq=10, seed=seed)
    end = dt.datetime.now()

    L_eval = applier.apply(df=dataset["eval"])

    ### Model evaluation
    logger.info("--Model Evaluation--")
    runtime = str(end - start)
    eval_preds = label_model.predict(L_eval, tie_break_policy=tie_break_policy)
    results_dict = get_results_dict(
        TASK,
        TECH,
        model_name,
        runtime,
        Y_eval,
        eval_preds,
        eval_set,
        n_train,
        n_eval,
        balanced_train,
        balanced_eval,
        seed,
        datetime_str,
    )

    ### Save output
    save_str = f"mod={model_name}_n={n_train}_bal={balanced_train}__baleval={balanced_eval}_s={seed}"
    save_results(output_dir, save_str, results_dict)


### main
if __name__ == "__main__":
    args = parse_args()

    # Set global vars
    TECH = "weak_supervision"
    TASK = args.task
    # TASK = "binary_abuse"
    # Set dirs
    path = os.getcwd()
    main_dir = path  # os.path.split(path)[0]
    data_dir = f"{main_dir}/data"
    raw_data_dir = f"{main_dir}/data/{TASK}/raw_data/"
    misc_data_dir = f"{main_dir}/data/{TASK}/misc/"
    output_dir = f"{main_dir}/results/{TASK}/{TECH}"

    path_annotations = raw_data_dir + args.filename_annotations
    path_keywords = misc_data_dir + args.filename_keywords
    # Run for multiple seeds
    for SEED in [1, 2, 3]:
        print(f"RUNNING for SEED={SEED}")
        main(
            TASK,
            TECH,
            SEED,
            args.model_name,
            data_dir,
            output_dir,
            args.n_train,
            args.n_eval,
            args.eval_set,
            args.balanced_train,
            path_annotations,
            path_keywords,
            args.tie_break_policy,
        )
