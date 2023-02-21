"""
Plotting functions

These functions are written to have standardised plots for the three methods.

It supposes that files are saved with the following format:

"mod={model}_n={n_train}_bal={balanced}_s={seed}.json"

Functions can be changed in function of this.

PROMPT ENGINEERING:
The following functions are thought to be ran for any given prompt.

"""

import numpy as np
import json
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.metrics import f1_score, roc_curve


"""Global variables"""

filename_format = "mod={}_n={}_bal={}_s={}"


def prepare_dataset(models: list, 
                    n_trains: list, 
                    seeds: list,
                    path: str,
                    bools: list = [True, False], 
                    filename_format: str = filename_format) -> dict:
    """ 
    Prepares a dictionary where each entry is the dictionary contained in every experiment file.
    The function takes into account that 
    
    Args:
    models: list of model names
    n_trains: list of training points
    bools: list of Booleans
    seeds: list of seeds
    path: path to files
    filename_format: high-level string of how files are named

    Returns:
    dataset: dict where the keys are the json filenames
    """

    dataset = dict()
    for model in models:
        for n in n_trains:
            for balanced in bools:
                for seed in seeds:
                    key = filename_format.format(model, n, balanced, seed)

                    filename = path+key+'.json'

                    f = open(filename)
                    data = json.load(f)
                    dataset[key] = data
                    f.close()
    return dataset


def get_f1_score(model: str, 
                n_trains: list, 
                seeds: list,
                balanced: bool,
                dataset: dict,
                average: str = 'macro',
                filename_format: str = filename_format) -> tuple:

    """
    returns a tuple of arrays with the means and standard deviations thought to be used to plot the learning curve.

    Args:
        models: model name
        n_trains: list of training points
        seeds: list of seeds
        balanced: boolean indicating if there was a balanced training
        average: str of average to be used when computing f1 scores
        dataset: dict used to compute f1 scores
        filename_format: high-level string of how files are named

    Returns:
        tuple of np.arrays
        f1_mean: mean f1 scores with shape=len(n_trains)
        f1_std: standard deviation f1 scores with shape=len(n_trains)
    """

    f1_mean = np.zeros(len(n_trains))
    f1_std = np.zeros(len(n_trains))

    for (j, n) in enumerate(n_trains):
        f1_seeds = np.zeros(len(seeds))
        for (i, seed) in enumerate(seeds):
            key = filename_format.format(model, n, balanced, seed)

            eval_true = dataset[key]['eval_true']
            eval_pred = dataset[key]['eval_pred']

            f1_seeds[i] = f1_score(eval_true, eval_pred, average=average)

        f1_mean[j] = f1_seeds.mean()
        f1_std[j] = f1_seeds.std()
    return f1_mean, f1_std


def get_fprs(model: str, 
            n_trains: list, 
            seeds: list,
            balanced: bool,
            dataset: dict,
            filename_format: str = filename_format) -> np.array:

    """ 
    Returns a sorted array with all unique fpr obtained when using sk_learn.metrics.roc_curve

    Args:
        models: name of model
        n_trains: list of training points
        seeds: list of seeds
        balanced: boolean indicating if there was a balanced training
        dataset: dict used to compute fprs
        filename_format: high-level string of how files are named

    Returns:
        fprs: a sorted np.array with all unique fprs
    """

    fprs = np.array([])
    for n in n_trains:
        for seed in seeds:
            key = filename_format.format(model, n, balanced, seed)

            eval_true = dataset[key]['eval_true']
            eval_pred = dataset[key]['eval_pred']
            
            fpr, _, _ = roc_curve(eval_true, eval_pred)
            
            fprs = np.concatenate([fprs, fpr])

    fprs = np.unique(fprs)
    return fprs


def get_tprs(fprs: np.array,
            model: str, 
            n_trains: list, 
            seeds: list,
            balanced: bool,
            dataset: dict,
            filename_format: str = filename_format) -> np.array:

    """
    returns an array with all interpolated tprs with respect to the fprs array given.

    Args:
        fprs: array of sorted fprs to be used as interpolation points
        models: name of model
        n_trains: list of training points
        seeds: list of seeds
        balanced: boolean indicating if there was a balanced training
        dataset: dict used to compute fprs
        filename_format: high-level string of how files are named

    Returns:
        tprs: an np.array with all interpolated tprs with shape = len(fprs), len(n_trains)*len(seeds) 
    """

    tprs = np.array([])
    for n in n_trains:
        for seed in seeds:
            key = filename_format.format(model, n, balanced, seed)
            
            eval_true = dataset[key]['eval_true']
            eval_pred = dataset[key]['eval_pred']
            
            fpr, tpr, _ = roc_curve(eval_true, eval_pred)
            
            tpr_interpolated = np.interp(fprs, fpr, tpr)
            tprs = np.concatenate([tprs, tpr_interpolated])
            
    tprs = tprs.reshape(int(len(tprs)/len(fprs)), len(fprs))
    return tprs

def get_mean_std_tprs(tprs: np.array) -> tuple:

    """
    returns the mean and standard deviation of the interpolated tprs.

    Args:
        tprs: np.array of interpolated tprs

    Returns:
        tprs_mean: np.array with the mean values of the interpolated tprs
        tprs_std: np.array with the standard deviations of the interpolated tprs
    """

    tprs_mean = tprs.mean(axis=0)
    tprs_std = tprs.std(axis=0)

    return tprs_mean, tprs_std

def timestr_to_float(string, separator):
    arr_time = string.split(separator)
    num = 0
    
    if len(arr_time) > 3:
        num = float(arr_time)*24*60*60
    else:
        num = 0
        
    for (i, timestr) in enumerate(arr_time[-3:]):
        exp = 60**(2-i)
        num += float(timestr)*exp
        
    return num

def get_times(model, n_trains, seeds, balanced, dataset, filename_format):
    times = np.zeros((len(seeds), len(n_trains)))
    for (j, n_train) in enumerate(n_trains):
        for (i, seed) in enumerate(seeds):
            key = filename_format.format(model, n_train, balanced, seed)

            time_str = dataset[key]['train_runtime']
            time_float = timestr_to_float(time_str, ":")

            times[i, j] = time_float

    return times

def define_figure_plots(figsize, ylabel):
    fig = plt.figure(figsize=figsize)

    plt.grid(alpha=0.3)

    plt.xlabel('Training points', size='x-large')
    plt.ylabel(ylabel, size='x-large')

    plt.xticks(size='large')
    plt.yticks(size='large')
    
    return fig

########### Plotting functions #################

def plot_experiment_roc_curves(figure: matplotlib.figure.Figure, 
                            model: str, 
                            n_trains: list, 
                            seeds: list,
                            balanced: bool,
                            dataset: dict,
                            fmt: str = 'k-',
                            alpha: float = 0.1,
                            filename_format: str = filename_format):

    """
    Plots all roc curves from all experiments for a given figure.

    Args:
        figure: figure where the curves are going to be plotted
        models: name of model
        n_trains: list of training points
        seeds: list of seeds
        balanced: boolean indicating if there was a balanced training
        dataset: dict used to compute fprs
        fmt: string to be passed as fmt argument to plt.plot
        alpha: float to be passed as alpha argument to plt.plot
        filename_format: high-level string of how files are named
    """

    for (j, n) in enumerate(n_trains):
        f1_seeds = np.zeros(len(seeds))
        for (i, seed) in enumerate(seeds):
            key = filename_format.format(model, n, balanced, seed)

            eval_true = dataset[key]['eval_true']
            eval_pred = dataset[key]['eval_pred']
            
            fpr, tpr, _ = roc_curve(eval_true, eval_pred)
            
            plt.plot(fpr, tpr, fmt, alpha=alpha)


def plot_mean_std(fig, x, mean, std, fmt, label, alpha_mean, threshold=0.5, min_alpha=0.1):
    
    alpha_fill = alpha_mean - threshold if alpha_mean > threshold else min_alpha
    color=fmt[0]
    
    plt.plot(x, mean, fmt, label=label, alpha=alpha_mean, figure=fig)
    plt.fill_between(x, mean-std, mean+std, alpha=alpha_fill, color=color, figure=fig)
                    

def plot_learning_curve(task: str,
                        method: str,
                        models,
                        n_trains: list,
                        seeds: list,
                        dataset: dict,
                        figsize: tuple,
                        alpha: float = 0.8,
                        ):
    
    fig = plt.figure(figsize=figsize)
    plt.xticks(size='large')
    plt.yticks(size='large')

    plt.xlabel('Training points', size='x-large')
    plt.ylabel('Macro F1 score', size='x-large')
    
    task = task.replace("_", " ")
    method = method.replace("_", " ")
    plt.title(f"Task: {task}, method: {method}", size='large')

    plt.grid(alpha=0.3)
    colors = [('b.-', 'r.-'), ('g.-', 'm.-')]

    for (i, model) in enumerate(models):
        balanced = True
        label = f'{model} - balanced'
        fmt = colors[i][0]
    
        f1_mean_true, f1_std_true = get_f1_score(model, n_trains, seeds, balanced, dataset)
        plot_mean_std(fig, n_trains, f1_mean_true, f1_std_true, fmt, label, alpha)
    
        balanced = False
        label = f'{model} - not balanced'
        fmt = colors[i][1]

        f1_mean_false, f1_std_false = get_f1_score(model, n_trains, seeds, balanced, dataset)
        plot_mean_std(fig, n_trains, f1_mean_false, f1_std_false, fmt, label, alpha)

    plt.legend(fontsize='large')
    
def plot_roc_curves(task: str,
                    model: str,
                    n_trains: list,
                    seeds: list,
                    balanced: bool,
                    dataset: dict,
                    figsize: tuple,
                    alpha: float = 0.8,
                   ):
    
    fprs = get_fprs(model, n_trains, seeds, balanced, dataset)
    tprs = get_tprs(fprs, model, n_trains, seeds, balanced, dataset)
    tprs_mean, tprs_std = get_mean_std_tprs(tprs)

    fig = plt.figure(figsize=figsize)

    plt.axes().set_aspect('equal', 'datalim')
    plt.plot([0, 1], [0, 1], 'r-')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xticks(size='large')
    plt.yticks(size='large')

    plt.xlabel('FP rate', size='x-large')
    plt.ylabel('TP rate', size='x-large')
    
    plt.grid(alpha=0.3)

    plt.title(f"Task: {task}, model: {model}, balanced: {balanced}", size='x-large')

    plt.plot(fprs, tprs_mean, 'b-', alpha=alpha, lw=2.)
    alpha_fill = alpha - 0.5 if alpha > 0.5 else 0.1
    plt.fill_between(fprs, tprs_mean-tprs_std, tprs_mean+tprs_std, alpha=alpha_fill, color='b')


    plot_experiment_roc_curves(fig, model, n_trains, seeds, balanced, dataset)


def plot_roc_ntrains(task, method, model, n_trains, seeds, balanced, dataset, figsize, color_palette: str = 'mako', filename_format: str = filename_format):
    fig = plt.figure(figsize=figsize)
    plt.axes().set_aspect('equal', 'datalim')
    plt.plot([0, 1], [0, 1], 'r-')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xticks(size='large')
    plt.yticks(size='large')

    plt.xlabel('FP rate', size='x-large')
    plt.ylabel('TP rate', size='x-large')

    plt.grid(alpha=0.3)
    plt.title(f"Task: {task}, method: {method}, model: {model}, balanced: {balanced}", size='large')

    colors = sns.color_palette(color_palette, len(n_trains))[::-1]

    for (j, n_train) in enumerate(n_trains):
        for seed in seeds:
            key = filename_format.format(model, n_train, balanced, seed)
            
            eval_true = dataset[key]['eval_true']
            eval_pred = dataset[key]['eval_pred']

            fpr, tpr, _ = roc_curve(eval_true, eval_pred)
            
            if seed == 1:
                plt.plot(fpr, tpr, figure=fig, color=colors[j], label=f"{n_train}")
            else:
                plt.plot(fpr, tpr, figure=fig, color=colors[j])
            
    plt.legend(fontsize='x-large', title='Training points')


def plot_times(task, method, models, n_trains, seeds, balanced, dataset, figsize, alpha: float = 0.8, filename_format: str = filename_format):

    fig = define_figure_plots(figsize, 'Training time')
    colors = [('b', 'r'), ('g', 'm')]

    alpha_fill = alpha-0.5

    task = task.replace('_', ' ')
    method = method.replace('_', '')
    plt.title(f"Task: {task}, method: {method}", size='large')


    for (i, model) in enumerate(models):    
        times_balanced = get_times(model, n_trains, seeds, True, dataset, filename_format)
        times_notbalanced = get_times(model, n_trains, seeds, False, dataset, filename_format)

        label = f'{model} - balanced'
        plt.plot(n_trains, times_balanced.mean(axis=0), colors[i][0]+'.-', alpha=alpha, label=label, figure=fig)
        plt.fill_between(n_trains, times_balanced.mean(axis=0)-times_balanced.std(axis=0), 
                        times_balanced.mean(axis=0)+times_balanced.std(axis=0), 
                        color=colors[i][0], alpha=alpha_fill, figure=fig)    

        label = f'{model} - not balanced'
        plt.plot(n_trains, times_notbalanced.mean(axis=0), colors[i][1]+'.-', alpha=alpha_fill+0.5, label=label, figure=fig)
        plt.fill_between(n_trains, times_notbalanced.mean(axis=0)-times_notbalanced.std(axis=0), 
                        times_notbalanced.mean(axis=0)+times_notbalanced.std(axis=0), 
                        color=colors[i][1], alpha=alpha_fill, figure=fig)

    plt.legend(fontsize='large')



