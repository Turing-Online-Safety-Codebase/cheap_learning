# Cheap Learning

Repo for Turing Online Safety Team Cheap Learning project

Abuse Detection Dataset: [Wikipedia Detox](https://github.com/ewulczyn/wiki-detox)

Movie sentiment Dataset: [IMDb](https://huggingface.co/datasets/imdb)
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