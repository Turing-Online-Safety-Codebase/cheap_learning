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



