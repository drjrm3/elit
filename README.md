# elit

The Ensemble Learning and Iterative Training (ELIT) python module packages the methods for Deep Learning for feature extraction in atom-resolved 
electron micrscopy via workflows based on ensemble learning and iterative training as described [here](https://arxiv.org/pdf/2101.08449.pdf) 
and implemented in the Google Colab Notebook [her](https://colab.research.google.com/github/aghosh92/ELIT_tutorial/blob/main/ELIT_tutorial.ipynb).

## Usage

The module has two modes: `train` and `infer`.

### elit train

```
usage: elit train [-h] --images IMAGES --masks MASKS [--models MODELS]
                  [--cycles CYCLES] --out OUT
```

`elit train` has three required input files, `IMAGES` and `MASKS`, both of which should be `.npy` files containing an array of size `(N, M, M)` representing `N` `MxM` images. `elit train` saves the output model as a `.pkl` file and the directory where this is created is represented by the `OUT`.

Optional arguments are `MODELS` which represents the number of models in the ensemble (default 20) and `CYCLES` which is the number of training cycles (default 2000).

### elit infer

