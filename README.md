# elit

The Ensemble Learning and Iterative Training (ELIT) python module packages the methods for Deep Learning for feature extraction in atom-resolved 
electron micrscopy via workflows based on ensemble learning and iterative training as described [here](https://arxiv.org/pdf/2101.08449.pdf) 
and implemented in the Google Colab Notebook [here](https://colab.research.google.com/github/aghosh92/ELIT_tutorial/blob/main/ELIT_tutorial.ipynb). ELIT makes extensive use of [atomai](https://github.com/pycroscopy/atomai).

## Usage

The module has two modes: `train` and `infer`.

### elit train

The `train` model of elit trains an ensemble of models for atom localization given a set of input images and output masks defining atomic locations via a U-net as described in `atomai`'s [documentation](https://atomai.readthedocs.io/en/latest/README.html#deep-ensembles).

```
usage: elit train [-h] --images IMAGES --masks MASKS [--models MODELS]
                  [--cycles CYCLES] --out OUT
```

`elit train` has three required input files, `IMAGES` and `MASKS`, both of which should be `.npy` files containing an array of size `(N, M, M)` representing `N` `MxM` images. `elit train` saves the output model as a `.pkl` file and the directory where this is created is represented by the `OUT`.

Optional arguments are `MODELS` which represents the number of models in the ensemble (default 20) and `CYCLES` which is the number of training cycles (default 2000).

### elit infer

```
usage: elit infer [-h] --image IMAGE --models MODELS
```

`elit infer`
