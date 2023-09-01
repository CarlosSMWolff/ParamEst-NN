[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8305509.svg)](https://www.dropbox.com/scl/fi/mkbs1kvn06mvsd5k5v0z9/data.zip?rlkey=k6otygqljpv9aj3bkjp0pjey3&dl=0)

# Parameter Estimation from Photon Counting data with Neural Networks

Parameter estimation via deep learning of quantum correlations in continuous photon counting measurements.
This repository contains [Jupyter](https://jupyter.org/) notebooks with the codes necessary to reproduce the results in <a href = "https://arxiv.org/abs/" target="_blank"> this paper</a>. 

<p align="center"><img src="notebooks/figures/fig1.png"  align=middle width=600pt />
</p>

Figure 1: Quantum parameter estimation strategies in open quantum systems. Parameters are encoded in the dynamics of an open quantum system: here, the frequency detuning $\Delta = \omega_q-\omega_L$ and amplitude $\Omega$ of an electromagnetic field driving a qubit. The quantum light radiated by the emitter is detected and the photodetection times recorded. The unknown parameters can be reconstructed by application through Bayesian parameter estimation. An alternative approach is based on the use of Neural Networks.


## How to use

### Installation

- Clone this directory
- cd to the current folder `cd ParamEst-NN`
- (Recommended) Create a new Python environment and activate it. For instance, using conda, you can create and activate an environment with name `paramest`, using Python 3.9, as
```
conda create -n paramest python=3.9
conda activate paramest
```
- Install the `paramest_nn` as an editable python package with its dependencies
```
pip install -e .
```

In Windows you might need to use a package manager such as Anaconda before
installing the `paramest_nn` package. 

The details to install conda or a lightweight version called miniconda can be
found below:

- [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- [`miniconda`](https://docs.conda.io/en/latest/miniconda.html)

### Populating the ```[datapath]``` folder
At the beginning of each notebook, the path to a directory where data is loaded and written is set as the variable `datapath`. 

#### Option #1: Download the data
You can download the necessary content of the ```[datapath]/``` folder in order to reproduce our results in <a href = "https://arxiv.org/abs/" target="_blank"> the paper</a>
from Zenodo at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8305509.svg)](https://www.dropbox.com/scl/fi/mkbs1kvn06mvsd5k5v0z9/data.zip?rlkey=k6otygqljpv9aj3bkjp0pjey3&dl=0).

Alternatively, you can directly run the notebooks setting the variable 
```python
download_required = True
```
and the data will be downloaded automatically. 

#### Option #2: Populate the folder with your own computations
By running the notebooks in this repository, the ```[datapath]/``` folder is populated with training and validation datasets, trained models, and cached results.
These folders have to be populated by running the following notebooks in order:
* [1-Trajectories_generation.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/notebooks/1-Trajectories_generation.ipynb) (populates `[datapath]/training-trajectories` and `[datapath]/validation-trajectories`)
* [2-Training.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/notebooks/2-Training.ipynb) (populates `[datapath]/models`, requires populated `[datapath]/training-trajectories`).
* [3-Results.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/notebooks/3-Results.ipynb) (populates `[datapath]/results-cache`, requires populated `[datapath]/validation-trajectories` and 
`[datapath]/models`)

##### Instructions for 2D Ultranest: TO DO


### Running codes in Google Colaboratory

The notebooks are ready to be used in Google Colaboratory, which can be done by pressing ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) at the top of each notebook.  

**Remark**: The notebook [2-Training.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/2-Training.ipynb) is set up to take advantage from Google Colaboratory's TPUs. This is the fastest option for training the models, and it is recommended for readers with access to a Colab Pro account.

Since notebooks [2-Training.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/2-Training.ipynb) and [3-Results.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/3-Results.ipynb) required populated `data` folders, these notebooks will download a compressed `data` folder from Zenodo when running in Google Colab. The download link can be changed to any suitable link to a compressed `data.zip` file containing the populated `data` folder. We recommend re-uploading the downloaded datafolder from Zenodo, or your own data folder, into [Dropbox](https://www.dropbox.com), and then using your own Dropbox link in these notebooks. This ensures faster download speeds when using Google Colaboratory.


## Notebooks
(Currently tested on TensorFlow 2.12.1)

### [1-Trajectories_generation.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/1-Trajectories_generation.ipynb)
Generates quantum trajectories for training and validating the neural networks.

### [2-Training.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/2-Training.ipynb)
Trains neural networks for the problem of quantum parameter estimation.

### [3-Results.ipynb](https://github.com/CarlosSMWolff/ParamEst-NN/blob/main/3-Results.ipynb)
Reproduces the main figures shown in the manuscript, assessing the performance of the trained models.

## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [CarlosSMWolff](https://github.com/CarlosSMWolff)
* Email: [carlossmwolff@gmail.com](carlossmwolff@gmail.com)

### BibTex reference format for citation for the Code
```
@misc{ParamEstNN,
title={},
url={https://github.com/CarlosSMWolff/ParamEst-NN},
note={GitHub repository containing deep learning approach generating fundamental and excited eigenfunctions for molecular potentials.},
author={Enrico Rinaldi, Carlos Sánchez Muñoz},
  year={2023}
}

