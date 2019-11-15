# Semantic Segmentation of Histopathological Slides

## Project Description
Mycosis fungoides (MF) is a slowly progressing but potentially life-threatening skin disease. If appropriate treatment is applied in early stages, the patients encounter a normal survival rate. Eczema on the other hand is a very common benign inflammatory skin disease whose clinical and histopathological features are similar to early stage MF. The early distinction of MF and eczema is of utmost importance in order to get the appropriate treatment for the patient.

The diagnosis of MF and eczema is usually based on the microscopic examination of histopathological slides (thin slices of stained tissue) and relies mostly on the presence of atypical lymphocytes in the epidermis. However, histological distinction from eczema may be very difficult due to similar histological features. Thus the final diagnosis of early MF is very challenging and relies on guidelines that include multiple  eighted criteria such as the clinical appearance, histopathology, immune-histology, and molecular pathology.

 
### Goal

The aim of this project is to develop machine learning techniques that can detect and highlight relevant features in histopathological images in order to support clinicians in distinguishing MF from eczema. In particular, one goal is to develop methods for semantic segmentation of such images that can automatically recognize spongiotic areas in the epidermis as well as regions in which lymphocytes are present. Building upon that, another goal is to develop techniques for quantifying the presence of spongiotic areas and lymphocytes in the epidermis. Ultimately, these techniques should become part of a system that can assist clinicians in distinguishing MF from eczema.

The project will be based on a a dataset consisting of 201 Hematoxylin-Eosin (HE) stained whole slide images (WSI) scanned at 20x magnification and including the label of the diagnosis at the WSI-level (MF or eczema) as well as other types of annotations.

## Getting Started
### Basic Project Architecture
The basic Project Architecture was taken from https://github.com/MrGemy95/Tensorflow-Project-Template, but heavily edited
to be compatible with Tensorflow 2.0. For a detailed explanation of the architecture have a look at the example 
files or visit the other repo mentioned above. This basic architectures allows several people to 
implement their individual models while still sharing important Code parts such as data-loading, logging etc. This avoids
replicated code that does the same thing and might be buggy. It also allows to collaborate easily as we all have the same code
structure. 

Quick start:
You need to create the following files:
    - config file
    - main file 
    - model 
    - data generator
    - trainer

### Logging/Experiment Tracking 

For Logging we use comet.ml. Why do we use this and not Tensorboard? Comet answers in the following way:
"Comet provides deeper reporting and more features compared to Tensorboard. Additionally, Comet allows users to view 
multiple experiments and manage all experiments from a single location, whereas Tensorboard is focused on single
 experiment views and runs locally on your machine. Finally, Tensorboard does not scale whereas Comet supports 1m+ experiments."
 
Especially for a larger Team its essential to track progress and be able to reproduce experiments. Lastly it allows you to 
 track your datasets and it calculates your git code hash even if you have uncommited changes. 
Also, you don't need to do anything except sign up(its free for students) and download it. Follow the instructions on their website (https://www.comet.ml/)
. Everything else I implemented in the base classes. 

### General 
Have a look at /tools/getting_started_with_slides.ipynb to get familiar with the type of data that we are working with. 

### First steps (Quick-start)

#### Adding an environment file

Before you start, please add a file called ".env" in the same directory as this README.md is in.
Please copy-paste the below contents:

```
PROJECT_PATH="/Users/jeremyscheurer/Code/semantic-segmentation-of-histopathological-slides"
```

and make it point to your project root folder.

#### Creating a virtualenv 

Create a virtualenvironment using following command (in the root folder):

```
virtualenv -p /usr/local/bin/python3.7 venv
```

where you point to your python 3 version. To get your python 3 version use 

```
which python3
```


#### Installing dependencies with pip

Install all dependencies into the virtualenvironment that you just created
```
source venv/bin/activate
pip install -r requirements.txt
```

#### Getting the Data
Execute 
```
wget -r -p --user user --password pw https://digipath.ise.inf.ethz.ch/mfec/
```

#### How to use the ETH Leonhard Cluster 
Some help here: 
- https://scicomp.ethz.ch/wiki/Getting_started_with_clusters
- https://scicomp.ethz.ch/wiki/Getting_started_with_clusters
- https://scicomp.ethz.ch/wiki/FAQ
- https://scicomp.ethz.ch/wiki/Python_on_Leonhard

Setup: 
```
module load python_gpu/3.7.1
module load eth_proxy
pip3 install -r requirements.txt --user
pip install --user tensorflow-gpu==2.0
```
Note: Don't update pip to version 19.3 as it is somehow broken. But you need at least pip version 19.0 to have TF version 2.0. 
So you might have to do the following: 
```
pip install -U --user pip==19.0
```

How to run a model: 
```
bsub -R "rusage[ngpus_excl_p=1]" -W 10:00 python3 mains/transfer_learning_unet_main.py -c configs/transfer_learning_unet_config.json```
```
If you have trouble because the root path is not added to the PYTHONPATH, do the following: 
```
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
```

## Citations
The initial framework for our project was taken from https://github.com/MrGemy95/Tensorflow-Project-Template
but heavily edited for tensorflow 2.0 compatibility. 

The whole transfer learning Code was taken from https://github.com/qubvel/segmentation_models, 
we added and changed a few things here and there but we are thankful for this amazing code. 