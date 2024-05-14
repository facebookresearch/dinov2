# Training

This notes aims to cover the different issues / technicalities encountered while trying to train & implement DL architectures.

## Dinov2

[Dinov2](https://github.com/facebookresearch/dinov2/tree/main) is a model made for training foundation models to images taking inspiration from llm.  
It uses a teacher and a student models, AdamW backpropagation and several other techniques to optimize training.

The model used as a backbone is a Vision Transformer.  
There are 4 model architectures :

- ViT small with 21 M parameters
- ViT base with 86 M parameters
- ViT large with 300 M parameters
- ViT giant with 1100 M parameters

Knowing that a parameter takes 4 bytes on memory, these models range from 84 MB to 4.4 GB to encapsulate.

### Necessary modifications

In order to make use of the codes in the repo on a custom dataset, some steps are needed (indeed the code given for pretraining is only for imagenet datasets).

**Datasets**  
Custom datases class needs to be implemented, a simple dataset built on the torch dataset class makes the job

**data collators**  
The data collate function is made for the Imagenet datasets that have classes, and happens after the cropping step (makes 2 global images and 8 locals).  
Simply tweeking a bit the collate function make it work.

**data loaders**  
No modifications on the dataloaders were made

### Empirical observations

The remarques made here are for models trained with the dinov2 codes given in the github repo.  

#### Batch size effect on memory usage

Batch size has an enormous effect on the memory usage.
Indeed with a **ViT large**:

- batch size of 32 makes the memory usage peak to 22 GB
- batch size of 64 makes the memory usage peak to 42 GB

To get to such proportions

#### Training time

With a ViT large, a batch size of 32 and a dataset made of 41K images 200 epochs takes **~5 hours** on an A500.  
The different datasets that we want to use are:

- medhi dataset : 41k images
- vexas dataset : 3.5K images
- TAMIS dataset : 4.5K images
- Saint antoine dataset : 12K images

> TODO Check the number of images per dataset and test the training time when using all of the datasets of interest

### Distributed training

To boost training (or just to train the giant model) the GPUs on the cluster need to be used collectively (in a distributed fashion).

> TODO : test to train a small ViT on 2 GPUs 

To do so the go to way seems to use the dinov2/run/ folder.  
By default the `output_dir` is `/checkpoint/guevel/experiments`, so it needs to be hand picked to work.  

The submit folder makes the sbatch command by itself, that is then put in the squeue.  
> needs to modify the command in order to take into account the correct python environment.

One command that seems to be promising is :
` python run/train/train.py --config-file configs/config_large.yaml --output-dir /home/guevel/OT4D/cell_similarity/logs --partition hard --ngpus 2`
when the conda env is activated.

### Impact of the model architecture

When trying to use the small ViT architecture, the code returned an error message concerning the forward_backward function -> **Is vit_large the only functional architecture ?**

### Saving

When indicating an output_dir inside the config file, the checkpointer doesn't care about it and just saves inside the root folder.  
