This repository contains the implementation for: [Differentially Private GAN for Time Series](Differentially%20Private%20GAN%20for%20Time%20Series.pdf)

## Abstract
Generative Adversarial Networks (GANs) are a modern solution aiming to encourage public sharing of data, even if the data contains inherently private information, by generating synthetic data that looks like, but is not equal to, the data the GAN was trained on. However, GANs are prone to remembering samples from the training data, 
therefore additional care is needed to guarantee privacy. 
Differentially Private (DP) GANs offer a solution to this problem by protecting user privacy through a mathematical guarantee, achieved
by adding carefully constructed noise at specific points in the training process. A state-of-the-art 
example of such a GAN is Gradient Sanitized Wasserstein GAN (GS-WGAN) [1] . 
This model is shown to create higher quality synthetic images than other DP GANs. 
To extend the applicability of GS-WGAN we first reproduce and extend the evaluation, 
verifying that the model outperforms DP-CGAN by an average of 40\% when assessed across three
qualitative metrics and two datasets. Secondly we propose improvements to the architecture and 
training procedure to make GS-WGAN applicable on timeseries data. The experimental results show that GS-WGAN is fit for generating synthetic 
timeseries through promising experimental results.


[1] D. Chen, T. Orekondy, and M. Fritz, “Gs-wgan: A
gradient-sanitized approach for learning differentially
private generators,” 2021.

## General info
This repo contains code from [GS-WGAN](https://arxiv.org/pdf/2006.08265.pdf), and [DP-CGAN](https://arxiv.org/pdf/2001.09700.pdf) <br>

### GS-WGAN
Implementation taken from [the author's implemenation](https://github.com/DingfanChen/GS-WGAN) with minor changes to the model to accept images with different sizes than 28x28 pixels, and added support for PTBand MIT-BIH datasets, both available on [kaggle](kaggle.com/shayanfazeli/heartbeat). 

### DP-CGAN
Code taken from [DP-MERF](https://arxiv.org/abs/2002.11603), which is the same implementation used in the GS-WGAN evaluation. [(link to git repo)](https://github.com/frhrdr/dp-merf)

### Evaluator
Contains a script that can compute Inception Score, Frechet Inception Distance and downstream classifier accuracy for a given GS-WGAN generator (`.pth` file) or a numpy archive containing generated (Fashion-)MNIST samples

## Training
To train GAN training either follow instructions in the corresponding README's in the model subfolders, or use the docker configurations supplied in [docker](docker/)
During training intermediate samples and generators are saved.

They can be run using docker as follows:
* Make sure Docker and docker-compose are installed, then do:
> cd docker/[DP-CGAN|GS-WGAN|eval]/ <br>
> docker-compose up

This starts a docker container which automatically install all dependencies for a run on CPU. <br>

For running without Docker it is easiest to create a new conda environment and install dependencies for the model you want to run. (can be found in `docker/[GS-WGAN|DP-CGAN|eval]/requirements.txt`) Or in the respective model's README.
<br> ! 

## Evaluating
During a training run intermediate samples (for DP-CGAN) or generators (for GS-WGAN) will be saved.<br>
Once training is finished the evaluator can be run via docker or via commandline, an example command:
> python evaluator.py --generator_dir ../../resources/gswgan/fashionmnist/gen --gen_data_size 10000 --dataset fashionmnist --save_location ../../resources/gswgan/fashion/ --num_batches 10

This computes Inception Score, Frechet Inception Distance and downstream classifier accuracy for all generator or sample files in the `--generator_dir`
* Refer to `eval/evaluator/evaluator.py`, or run `python evaluator.py -h` for the complete list of arguments.

### Inception Score
For the IS you first need to train a classifier (seperate for MNIST & FashionMNIST, change in the `MLP_for_Inception.py` file) by running:
> python eval/evaluator/MLP_for_Inception.py
This trains a classifier (depending on the selected dataset) with test set accuracy of 98% or 91% accuracy, respectively for MNIST and Fashion-MNIST.<br>
Line 26 & 27 (`FASHION_MNIST_PATH` & `MNIST_PATH`) Should point to these classifiers if you want to compute the IS.



## Time Series
This GS-WGAN implementation supports two new datasets: PTB & MIT-BIH.
For PTB & MIT-BIH set Kaggle keys in the environment:
>export KAGGLE_USERNAME=username <br>
>export KAGGLE_KEY=key <br>
The datasets will then be automatically downloaded upon training start.

or download the datasets from kaggle: https://www.kaggle.com/shayanfazeli/heartbeat, and create a new folder: `resources/data/ecg/` and put the .csv's there.
