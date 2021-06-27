# rp-group-42-ptemarvelde
This repository contains the implementation for: [Differentially Private GAN for Time Series](https://www.google.com).

##Abstract:
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

## Training
The root folders in this repo point to the two GAN model implementation compared for this research and the evaluation code. <br>
Each folder has a corresponding folder in the docker root folder, containing a `Dockerfile` and `docker-compose.yml` which can be used to run the training and evaluations. <br>
During training intermediate samples and generators are saved.

They can be run using docker as follows:
* Make sure Docker and docker-compose are installed, then do:
> cd docker/[DP-CGAN|GS-WGAN|eval]/ <br>
> docker-compose up

This starts a docker container which automatically install all dependencies for a run on CPU. <br>

For running without Docker it is easiest to create a new conda environment and install dependencies for the model you want to run. (can be found in `docker/model/requirements.txt`)
<br> ! 
### GS-WGAN
Instructions for running GS-WGAN are on the repo containing the original implementation: https://github.com/DingfanChen/GS-WGAN <br>


### DP-CGAN
The implementation of DP-CGAN (from https://github.com/frhrdr/dp-merf) is a bit less nice to use. <br>
Changes to the configuration need to be made in `dp-merf/dpcgan/dp_cgan_reference_im.py` after that it can be run using Docker or:
>python dp_cgan_reference_im.py

## Evaluating
During a training run intermediate samples (for DP-CGAN) or generators (for GS-WGAN) will be saved.<br>
Once training is finished the evaluator can be run via docker or via commandline, an example command:
> python evaluator.py --generator_dir ../../resources/gswgan/fashionmnist/gen --gen_data_size 10000 --dataset fashionmnist --save_location ../../resources/gswgan/fashion/ --num_batches 10

This computes Inception Score, Frechet Inception Distance and downstream classifier accuracy for all generator or sample files in the `--generator_dir`
* Refer to `eval/evaluator/evaluator.py` for the complete list of arguments.

### Inception Score
For the IS you first need to train a classifier (seperate for MNIST & FashionMNIST, change in the `MLP_for_Inception.py` file) by running:
> python eval/evaluator/MLP_for_Inception.py



## Time Series
This GS-WGAN implementation supports two new datasets: PTB & MIT-BIH.
For PTB & MIT-BIH set Kaggle keys in the environment:
>export KAGGLE_USERNAME=username <br>
>export KAGGLE_KEY=key

or download the datasets from kaggle: https://www.kaggle.com/shayanfazeli/heartbeat