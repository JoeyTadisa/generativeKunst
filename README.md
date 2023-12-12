# Exploring Generative Models for Abstract Art Synthesis
Exploring Generative Models for Artistic Synthesis in fulfilment of my Bachelor's Thesis.

Wandb.ai was used to help track experiments. The link to the supporting experimental runs can be found here: [Kunst Thesis | Wandb.ai](https://wandb.ai/kunst-thesis/projects)

Please note that, for the fine-tuning scripts, the templates/tutorials made by HUGGINGFACE were used to make these. The tutorial can be found here: [Text-to-Image Tutorial | Huggingface](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

# Dataset
The dataset was created using the images from Christin Kirchner's website with her consent. For more of her work, refer to her website here: [Christin Kirchner](https://www.christin-kirchner.com/).
In order to reproduce the dataset, run the following script: `python3 prepare_data.py`

## Docker Environment
It is recommended that you have a GPU device available in order to run the following scripts. In my case, an NVIDIA GeForce RTX 3090 was used.

Additional environmental information is as follows:
NVIDIA-SMI Version: 535.129.03
Driver Version: 535.129.03
CUDA Version: 12.2

For a look into the list of requirements, please refer to the *'requirements.txt'* file.
Assuming, you have docker already installed and that you are in the same directory as the Dockerfile:
1. Run the following command: `docker build . -t <CONTAINER_NAME>` Expect the image to be roughly 15GB in size. 
2. Run the following command: `docker run -it --gpus device=<DEVICE_NUMBER> -v "$(pwd)":/<HOME_DIRECTORY_IN_CONTAINER> <CONTAINER_NAME>`. If you desire to adjust the run command further, please refer to the official docker documentation. Otherwise, this command should provide your container with a GPU device. The -v tag will mount and share the program files and directory over to your container with your host machine. Meaning whatever is also generated in the container is also automatically available in the host machine's directory.
3. Once inside the container, you should find bash scripts that follow the structure, `tune_<IDENTIFIER>.sh`. These will faciliate the parameters, directories and similar information for a fine-tuning a stable diffusion model.

## Overview of the models used
(A) Baseline: pretrained stable_diffusion_v1.5 TODO: simple script to run

(B) Fine-tuned: kirchner_lower_lr
In order to reproduce this model, please run the following command inside the container, `sh tune_kirchner.sh`

(C) Adjust hyperparameters using FID: `kirchner_12_12_2023_silvery_sweep_pre_dataset_update_lower_t_steps`
In order to reproduce this model, please run the following command inside the container, `sh tune_silvery_sweep_1.sh`
The best model parameters were chosen by selecting the model that was able to achieve the lowest FID score in it's FID-curve against max_training_steps. (the minimum of the curve - thus silvery-sweep-1 hyperparameters were used.)

(D) Dataset improvement: TBD - with updates to the meta description to include the collection as part of the description.

(E) Augmentation: TBD - with reference to the setups that brought improvements to the GANs performance in a similar domain.
