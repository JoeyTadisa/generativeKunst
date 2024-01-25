import argparse
import re
import os
import torch
import numpy as np
import random
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Images from Lora Weights")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs='+',
        # default=["Create abstract artwork by Christin Kirchner in the curves and swatches collection.",
        #          "Create abstract artwork by Christin Kirchner in unlimited horizons and abstract collection.",
        #          "Create abstract artwork by Christin Kirchner in the black line collection.",
        #          "Create abstract artwork by Christin Kirchner in the dynamic spiral collection."
        #          "Create abstract artwork by Christin Kirchner in the unlimited horizons collection."
        #          ],
        # default=["Create abstract artwork by Christin Kirchner"],
        default=["Create abstract portrait ofs Barack Obama by Christin Kirchner."],
        help="prompts to generate the images",
    )
    
    parser.add_argument("--height", type=int, default=512, help="height of the generated image")
    parser.add_argument("--width", type=int, default=512, help="width of the generated image")
    parser.add_argument("--guidance_scale", type=float, default=8.0, help="guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=3, help="number of images to generate per prompt")

    parser.add_argument(
            "--model_steps",
            type=str,
            nargs='+',
            default=[
                #"./models/pretrained/runwayml/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt:15",
                     #"./models/lora/kirchner_lower_lr:15", #fine-tuned pre-dataset augmentation
                     #"./models/lora/kirchner_12_12_2023_silvery_sweep_pre_dataset_update_lower_t_steps:15", #minimized for FID
                     "./models/lora/kirchner_18_12_2023_test_stellar_sweep_subjective_best_final:43", # subjective
                     #"./models/lora/kirchner_18_12_23_simplified_captions/pytorch_lora_weights_80ve6qqh.safetensors:19", # min fid denim sweep 4
                     #"./models/lora/kirchner_18_12_2023_test_spring_sweep_max_inception:25",
                     #"./models/lora/kirchner_18_12_23_simplified_captions/pytorch_lora_weights_vz0gsblc.safetensors:33" #max-msssim
                     ],
            help="the paths to the trained model files and their corresponding inference steps",
        )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./outputs/demo/",
        help="the path to the base folder to hold generated images",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="inference steps",
    )
    # parser.add_argument("--eta", type=float, default=0.0, help="eta parameter for DDIMScheduler")

    args = parser.parse_args()
    return args

def is_black(img):
    return np.all(np.array(img) == 0)

def seed_everything(seed=None):
    seed = seed or torch.randint(1, 1_000_000, (1,)).item()
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def main():
    args = parse_args()
    #file_name = re.sub(r'\W+', '-', args.prompts)
    
    os.makedirs(args.output_folder, exist_ok=True)

    seed = seed_everything(42)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    attempts = 0
    valid_img_counter = 0
    max_attempts =  args.num_images_per_prompt * 2 
    images = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = 'runwayml/stable-diffusion-v1-5'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16) if torch.cuda.is_available() else StableDiffusionPipeline.from_pretrained(model_id)
    pipe.height = args.height
    pipe.width = args.width
    print(f'device is {device}')

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    for model_step in args.model_steps:
        print(f'Model: {model_step}')
        model_path, steps = model_step.split(":")
        steps = int(steps)
        model_name = os.path.basename(model_path)
        pipe.to(device)
        model_output_folder = os.path.join(args.output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)
        pipe.unet.load_attn_procs(model_path)
        attempts = 0
        valid_img_counter = 0
        images = []
        while len(images) < args.num_images_per_prompt and attempts < max_attempts:
            prompt = random.choice(args.prompts)
            file_name = re.sub(r'\W+', '-', prompt)
            image = pipe(prompt, num_inference_steps=steps,generator=generator, guidance_scale = args.guidance_scale).images[0]
            
            if not is_black(image):
                valid_img_counter += 1
                print(f'valid: {valid_img_counter}')
                images.append(image)
                image.save(os.path.join(model_output_folder, f"{file_name}_img_{valid_img_counter}_seed{seed}_scale{args.guidance_scale}.png"))
            attempts += 1
            print(f'attempts: {attempts}')

if __name__ == "__main__":
    main()
