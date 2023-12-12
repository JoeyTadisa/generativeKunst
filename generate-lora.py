import argparse
import re
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Images from Lora Weights")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Create a painting by Christin Kirchner",
        help="prompt to generate the image",
    )
    
    parser.add_argument("--height", type=int, default=512, help="height of the generated image")
    parser.add_argument("--width", type=int, default=512, help="width of the generated image")
    parser.add_argument("--guidance_scale", type=float, default=8.0, help="guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=50, help="number of images to generate per prompt")

    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/lora/kirchner_12_12_2023_silvery_sweep_pre_dataset_update_lower_t_steps",
        help="the path to the trained model file",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./outputs/lora/12_12_2023_silvery_sweep_pre_dataset_update_lower_t_steps_v2",
        help="the path to folder to hold generated images",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="inference steps",
    )
    # parser.add_argument("--eta", type=float, default=0.0, help="eta parameter for DDIMScheduler")

    args = parser.parse_args()
    return args

def seed_everything(seed=None):
    seed = seed or torch.randint(1, 1_000_000, (1,)).item()
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def main():
    args = parse_args()
    file_name = re.sub(r'\W+', '-', args.prompt)
    
    os.makedirs(args.output_folder, exist_ok=True)

    seed = seed_everything(-1)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = 'runwayml/stable-diffusion-v1-5'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16) if torch.cuda.is_available() else StableDiffusionPipeline.from_pretrained(model_id)
    pipe.height = args.height
    pipe.width = args.width
    print(f'device is {device}')

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.load_attn_procs(args.model_path)
    pipe.to(device)

    for i in range(args.num_images_per_prompt):
        image = pipe(args.prompt, num_inference_steps=args.steps,generator=generator, guidance_scale = args.guidance_scale).images[0]
        image.save(os.path.join(args.output_folder, f"{file_name}_img_{i + 1}_seed{seed}_scale{args.guidance_scale}.png"))

if __name__ == "__main__":
    main()
