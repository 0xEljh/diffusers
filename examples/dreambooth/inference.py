from diffusers import StableDiffusionPipeline
import torch
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model/model id")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text. Include class identifier (e.g. sks)")
    parser.add_argument("--num_samples", type=int, default=1)
    return parser.parse_args()


def main(args):
    model_id = args.model
    # make no assumptions about model datatype.
    # not possible to use more vram than training
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")

    prompt = args.prompt
    images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images
    print(len(images))

    now = str(datetime.now()).replace(" ", "_")
    for i, image in enumerate(images):
        image.save(f"output/{now}_{i}.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
