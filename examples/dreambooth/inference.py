from diffusers import StableDiffusionPipeline
import torch
import argparse
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model/model id")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text. Include class identifier (e.g. sks)")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    return parser.parse_args()


prompt_templates = {
    "cyberpunk": "grungy portrait cyberpunk futuristic neon, hong kong streets, decorated with traditional japanese ornaments by ismail inceoglu dragan bibin hans thoma greg rutkowski Alexandros Pyromallis Nekro Rene Margitte illustrated Perfect face, fine details, realistic shaded, fine-face, pretty face",
    "floral": "A portrait with isolated flowers with strong dark comic outlines, colorful, psychedelic, intricate, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by greg rutkowski and aubrey beardsley",
    "psychedelic": "Duotone trippy 1960s lsd concept illustration. volumetric lighting. golden ratio accidental renaissance. by sachin teng and sergey kolesov and ruan jia and heng z. graffiti art, scifi, fantasy, hyper detailed. octane render. concept art. trending on artstation.",
    "royal": "A soft and breathtaking detailed portrait painting with a crown on their head in the style of gustav klimt, shiny gold, elegant, highly detailed, artstation, fluo colors, concept art, matte, sharp focus, art by gustav klimt and alphonse mucha",
    "horror": "Portrait highly detailed, skeleton make up, worms coming out of eyes, bloody, skulls background, rpg, highly detailed, (hdr),center, Greg Rutkowski, very inspirational, maskerart, realistic",
    "oil": "portrait in oil painting in the style of michelangelo",
    "colorful": "portrait by kenne gregoire, james jean, tran nguyen, wlop, jakub rebelka. trending on artstation, 8k, masterpiece, chill summer, graffiti paint, fine detail, full of color, intricate detail, golden ratio illustration",
    "occult": "Portrait of intricate, highly detailed, occult imagery, digital painting, artstation, concept art, illustration, by gil elvgen, greg manchess, mucha",
}


def main(args):
    model_id = args.model
    # make no assumptions about model datatype.
    # not possible to use more vram than training
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    if args.prompt == "all":
        prompt = ["sks person " + x for x in prompt_templates.values()]
    else:
        prompt_text = (
            args.prompt if args.prompt not in prompt_templates else ("sks person " + prompt_templates[args.prompt])
        )
        prompt = [prompt_text] * args.num_samples  # no. of prompts = no. of images generated
    images = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance_scale).images
    print(f"Generated {len(images)} images")

    # get current time in format: 2021-07-01_12-00-00
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i, image in enumerate(images):
        # image name = model_name + prompt + index
        if args.prompt == "all":
            prompt = list(prompt_templates.keys())[i]
        else:
            prompt = args.prompt

        # get model name:
        model_name = Path(args.model).stem

        # save images to parent directory of model:
        filepath = Path(args.model).parent / f"{model_name}_{now}_{prompt}_{i}.png"
        image.save(filepath)


if __name__ == "__main__":
    args = parse_args()
    main(args)
