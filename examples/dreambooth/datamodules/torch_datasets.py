import random
from pathlib import Path
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

text_templates = [
    "a painting of {}",
    "a photo of {}",
    "a picture of {}",
    # "a drawing of {}",
    # "a sketch of {}",
    "a portrait of {}",
    "an image of {}",
]


def word_count(text):
    return len(text.split(" "))


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        with_prior_preservation=True,
        size=512,
        center_crop=False,
        num_class_images=None,
        pad_tokens=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.pad_tokens = pad_tokens

        self.instance_images_path = []
        self.class_images_path = []

        for concept in concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_img_path = [
                    (x, concept["class_prompt"]) for x in Path(concept["class_data_dir"]).iterdir() if x.is_file()
                ]
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)

        # may want to consider applying different transforms for class and instance images.
        # class images can be augmented more heavily

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
                            transforms.RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2), scale=(0.8, 1.2)),
                        ]
                    ),
                    p=0.1,
                ),
                transforms.RandomGrayscale(p=0.02),  # might be useful for the model to learn the color of the object
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # augmentation pipeline for regularization/class images
        self.reg_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2),
                            transforms.RandomAffine(degrees=(-20, 20), translate=(0.2, 0.2), scale=(0.8, 1.2)),
                        ]
                    ),
                    p=0.7,
                ),
                transforms.RandomGrayscale(p=0.1),  # might be useful for the model to learn the color of the object
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):

        example = {}
        instance_path, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        # replace instance prompt with a random text template if it is just a single word + identifier
        # find identifier in the prompt and remove it for counting the number of words
        # currently assume identifier is sks
        if word_count(instance_prompt.replace("sks", "")) > 1:
            instance_prompt = random.choice(text_templates).format(instance_prompt)
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="max_length" if self.pad_tokens else "do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.with_prior_preservation:
            class_path, class_prompt = self.class_images_path[index % self.num_class_images]

            if word_count(class_prompt) > 1:
                class_prompt = random.choice(text_templates).format(class_prompt)

            class_image = Image.open(class_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            # example["class_images"] = self.image_transforms(class_image)
            example["class_images"] = self.reg_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="max_length" if self.pad_tokens else "do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]
