import os
from PIL import Image
import numpy as np
import albumentations as A

class DonutDataset:
    def __init__(self, hf_dataset, image_root_dir, processor, augment=False):
        self.dataset = hf_dataset
        self.image_root_dir = image_root_dir
        self.processor = processor
        self.augment = False

        # Define augmentation pipeline (nhẹ nhàng)
        self.transform = A.Compose([
            A.Rotate(limit=3, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ImageCompression(quality_lower=80, quality_upper=95, p=0.3)

        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image_path = os.path.join(self.image_root_dir, example["image"])
        image = Image.open(image_path).convert("RGB")

        if self.augment:
            image_np = np.array(image)
            image_np = self.transform(image=image_np)["image"]
            image = Image.fromarray(image_np)

        # Tạo pixel values (ảnh)
        inputs = self.processor(
            image,
            return_tensors="pt"
        )

        # Tokenize text đích để tạo labels
        labels = self.processor.tokenizer(
            example["target"],
            add_special_tokens=True,
            return_tensors="pt"
        ).input_ids

        # Chuyển padding thành -100 (để Trainer bỏ qua)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "labels": labels.squeeze(0)
        }
