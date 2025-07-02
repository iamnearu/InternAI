from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import os

# Load mô hình đã fine-tune
MODEL_DIR = "donut_output"  # thư mục bạn đã save sau khi train

processor = DonutProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()


def run_donut_inference(image_path):
    # Load ảnh
    image = Image.open(image_path).convert("RGB")

    # Encode ảnh và sinh output
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_length=768)
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return sequence
