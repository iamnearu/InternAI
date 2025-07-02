import os
import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers import Seq2SeqTrainer
from datasets import load_dataset
from dataset.donut_dataset import DonutDataset
from utils.custom_collator import donut_collator

# -----------------------------
# 1. Cấu hình cơ bản
# -----------------------------
MODEL_NAME = "naver-clova-ix/donut-base"
IMAGE_DIR = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\testsets"
TRAIN_CSV = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\donut_labels_train.csv"
VAL_CSV = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\donut_labels_val.csv"
OUTPUT_DIR = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\donut_output"
BATCH_SIZE = 2
EPOCHS = 10

# -----------------------------
# 2. Load processor & model
# -----------------------------
processor = DonutProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# 3. Load datasets
# -----------------------------
train_data = load_dataset("csv", data_files=TRAIN_CSV)["train"]
val_data = load_dataset("csv", data_files=VAL_CSV)["train"]

train_dataset = DonutDataset(train_data, IMAGE_DIR, processor)
val_dataset = DonutDataset(val_data, IMAGE_DIR, processor)

# -----------------------------
# 4. Training arguments
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",  # warning: deprecated, sẽ sửa sau nếu cần
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# -----------------------------
# 5. Subclass Trainer để fix lỗi forward()
# -----------------------------
class DonutTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v for k, v in inputs.items() if k in ["pixel_values", "labels"]}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# -----------------------------
# 6. Trainer + EarlyStopping
# -----------------------------
trainer = DonutTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,  # không cảnh báo nếu dùng tokenizer riêng
    data_collator=donut_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# -----------------------------
# 7. Train
# -----------------------------
if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
