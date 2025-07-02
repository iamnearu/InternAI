import torch

def donut_collator(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = [example["labels"] for example in batch]

    # Pad labels bằng tay
    max_len = max(l.size(0) for l in labels)
    padded_labels = torch.full((len(labels), max_len), fill_value=-100)

    for i, label in enumerate(labels):
        padded_labels[i, :label.size(0)] = label

    return {
        "pixel_values": pixel_values,
        "labels": padded_labels
    }
