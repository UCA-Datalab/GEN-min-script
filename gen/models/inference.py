from transformers import BertTokenizer
import torch
from torchvision import transforms
import pandas as pd
from gen.models.classification_BertNet import BertNet
from gen.models.segmentation_LSTM import Segmentator
from typing import Iterable
import json
from typing import List

def process_text(text, tokenizer):
    text_tokenized = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"]

    return text_tokenized


def inference(
    images: Iterable,
    texts: Iterable[str],
    batch_size: int = 32,
    cuda: bool = False,
    encoder_path: str = "models/BertNet_group_classes.pt",
    segmentator_model_path: str = "models/LSTM_group_classes.pt",
) -> List[int]:
    device = (
        torch.device("cuda")
        if cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )

    with open("labels.json","r") as f:
        labels = json.load(f)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    images = [image_transforms(image) for image in images]

    tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

    # ===== CREATE DATASETS AND DATALOADERS =====
    encoder = BertNet(len(labels)).to(device)
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()

    model = Segmentator(n_classes=len(labels), input_size=encoder.fc.in_features).to(
        device
    )
    model.load_state_dict(torch.load(segmentator_model_path))
    model.eval()

    df_dicts = []
    images_stacked = torch.stack(images).to(device)

    texts_tokenized = process_text(texts, tokenizer).to(device)

    encoded_results = []
    for i in range(0, len(images), batch_size):
        im = images_stacked[i : i + batch_size].to(device)
        t = texts_tokenized[i : i + batch_size].to(device)
        with torch.no_grad():
            _, encoded_page = encoder(t, im)

        encoded_results.append(encoded_page)

    encoded_pages = torch.cat(encoded_results).unsqueeze(0)
    with torch.no_grad():
        outputs = (
            model(encoded_pages, torch.ones((1, encoded_pages.shape[1])))
            .squeeze(0)
            .cpu()
        )

    outputs = torch.argmax(outputs, 1)
    results_labeled = [labels[r] for r in outputs]
    return results_labeled
