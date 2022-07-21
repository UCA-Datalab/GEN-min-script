import numpy as np
import easyocr
import re
from transformers import BertTokenizer
from pdf2image import convert_from_bytes


def ocr_with_rotation(ocr_reader, image, tokenizer):
    best_ocr = ""
    best_score = 0

    for i in range(4):
        rotation_degree = i * 90
        rotated_image = image.rotate(rotation_degree)
        text = " ".join(
            ocr_reader.readtext(
                np.array(rotated_image), detail=0, workers=4, batch_size=32
            )
        )

        tokenized_text = tokenizer.tokenize(text)

        score = 0
        for t in tokenized_text:
            if (
                t != "[UNK]"
                and not re.search(
                    '#|@|"|\{|\}|\(|\)|=|\-|\+|\?|\!|\<|\>|:|[0-9]|\.|\_|\*|%|\;', t
                )
                and len(t) > 2
            ):
                score += 1

        if i == 0 or score > best_score:
            best_score = score
            best_ocr = text

        if best_score > 20:
            break

    return best_ocr


def get_images(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes)
    return pages


def process_pdf(pdf_bytes):
    images = get_images(pdf_bytes)

    ocr_reader = easyocr.Reader(["es"])
    tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

    texts = []
    for image in images:
        texts.append(ocr_with_rotation(ocr_reader, image, tokenizer))

    return images, texts
