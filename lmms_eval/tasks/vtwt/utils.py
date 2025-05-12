import os
from PIL import Image


def vtwt_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # Allow user to override image path, otherwise use default
    base_path = "dataset/vtwt/images"
    image_path = os.path.join(base_path, doc["image"])
    image = Image.open(image_path)
    return [image]

def vtwt_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{post_prompt}"

def vtwt_doc_to_target(doc):
    return doc["positive_phrase"]

def vtwt_doc_to_choice(doc):
    return [doc["positive_phrase"], doc["negative_phrase"]]

def vtwt_process_results(doc, results):
    # results: list of (loss, is_greedy) for each choice
    pred = int(min(range(len(results)), key=lambda i: results[i][0]))
    gold = 0  # positive_phrase is always the ground truth (index 0)
    acc = 1.0 if pred == gold else 0.0
    return {"acc": acc} 