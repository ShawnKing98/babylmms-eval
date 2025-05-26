import os
from PIL import Image
import numpy as np

def labeled_s_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    base_path = "dataset/labeled_s/images"
    image_path = os.path.join(base_path, doc["image"])
    image = Image.open(image_path)
    return [image]

def labeled_s_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{post_prompt}"

def labeled_s_doc_to_target(doc):
    return doc["target_category"]

def labeled_s_doc_to_choice(doc):
    return [doc["target_category"]]

def labeled_s_process_results(doc, results):
    # doc: an entry of the dataset csv
    # results: list of a single (loss, is_greedy)
    info = {"loss": results[0][0], "trial_id": doc["trial_id"], "target_category": doc["target_category"], "real_category": doc["real_category"]}
    return {"acc": info}

def labeled_s_aggregate_acc(items):
    lowest_loss = [np.inf] * (len(items) // 4)
    acc = [None] * (len(items) // 4)
    for item in items:
        if item["loss"] < lowest_loss[item["trial_id"]]:
            lowest_loss[item["trial_id"]] = item["loss"]
            if item["real_category"] == item["target_category"]:
                acc[item["trial_id"]] = 1
            else:
                acc[item["trial_id"]] = 0
    return sum(acc) / len(acc)