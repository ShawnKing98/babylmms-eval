import os
from PIL import Image
import torch

def baby_winoground_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # Returns a loaded image
    if doc["image_tag"] == "positive":
        base_path = "dataset/baby_winoground/positive_images"
    elif doc["image_tag"] == "negative":
        base_path = "dataset/baby_winoground/negative_images"
    else:
        raise ValueError(f"Invalid image tag: {doc['image_tag']}")
    image = Image.open(os.path.join(base_path, doc["image"]))
    return [image]

def baby_winoground_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{post_prompt}"


def baby_winoground_doc_to_target(doc):
    # Seems like a dummy function that is not really used?
    return doc['image_tag']

def baby_winoground_doc_to_choice(doc):
    # Choices are the two phrases
    return [doc["positive_phrase"], doc["negative_phrase"]]

def baby_winoground_process_results(doc, results):
    # doc: corresponding image and phrases
    # results: list of (loss, is_greedy) for two phrases and one image
    if doc["image_tag"] == "positive":
        scores = {"img0_phrase0_loss": results[0][0], "img0_phrase1_loss": results[1][0], "instance_id": doc["instance_id"]}
    elif doc["image_tag"] == "negative":
        scores = {"img1_phrase0_loss": results[0][0], "img1_phrase1_loss": results[1][0], "instance_id": doc["instance_id"]}
    else:
        raise ValueError(f"Invalid image tag: {doc['image_tag']}")
    return {"image_score": scores, "text_score": scores, "positive_context_group_score": scores, "negative_context_group_score": scores, "group_score": scores}

def get_loss_matrix(items):
    losses = [{} for _ in range(len(items) // 2)]
    for item in items:
        losses[item["instance_id"]].update(item)
    # Convert list of dicts to tensor
    loss_matrix = torch.zeros((len(losses), 2, 2))
    for i, item in enumerate(losses):
        # Fill in the 2x2 matrix for each instance
        loss_matrix[i, 0, 0] = item["img0_phrase0_loss"]
        loss_matrix[i, 0, 1] = item["img0_phrase1_loss"] 
        loss_matrix[i, 1, 0] = item["img1_phrase0_loss"]
        loss_matrix[i, 1, 1] = item["img1_phrase1_loss"]
    return loss_matrix
    
def baby_winoground_aggregate_image_score(items):
    # items: list of dicts, returned by process_results
    loss_matrix = get_loss_matrix(items)
    image_score = (loss_matrix[:, 0, 0] < loss_matrix[:, 1, 0]) & (loss_matrix[:, 1, 1] < loss_matrix[:, 0, 1])
    return image_score.float().mean().item()

def baby_winoground_aggregate_text_score(items):
    loss_matrix = get_loss_matrix(items)
    text_score = (loss_matrix[:, 0, 0] < loss_matrix[:, 0, 1]) & (loss_matrix[:, 1, 1] < loss_matrix[:, 1, 0])
    return text_score.float().mean().item()

def baby_winoground_aggregate_positive_context_group_score(items):
    loss_matrix = get_loss_matrix(items)
    positive_context_group_score = (loss_matrix[:, 0, 0] < loss_matrix[:, 1, 0]) & (loss_matrix[:, 0, 0] < loss_matrix[:, 0, 1])
    return positive_context_group_score.float().mean().item()

def baby_winoground_aggregate_negative_context_group_score(items):
    loss_matrix = get_loss_matrix(items)
    negative_context_group_score = (loss_matrix[:, 1, 1] < loss_matrix[:, 0, 1]) & (loss_matrix[:, 1, 1] < loss_matrix[:, 1, 0])
    return negative_context_group_score.float().mean().item()

def baby_winoground_aggregate_group_score(items):
    loss_matrix = get_loss_matrix(items)
    group_score = (loss_matrix[:, 0, 0] < loss_matrix[:, 1, 0]) & (loss_matrix[:, 1, 1] < loss_matrix[:, 0, 1]) \
        & (loss_matrix[:, 0, 0] < loss_matrix[:, 0, 1]) & (loss_matrix[:, 1, 1] < loss_matrix[:, 1, 0])
    return group_score.float().mean().item()
