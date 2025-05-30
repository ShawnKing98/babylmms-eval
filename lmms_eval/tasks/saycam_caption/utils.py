import os
from PIL import Image
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# Given a doc, return the loaded image
def saycam_caption_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    image_path = os.path.join("dataset/SAYCam_caption/images", doc["image"])
    image = Image.open(image_path)
    return [image.convert("RGB")]

# Given a doc, return the prompt
def saycam_caption_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{post_prompt}"

# Given a doc and model result, return the dict for metric computation
def saycam_caption_process_result(doc, result):
    # Collect prediction and ground truth for later batch processing
    info = {"gt": doc["text"], "pred": result[0]}
    return {"meteor": info}

# Aggregation for METEOR using nltk's meteor_score
def saycam_caption_aggregation_result(results, args=None):
    meteor_scores = []
    for item in results:
        gt = item["gt"]
        pred = item["pred"]
        # meteor_score expects references as a list of strings, and a single hypothesis string
        score = meteor_score([word_tokenize(gt)], word_tokenize(pred))
        meteor_scores.append(score)
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    return avg_meteor