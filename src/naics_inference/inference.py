import json
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class Prediction:
    code: str
    prob: float


class InferenceModel:
    def __init__(self, artifact_dir: str, device: str = "cpu", debug: bool = False):
        self.debug = debug
        self.artifact_dir = artifact_dir
        self.device = device

        model_dir = f"{artifact_dir}/model"
        meta_path = f"{artifact_dir}/metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            self.default_max_length = int(self.metadata["max_length"])
        
        if self.debug:
            print("default_max_length:", self.default_max_length)

        tok_dir = f"{artifact_dir}/tokenizer"
        label_path = f"{artifact_dir}/label_list.json"

        # ✅ TODO 1: load label_list.json -> self.label_list (list[str])
        with open(label_path, "r", encoding="utf-8") as f:
             self.label_list = json.load(f)

        # print datapoint (可先注释，跑通后打开)
        if self.debug:
            print("label_list len:", len(self.label_list), "sample:", self.label_list[:3])
            print(type(self.label_list), len(self.label_list), self.label_list[:3])

        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, business_name: str, text: str, topk: int = 3, max_length: int | None = None, debug: bool | None = None) -> List[Prediction]:
        # ✅ TODO 2: tokenize -> inputs dict with input_ids/attention_mask
        dbg = self.debug if debug is None else debug
        full_text = (business_name + " " + text).strip()
        if max_length is None:
            max_length = self.default_max_length

        inputs = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
         )

        # print datapoint
        if dbg:
            print("input_ids shape:", inputs["input_ids"].shape)
            print("mask tail:", inputs["attention_mask"][0][-10:].tolist())

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].detach().cpu().numpy()  # shape: [num_labels]
            assert len(self.label_list) == logits.shape[0], (len(self.label_list), logits.shape)

        # ✅ TODO 3: logits -> probs (softmax)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        top_idx = np.argsort(-probs)[:topk]

        # print datapoint
        if dbg:
            print("logits shape:", logits.shape, "top_idx:", top_idx.tolist())

        # ✅ TODO 4: map idx -> code using self.label_list
        preds = [Prediction(code=self.label_list[i], prob=float(probs[i])) for i in top_idx]
        return preds