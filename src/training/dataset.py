# src/training/dataset.py
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


# =========================
#  Load JSONL
# =========================
def load_jsonl(path: str) -> List[Dict]:
    """
    Read a JSONL file where each line is a JSON object.
    Expected keys per row: business_name, text, label
    """
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# =========================
#  Build label maps
# =========================
def build_label_maps(rows: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label2id/id2label from TRAIN rows only.
    Using sorted() keeps mapping deterministic.
    """
    labels = sorted({r["label"] for r in rows})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


@dataclass
class NaicsExample:
    text: str
    label_id: int


class NaicsTextDataset(Dataset):
    """
    Minimal dataset for classification:
      input text = business_name + " " + text
      output label = label_id (int)
    """

    def __init__(self, rows: List[Dict], label2id: Dict[str, int], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[NaicsExample] = []

        for r in rows:
            business_name = (r.get("business_name") or "").strip()
            text = (r.get("text") or "").strip()

            # =========================
            # Text concatenation rule
            # =========================
            full_text = (business_name + " " + text).strip()

            label = r["label"]
            if label not in label2id:
                raise ValueError(f"Label '{label}' not in label2id. Did you build mapping from TRAIN split?")

            self.examples.append(NaicsExample(text=full_text, label_id=label2id[label]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        enc = self.tokenizer(
            ex.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        #batch size from (1,max_len) -> (max_len,)
        #without squeeze dataloader: (b,1,l)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex.label_id, dtype=torch.long)
        return item
