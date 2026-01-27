# src/training/train.py
import os
import json
import yaml
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import f1_score, accuracy_score

from .dataset import load_jsonl, build_label_maps, NaicsTextDataset


def compute_metrics(eval_pred):
    """
    你需要知道：eval_pred = (logits, labels)
    - logits: [batch, num_labels]
    - labels: [batch]
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main(config_path: str):
    # =========================
    # 读取 YAML 配置
    # 工业训练必须 config-driven 才可复现、可调参
    # =========================
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Empty config")

    # 固定随机种子：保证可复现
    set_seed(int(cfg["seed"]))

    # =========================
    # 读取 train/val 数据
    # label mapping 一定从 TRAIN 建（val 出现新 label 应该报错）
    # =========================
    train_rows = load_jsonl(cfg["train_file"])
    val_rows = load_jsonl(cfg["val_file"])
    label2id, id2label = build_label_maps(train_rows)

    # =========================
    # TODO 3: 初始化 tokenizer + model
    # 你要知道：num_labels 必须 == len(label2id)，否则 head 维度不对
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    

    # =========================
    # TODO 4: 构建 Dataset
    # 你要知道：这里用你写的 NaicsTextDataset
    # full_text 拼接规则在 dataset.py 里已经固定了
    # =========================
    train_ds = NaicsTextDataset(train_rows,label2id,tokenizer,int(cfg["max_length"]))
    val_ds = NaicsTextDataset(val_rows,label2id,tokenizer,int(cfg["max_length"]))

    # TrainingArguments 这些我先给你一个合理默认值（你后面会学会调参）
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        logging_steps=int(cfg["logging_steps"]),
        evaluation_strategy=str(cfg["eval_strategy"]),
        save_strategy=str(cfg["save_strategy"]),
        save_total_limit=int(cfg["save_total_limit"]),
        fp16=bool(cfg["fp16"]),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    # =========================
    # TODO 5: 保存 artifacts（工业界非常关键）
    # 你要知道：推理端只需要读 artifacts 就能跑
    # 要保存：
    #   - tokenizer/  (tokenizer.json/vocab等)
    #   - model/      (HF权重+config)
    #   - label_list.json (index -> code)，对标你工作里的 labellist
    # =========================
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    tokenizer.save_pretrained(os.path.join(out_dir,"tokenizer"))
    
    trainer.save_model(os.path.join(out_dir,"model"))
    
    label_list = [id2label[i] for i in range(len(id2label))]
    
    with open(os.path.join(out_dir,"label_list.json"),"w", encoding= "utf-8") as f:
        json.dump(label_list,f,ensure_ascii=False,indent=2)
        
    print("Saved artifacts to:", out_dir)
    
    metadata = {
    "model_name": cfg["model_name"],
    "max_length": int(cfg["max_length"]),
    "num_labels": len(label2id),
    }

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 运行：python -m src.training.train configs/train.yaml
    import sys
    main(sys.argv[1])
