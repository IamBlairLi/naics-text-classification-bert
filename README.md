# NAICS Text Classifier (DistilBERT)

Production-style NAICS/industry classification using DistilBERT.
Train on (business_name + text) -> NAICS code, export artifacts, and later serve via an inference SDK + API.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.training.train configs/train.yaml
