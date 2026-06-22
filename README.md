# CSC3109 - Machine Learning | Team 16

Image classification of Singapore waterway scenes: **beach**, **ferry_terminal**, **harbor**, **river**.

## Setup

```bash
pip install -r requirements.txt
```

## Running in Google Colab

Each notebook contains a **Colab setup cell** at the top. Run it first - it clones the repo and installs dependencies automatically.

## Project Structure

```
CSC3109-T16-Project/
├── data/           # Training & validation images (included in repo)
├── notebooks/      # One notebook per model + EDA
├── src/            # Shared Python modules (dataset, models, train, evaluate)
├── results/        # Saved models, confusion matrices, training curves
├── app/            # FastAPI inference endpoint
└── report/         # Report notes and figures
```

## Models

| Notebook | Model | Pretrained |
|---|---|---|
| `02_custom_cnn.ipynb` | Custom CNN | No |
| `03_resnet18.ipynb` | ResNet-18 | ImageNet |
| `04_efficientnet_b0.ipynb` | EfficientNet-B0 | ImageNet |
| `05_mobilenet_v3.ipynb` | MobileNet V3 Small | ImageNet |
| `06_vit_b16.ipynb` | ViT-B/16 | ImageNet |

## Running the API

```bash
uvicorn app.main:app --reload
# POST /predict  with an image file
```
