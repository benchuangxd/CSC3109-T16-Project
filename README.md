# CSC3109 - Machine Learning | Team 16

Image classification of Singapore waterway scenes: **beach**, **ferry_terminal**, **harbor**, **river**.

## Setup

```bash
pip install -r requirements.txt
```

## Running in Google Colab

**Step 1** — Open a notebook directly in Colab:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → GitHub tab**
3. Paste the repo URL: `https://github.com/benchuangxd/CSC3109-T16-Project`
4. Select the notebook you want to run

Or click an **Open in Colab** badge below.

**Step 2** — Run the first cell in the notebook. It clones the repo (for `src/` and `data/`) and installs all dependencies automatically.

### Notebooks

| Notebook | Description | Open in Colab |
|---|---|---|
| `01_eda.ipynb` | Exploratory Data Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchuangxd/CSC3109-T16-Project/blob/main/notebooks/01_eda.ipynb) |
| `02_custom_cnn.ipynb` | Custom CNN (Daniel) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchuangxd/CSC3109-T16-Project/blob/main/notebooks/02_custom_cnn.ipynb) |
| `03_resnet18.ipynb` | ResNet-18 (Han Sheng) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchuangxd/CSC3109-T16-Project/blob/main/notebooks/03_resnet18.ipynb) |
| `04_efficientnet_b0.ipynb` | EfficientNet-B0 (Haley) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchuangxd/CSC3109-T16-Project/blob/main/notebooks/04_efficientnet_b0.ipynb) |
| `05_mobilenet_v3.ipynb` | MobileNet V3 (Jocasta) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchuangxd/CSC3109-T16-Project/blob/main/notebooks/05_mobilenet_v3.ipynb) |
| `06_vit_b16.ipynb` | ViT-B/16 (Jun Hao) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchuangxd/CSC3109-T16-Project/blob/main/notebooks/06_vit_b16.ipynb) |
| `07_comparison.ipynb` | Model Comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benchuangxd/CSC3109-T16-Project/blob/main/notebooks/07_comparison.ipynb) |

## Project Structure

```
CSC3109-T16-Project/
├── data/           # Training & validation images (included in repo)
├── notebooks/      # One notebook per model + EDA
├── src/            # Shared Python modules (dataset, models, train, evaluate)
├── results/        # Saved models, confusion matrices, training curves
├── app/            # Streamlit inference app (Dockerized)
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

## Running the App

```bash
cd app
pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501 and upload an image to see the prediction.

Or run it containerised:

```bash
cd app
docker build -t csc3109-t16:1.0 .
docker run -p 8501:8501 csc3109-t16:1.0
```
