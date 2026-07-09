# Report Notes - CSC3109 Team 16

## Dataset
- 4 classes: beach, ferry_terminal, harbor, river
- Train: 2,800 images (700/class) | Val: 400 images (100/class)
- All images 256×256 px; resized/cropped to 224×224 for models

## Models Trained
| Model | Member | Pretrained |
|---|---|---|
| Custom CNN | Member 1 | No |
| ResNet-18 | Member 2 | ImageNet |
| EfficientNet-B0 | Member 3 | ImageNet |
| MobileNet V3 Small | Member 4 | ImageNet |
| ViT-B/16 | Member 5 | ImageNet |

## Results

Evaluated on the 30% validation split (840 images, 210 per class).

| Model | Accuracy | Precision | Recall | F1 | Params (M) | Pretrained |
|---|---|---|---|---|---|---|
| Custom CNN | 0.9440 | 0.9491 | 0.9440 | 0.9424 | 0.32 | No |
| ResNet-18 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 11.18 | ImageNet |
| EfficientNet-B0 | 0.9964 | 0.9965 | 0.9964 | 0.9964 | 4.01 | ImageNet |
| MobileNet V3 Small | 0.9964 | 0.9964 | 0.9964 | 0.9964 | 1.52 | ImageNet |
| ViT-B/16 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 85.80 | ImageNet |

## Key Observations
- All four **pretrained** models reached ≥99.6% accuracy — transfer learning from ImageNet is highly effective on this 4-class aerial-scene task.
- The **Custom CNN** (trained from scratch, only 0.32M params) trails at 94.4%. Its per-class report shows the weakness is `ferry_terminal` recall (0.79) — it confuses ferry terminals with harbors (both are man-made waterfront structures).
- **ResNet-18** and **ViT-B/16** both achieve a perfect score on the validation split, but ResNet-18 does so with ~8× fewer parameters (11.18M vs 85.80M).
- **MobileNet V3** matches EfficientNet-B0's accuracy (99.64%) with the fewest parameters of the pretrained models (1.52M) — best accuracy-per-parameter.

## Conclusion
For **deployment**, ResNet-18 is the recommended model: it ties for the best accuracy (100%) while being far lighter than ViT-B/16, making it faster and cheaper to serve in the Docker container. If minimising model size is the priority, MobileNet V3 (1.52M params, 99.64%) is the best lightweight alternative.

> Note: perfect / near-perfect scores suggest the validation set is visually easy to separate. The held-out `val 16` set should be used as a final unbiased test to confirm these results generalise.
