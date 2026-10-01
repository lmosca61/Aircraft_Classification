# Fine-Grained Aircraft Classification

Classifying 100 aircraft variants using a custom convolutional network trained from scratch and an ImageNet-pretrained ResNet-18.

This project explores network design, regularization, and transfer learning on **FGVC-Aircraft**, with validation-based model selection and ablation studies.

## Results

| Model | Best validation Top-1 | Test Top-1 | Test Top-5 |
|---|---:|---:|---:|
| Custom residual CNN | 58.15% | 60.04% | 88.06% |
| ResNet-18 — Part 2A | 68.83% | 68.65% | 91.09% |
| ResNet-18 — Part 2B | **73.93%** | **74.50%** | **92.65%** |

Results use single-view evaluation. Part 2B improves test Top-1 accuracy by **5.85 percentage points** over Part 2A and **14.46 percentage points** over the custom network.

## Dataset

[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) is a fine-grained image classification dataset. The task uses 100 aircraft variants, many of which differ only through subtle visual details such as wings, engines, and fuselage proportions.

The notebook uses the official splits:

- **Training:** parameter updates and training augmentation.
- **Validation:** checkpoint selection and experimental comparisons.
- **Test:** reporting classification performance.

Aircraft bounding-box annotations are used to extract the relevant image region before preprocessing. The dataset is downloaded through Torchvision and is not included in this repository.

## Part 1 — Custom Network

The custom CNN is built directly from PyTorch layers and trained without pretrained weights.

Its architecture includes:

- An initial convolution and max-pooling layer.
- Eight residual blocks arranged in four stages.
- Batch Normalization and ReLU activations.
- Stochastic residual-branch dropping during training.
- Global average pooling and a 100-class linear classifier.

Training uses SGD with momentum, five epochs of learning-rate warm-up, cosine decay, Mixup, label smoothing, and weight decay.

### Ablation studies

| Configuration | Best validation Top-1 | Change from baseline |
|---|---:|---:|
| Full custom model | **58.15%** | — |
| Without Batch Normalization | 38.28% | −19.87 pp |
| Without residual connections* | 57.67% | −0.48 pp |
| Without Mixup | 55.90% | −2.25 pp |

*Removing residual connections also disables stochastic depth.*

Batch Normalization provides the clearest benefit in these experiments. Mixup provides a smaller improvement, while the residual-connection comparison shows only a marginal difference.

An additional learning-rate experiment reached **48.21%** validation accuracy with ReduceLROnPlateau. This comparison is exploratory: its two-stage implementation restores the best warm-up weights before continuing, unlike the uninterrupted cosine baseline.

## Part 2 — Transfer Learning

Both experiments use Torchvision’s ResNet-18 with **ImageNet-1K V1** pretrained weights.

### Part 2A: reuse the custom model’s training settings

The original ImageNet classifier is replaced with a 100-class linear layer. The network is fine-tuned using the same optimizer, learning-rate schedule, epoch budget, augmentation, Mixup, and label smoothing used for the custom model.

### Part 2B: adapt the training recipe

The second configuration introduces:

- **384 × 384 inputs** to retain finer visual details.
- **Letterboxing** to preserve the complete expanded bounding-box region.
- **ImageNet normalization** and mild colour augmentation.
- A classification head with **dropout of 0.30**.
- **Three epochs of classifier-only training**, with the backbone and its Batch Normalization statistics frozen.
- **Forty epochs of full-network fine-tuning**.
- Different initial learning rates for the backbone and classifier: **5e-5** and **5e-4**.
- AdamW, cosine decay, and Mixup, without additional label smoothing.

The best checkpoint is selected using validation accuracy.

## Evaluation and Analysis

The notebook includes:

- Training and validation learning curves.
- Ablation comparisons.
- Top-1 and Top-5 test accuracy.
- A row-normalized confusion matrix.
- The most frequent class-confusion pairs.
- Example predictions with confidence scores.

Training accuracy with Mixup is weighted according to the two mixed labels. It should not be interpreted as directly equivalent to standard validation accuracy.

Parts 2A and 2B use different label-smoothing settings, so their training and validation losses are not directly comparable in absolute magnitude. The final test table uses a common loss criterion.

## Running the Notebook

Open [`aircraft-classification.ipynb`](aircraft-classification.ipynb) in Kaggle or another Jupyter environment with a CUDA-enabled GPU.

Internet access is needed for the initial dataset and pretrained-weight downloads.

### Staged execution

The notebook divides the experiments into three stages to accommodate long training runs.

| Setting | Work performed | Checkpoint output |
|---|---|---|
| `RUN_STAGE = 1` | Custom model and BatchNorm ablation | `stage1_main_bn.pt` |
| `RUN_STAGE = 2` | Restore Stage 1 and train the remaining Part 1 experiments | `stage2_complete_part1.pt` |
| `RUN_STAGE = 3` | Restore Part 1 results, reproduce its plots, train Parts 2A and 2B, and evaluate the final models | — |

For a complete run:

1. Set `RUN_STAGE = 1` and execute the notebook.
2. Make the Stage 1 checkpoint available to the next session.
3. Set `RUN_STAGE = 2` and execute the notebook.
4. Make the Stage 2 checkpoint available to the next session.
5. Set `RUN_STAGE = 3` and execute the notebook.

On Kaggle, attach the preceding stage’s notebook output as an input. The code searches for the checkpoint automatically; explicit paths can be set when multiple matching files exist.

**Stage 3 requires the Stage 2 checkpoint and still trains both ResNet-18 configurations.** These checkpoints support hand-off between completed stages, not recovery from an interrupted training epoch.


## References

- He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385), CVPR, 2016.
- Maji et al., [*Fine-Grained Visual Classification of Aircraft*](https://arxiv.org/abs/1306.5151), arXiv, 2013.
- Zhang et al., [*mixup: Beyond Empirical Risk Minimization*](https://arxiv.org/abs/1710.09412), ICLR, 2018.
- Loshchilov and Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101), ICLR, 2019.
- Loshchilov and Hutter, [*SGDR: Stochastic Gradient Descent with Warm Restarts*](https://arxiv.org/abs/1608.03983), ICLR, 2017.
- Zhang, Lipton, Li, and Smola, [*Dive into Deep Learning — Fine-Tuning*](https://d2l.ai/chapter_computer-vision/fine-tuning.html), Cambridge University Press, 2023.
- PyTorch, [*Transfer Learning for Computer Vision Tutorial*](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
