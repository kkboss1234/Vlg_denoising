
# Image Denoising 😃🌟 

With the goal of recovering high-quality image content from its degraded version, image restoration enjoys numerous applications, such as in photography, security, medical imaging, and remote sensing. In this project I have implemented an model named MirNet for low-light image enhancement. It is a fully-convolutional architecture that learns an enriched set of features that combines contextual information from multiple scales, while simultaneously preserving the high-resolution spatial details

## Table of Contents 💡
- **Introduction**
- **Pipeline**
- **How to install?**
- **More Knowledge**
## Introduction 🍁
This repository implements the MIRNet model for enhancing low-light images. MIRNet, which stands for Multi-Scale Residual Attention Network, is a powerful neural network designed to improve the visibility and quality of images captured in poor lighting conditions. This implementation includes training on the LOL dataset and testing on a custom test set to demonstrate the model's effectiveness.

## Pipeline ❄️
1. Data Preparation
- **Training Data**: Images are sourced from the LOL dataset, specifically the `our485` directory.
- **Testing Data**: Images are sourced from the `data/test` directory.
- **Preprocessing**: Images are read and normalized to a [0, 1] range, followed by random cropping to a fixed size of 128x128 pixels.

### 2. Model Architecture

#### 2.1 Input Layer
- The model accepts images of variable sizes with 3 channels (RGB).

#### 2.2 Recursive Residual Groups (RRG)
- Each RRG contains multiple Multi-Scale Residual Blocks (MRB).
- Each MRB integrates features from three different scales using down-sampling and up-sampling modules.

#### 2.3 Dual Attention Unit (DAU)
- Each MRB contains DAUs that incorporate both spatial and channel attention mechanisms to emphasize important features.

#### 2.4 Selective Kernel Feature Fusion (SKFF)
- SKFF modules are used to combine multi-scale features effectively by learning to weight features from different scales dynamically.

#### 2.5 Multi-Scale Residual Block (MRB)
- MRBs apply DAUs and SKFFs to learn robust features at multiple scales and improve feature representation.

### 3. Training

#### 3.1 Loss Function
- **Charbonnier Loss**: A smooth approximation of L1 loss, used for its robustness to outliers.

#### 3.2 Metrics
- **Peak Signal-to-Noise Ratio (PSNR)**: Used to measure the quality of the reconstructed image.

#### 3.3 Optimizer
- **Adam Optimizer**: Used for training with a learning rate of 1e-4.

#### 3.4 Training Process
- The model is trained for 50 epochs with a learning rate reduction strategy based on validation PSNR.

### 4. Inference
- The trained model is used to enhance low-light images.
- The inference pipeline includes reading the image, normalizing it, passing it through the model, and post-processing the output to generate the final enhanced image.

### 5. Evaluation
- Enhanced images are compared with original low-light images and images processed using autocontrast for qualitative assessment.

## Results
- The effectiveness of the MIRNet model is demonstrated through qualitative comparisons, showcasing significant improvements in image visibility and quality.

## How to Install my Repository and work with it on your system ?
## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/kkboss1234/Vlg_denoising.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ./Vlg_denoising/
    next line:
    cd ./Project/
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt

    ```
## Training

To train the model, run:
```bash
python main.py
```

## How to Install my Repository and work with it on your system ?
## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/kkboss1234/Vlg_denoising.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ./Vlg_denoising/
    next line:
    cd ./Project/
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt

    ```
## Training

To train the model, run:
```bash
python main.py
```
## More Knowledge
For More Knowledge:
U can view this report created by me :
https://drive.google.com/file/d/1njtcuIq4T7L-6FEPvJlMt08ys4sMvt40/view?usp=sharing

Or you can view the research paper at:
https://www.ijfmr.com/papers/2023/3/4001.pdf
