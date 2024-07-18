# Lane Detection Using U-Net

This project aims to detect lanes in images using a U-Net model. The dataset used for this project is the TuSimple Preprocessed dataset. The project involves data preprocessing, model training, and evaluation.


<img align = 'center' src="https://i.ytimg.com/vi/KzRkS-8oNtc/maxresdefault.jpg" alt="Lane Detection Example" width="500"/>

## Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- Matplotlib
- Kaggle API

## Dataset
The dataset used in this project is the TuSimple Preprocessed dataset. You need a Kaggle account and API key to download the dataset.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/lane-detection-unet.git
    cd lane-detection-unet
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up Kaggle API:
    - Go to your Kaggle account and create a new API token.
    - Place the `kaggle.json` file in the `~/.kaggle/` directory.

4. Download and unzip the dataset:
    ```bash
    kaggle datasets download -d hikmatullahmohammadi/tusimple-preprocessed
    unzip tusimple-preprocessed.zip
    ```

## Usage
1. Run the data preprocessing script to prepare the training and testing datasets:
    ```python
    python preprocess.py
    ```

2. Train the U-Net model:
    ```python
    python train.py
    ```

3. Evaluate the trained model:
    ```python
    python evaluate.py
    ```

## Model Architecture
The model used in this project is a U-Net, which consists of a contraction path and an expansion path. The contraction path captures context through a series of convolutional and max-pooling layers, while the expansion path enables precise localization through a series of transposed convolutions and concatenations with corresponding feature maps from the contraction path.

## Training
The training process involves:
1. Initializing the model with the specified architecture.
2. Compiling the model with Adam optimizer and binary cross-entropy loss.
3. Training the model on the training dataset with a validation split.

Training parameters:
- Batch size: 16
- Epochs: 25
- Early stopping with patience: 5

## Results
The trained model achieves the following results on the validation set:
- Loss: 0.0541
- Accuracy: 0.9597

Sample predictions from the model on the test set are visualized in the `evaluate.py` script.

## License
This project is licensed under the MIT License.

