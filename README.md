# Stanford-Ribonanza-RNA-Folding(Kaggle competition)
![ ](header.png)

## Overview
Understanding the folding patterns of RNA molecules is crucial for gaining deeper insights into the fundamental workings of nature, the origins of life, and the development of innovative solutions in medicine and biotechnology. This project aims to contribute to this understanding by creating a model capable of predicting the structures of any RNA molecule and generating corresponding chemical mapping profiles.

## Goal
The primary goal of this project is to develop an algorithmic solution that accurately predicts the 3D structures of RNA molecules. By doing so, we aim to facilitate research in various fields, including molecular biology, medicine, and biotechnology. The resulting model can be utilized by biologists and biotechnologists worldwide, offering a valuable tool for advancing scientific discoveries and addressing complex challenges like climate change.
## Input data
The training data consists of RNA sequences and the 'reactivity' corresponding to each sequence and some other features such as SNR, SN filter, reactivity, and number of reads. The goal is to predict the corresponding reactivity of each sequence of RNA molecules in the test data.
## Data Preparation
In this project, the training data is sourced from two distinct types of chemical experiments. To enhance the model's learning capabilities, the data is initially divided into two groups for separate training.

### 1. Sequence Lengths and Padding
The lengths of the RNA sequences in the dataset vary, and the model needs to effectively predict reactivity for sequences of different lengths. To address this, all sequences are padded to the maximum length present in the dataset. Padding is accomplished by repeating the initial segment of each sequence at its end, ensuring uniformity in sequence lengths.

### 2. Handling Missing Data
For each sequence, the reactivity values for the first and last 26 points are marked as NaN (Not a Number) due to measurement limitations. To address this, random numbers within the range of (-0.1, 0.1) are chosen for these NaN values, taking into consideration the typical range of reactivity values.

### 3. Filtering Zero-Value SN Points
Data points with zero values for the Signal-to-Noise (SN) filter are excluded during training. This step ensures that the model is trained on reliable and informative data.

### 4. Confidence Threshold
To further refine the dataset, a threshold is applied to both the number of reads and the Signal-to-Noise Ratio (SNR). Sequences falling below this threshold are filtered out, focusing on sequences with higher confidence levels for training.
## Model
In this work, a Bidirectional-LSTM(Long Short-Term Memory) model with attention was used to predict the reactivities. 

<img src="attn_model.png" width="300">
### Bi-LSTM Layer:
This layer handles input RNA sequences with dimensions (Tx, input_size), where Tx represents the sequence length, and input_size signifies the number of features in the RNA sequence.

### Attention Mechanism:
The model incorporates an attention mechanism. This dynamic approach significantly enhances the model's ability to assign varying degrees of importance to different positions.

### Post-Attention LSTM Cell: 
After attention, an LSTM cell processes the attention-weighted context vector, capturing complex dependencies unveiled by the attention mechanism.

### Output Layer:
The final output is generated through a Dense layer, predicting reactivity values for each position in the RNA sequence.
