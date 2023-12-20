# Stanford-Ribonanza-RNA-Folding(Kaggle competition)
# Overview
Understanding the folding patterns of RNA molecules is crucial for gaining deeper insights into the fundamental workings of nature, the origins of life, and the development of innovative solutions in medicine and biotechnology. This project aims to contribute to this understanding by creating a model capable of predicting the structures of any RNA molecule and generating corresponding chemical mapping profiles.

# Goal
The primary goal of this project is to develop an algorithmic solution that accurately predicts the 3D structures of RNA molecules. By doing so, we aim to facilitate research in various fields, including molecular biology, medicine, and biotechnology. The resulting model can be utilized by biologists and biotechnologists worldwide, offering a valuable tool for advancing scientific discoveries and addressing complex challenges like climate change.
# Input data
The training data consists of RNA sequences and the 'reactivity' corresponding to each sequence and some other features such as SNR, SN filter, reactivity, and number of reads. The goal is to predict the corresponding reactivity of each sequence of RNA molecules in the test data.
# Model
in this work, a Bidirectional-LSTM model with attention was used to predict the reactivities. 
