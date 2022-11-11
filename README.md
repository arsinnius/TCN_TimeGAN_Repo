# TCN_TimeGAN_Repo
Adapted from the excellent paper by Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar:  
[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks),  
Neural Information Processing Systems (NeurIPS), 2019.

- Last updated Date: April 24th 2020
- [Original code](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/timegan/) author: Jinsung Yoon (jsyoon0823@gmail.com)
## Description
In the original timegan paper, the autoencoder was built using an RNN. This project replaces the RNN with the TCN-AE autoencoder by Thill et al. in their paper "Temporal convolutional autoencoder for unsupervised anomaly detection in time series."

The modified model was tested on two different time series - financial and climate. Using the default settings, TCN-AE outperformed the RNN on the financial time series. However it didn't perform very well on the climate data. The belief is that the performance on both series could be vastly improved if the parameters of the TCN-AE were individually optimized for each time series type. 
## Getting Started
Modify the original timegan using tcn.