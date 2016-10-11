## Adversarial AutoEncoder
----------------------------

[Adversarial Autoencoder [arXiv:1511.05644]](http://arxiv.org/abs/1511.05644) implemented with [MXNet](https://github.com/dmlc/mxnet).

### Requirements
* MXNet
* numpy
* matplotlib
* scikit-learn
* OpenCV

### Unsupervised Adversarial Autoencoder
Please run aae\_unsupervised.py for model training. Set task to `unsupervised` in visualize.ipynb to display the results. Notice the desired prior distribution of the 2-d latent variable can be one of {gaussian, gaussian mixture, swiss roll or uniform}. In this case, no label info is being used during the training process.

### Supervised Adversarial Autoencoder
Please run aae\_supervised.py for model training. Set task to `supervised` in visualize.ipynb to display the results. Notice the desired prior distribution of the 2-d latent variable can be one of {gaussian mixture, swiss roll or uniform}. In this case, label info of both real and fake data is being used during the training process.

### Semi-Supervised Adversarial Autoencoder
Not implemented yet.
