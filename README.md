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

Some results:

p(z) and q(z) with z_prior set to gaussian distribution.

![p(z) gaussian](http://closure11.com/images/post/2016/10/gaussian_unsupervised_pz.png)
![q(z) gaussian](http://closure11.com/images/post/2016/10/gaussian_unsupervised_qz.png)

p(z) and q(z) with z_prior set to 10 gaussian mixture distribution.

![p(z) gaussian](http://closure11.com/images/post/2016/10/gaussian_mixture_unsupervised_pz.png)
![q(z) gaussian](http://closure11.com/images/post/2016/10/gaussian_mixture_unsupervised_qz.png)

p(z) and q(z) with z_prior set to swiss roll distribution.

![p(z) gaussian](http://closure11.com/images/post/2016/10/swiss_roll_unsupervised_pz.png)
![q(z) gaussian](http://closure11.com/images/post/2016/10/swiss_roll_unsupervised_qz.png)

### Supervised Adversarial Autoencoder
Please run aae\_supervised.py for model training. Set task to `supervised` in visualize.ipynb to display the results. Notice the desired prior distribution of the 2-d latent variable can be one of {gaussian mixture, swiss roll or uniform}. In this case, label info of both real and fake data is being used during the training process.

Some results:

p(z), q(z) and output images from fake data with z_prior set to 10 gaussian mixture distribution.

![p(z) gaussian](http://closure11.com/images/post/2016/10/gaussian_mixture_supervised_pz.png)
![q(z) gaussian](http://closure11.com/images/post/2016/10/gaussian_mixture_supervised_qz.png)
![output images from gaussian fake data](http://closure11.com/images/post/2016/10/gaussian_mixture_supervised_output.png)

p(z) and q(z) with z_prior set to swiss roll distribution.

![p(z) gaussian](http://closure11.com/images/post/2016/10/swiss_roll_supervised_pz.png)
![q(z) gaussian](http://closure11.com/images/post/2016/10/swiss_roll_supervised_qz.png)

p(z) and q(z) with z_prior set to 10 uniform distribution.

![p(z) gaussian](http://closure11.com/images/post/2016/10/uniform_supervised_pz.png)
![q(z) gaussian](http://closure11.com/images/post/2016/10/uniform_supervised_qz.png)


### Semi-Supervised Adversarial Autoencoder
Not implemented yet.
