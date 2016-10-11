import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_latent_variable(X, Y):
    # print '%d samples in total' % X.shape[0]
    if X.shape[1] != 2:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        print pca.explained_variance_ratio_
    plt.figure(figsize=(8, 8))
    plt.axes().set_aspect('equal')
    color = plt.cm.rainbow(np.linspace(0, 1, 10))
    for l, c in enumerate(color):
        inds = np.where(Y==l)
        # print '\t%d samples of label %d' % (len(inds[0]), l)
        plt.scatter(X[inds, 0], X[inds, 1], c=c, label=l, linewidth=0, s=8)
    # plt.xlim([-5.0, 5.0])
    # plt.ylim([-5.0, 5.0])
    plt.legend()
    plt.show()
