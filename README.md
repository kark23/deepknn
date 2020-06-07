# DEEP-KNN Module

## Motivation

Distance based metrics over the input space are central to many analytical/ML based methods, such as clustering, KNN, matching, etc. Whole these methods have various respective benefits, their usage can also often be limited when constrained to euclidean distance on the input space. Limitations include:
* Lack of accounting for input relationship with dependent variables/objective
* Limitations with respect to variable interaction
* Euclidean distance struggles with representation of more complex latent space patterns

<div align="center">
  
![alt text](https://github.com/kark23/deepknn/blob/master/figs/fig1.PNG?raw=true)

Figure 1: Intuitive Cluster Differentiation v. Result of K-Means

</div>

Depending on application, there are a variety of ways to mitigate these limitations, including:
* Dimensional reduction/ latent space transformation
* Alternative distance metrics
* Usage of other methods

Commonly used dimensional reduction techniques include:
* Principle Component Analysis (PCA)
* Linear Discriminant Analysis (LDA)
* Partial Least Squares (PLS)
* Autoencoders

Autoencoders have demonstrated benefit for more complex input spaces representations such as raw image data, whcih suggests benefit when used in conjunction with distance based methods mentioned earlier (clustering, KNN, matching, etc.). The purpose of this project is to modularize an AE-KNN hybrid implementation which can be used across a diverse span of applications. Although this specific use case is the primary focus of the project, influence was drawn from various related papers/projects worth looking into for further information about the subject:
* [Deep Clustering for Unsupervised Learning of Visual Features- Facebook AI Rsch (2019)](https://arxiv.org/pdf/1807.05520.pdf) [(Code)](https://github.com/facebookresearch/deepcluster)
* [AEkNN: An AutoEncoder kNN-based classifier withbuilt-in dimensionality reduction (2018)](https://arxiv.org/pdf/1802.08465.pdf)
* [Learning a Neighborhood Preserving Embedding by Convolutional Autoencoder](https://github.com/zhan1182/autoencoder-kNN)

<div align="center">
  
![alt text](https://github.com/kark23/deepknn/blob/master/figs/fig3.PNG?raw=true)

Figure 2: Classification Performance Benefits of Autoencoder v. Other Dimensional Reduction Techniques for Various Datasets, from [AEkNN: An AutoEncoder kNN-based classifier withbuilt-in dimensionality reduction (2018)](https://arxiv.org/pdf/1802.08465.pdf) (note only LDA ever outperformed AE but their AE had no dependent variable objective like LDA, whereas this module has that capability)

</div>

## Implementation
For an instance of the deepknn model class, there a decent degree of customizability for the user. Flexibilites for the user include:
* Autoencoder shape/depth control
* Decoded layer shape
* Convolutional neural net associated optionality
* Weight/bias adjustments
* Regressor/classifier constraint for the additional objective function

<div align="center">
  
![alt text](https://github.com/kark23/deepknn/blob/master/figs/fig2.PNG?raw=true)

Figure 3: Autoencoder Structure with Laten Space for KNN, from [AEkNN: An AutoEncoder kNN-based classifier withbuilt-in dimensionality reduction (2018)](https://arxiv.org/pdf/1802.08465.pdf) 

</div>

The autoencoder training structure allows for partial constraint to an additional objective function to some regressor/classifier dependent variable. During the training phase, experimentation has shown it beneficial to break the training into pre-training/training phases where the pre-training weights the additional obective at 0 in order to get a reasonable base NN before applying the additional constraint. Optionality for the training phases includes adjustment relative to:
* Learning rate
* Batch size
* Epoch size
* Ratio of weighting for objective function vs. AE reconstruction error (note user must account for varying magnitudes for different functions when choosing this ratio)

The predict method for the deepknn class uses KNN over the AE produced latent space to make a classification/regression prediction. Work is also in progress to allow for particle filter pdf approximation as an alternative to KNN.

## Example Usage
```
from main import *
from tensorflow.examples.tutorials.mnist import input_data

mnist= input_data.read_sata_sets("MNIST_data/", one_hot=False)
x, y= mnist.train.next_batch(mnist.test.num_examples)
tst_x, tst_y= mnist.test.next_batch(mnist.test.num_examples)
aek=aeknn('class', x, y, dtyp='numpy', cnn=True)
aek.train(x, y, save='tmp/mod.ckpt', ratio=0., batch=100, epoch=100)
aek.train(x, y, save='tmp/mod.ckpt', ratio=.99, batch=4000, epoch=100, load='tmp/mod.ckpt')
preds=aek.predict(x, y, tst_x, 5, 'tmp/mod.ckpt')
np.count
```

