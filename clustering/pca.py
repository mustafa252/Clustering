

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load MNIST_784
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
mnist

# keys
mnist.keys()

# data
x = mnist.data
x
# targets
y = mnist.target
y
# frame
mnist.frame



##############################################################################
####### training + compression 

# import model
from sklearn.decomposition import PCA

# dimensions before compression
x.shape
# create model
pca = PCA(n_components=2)

# fit
pca.fit(x)
# transform
x_pca = pca.transform(x)
 
# dimesions after compression
x_pca.shape


# plotting pca
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 200
y = y.astype('int')
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.colorbar()
plt.show()




# show how much data in each pca(pca1, pca2)
pca.explained_variance_ratio_ 


# data reconstruction
x_recover = pca.inverse_transform(x_pca)
x_recover.shape




##############################################################################
####### choose the right number of principle

# take all the number of components in training
pca = PCA(n_components=300)
pca.fit(x)

# show the explained variance ratio
pca.explained_variance_ratio_

# cumulative sum
cumsum = np.cumsum(pca.explained_variance_ratio_)

# plot the variance
plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 200
plt.plot(cumsum)
plt.plot([0,300], [0.95, 0.95], 'r--')
plt.xlabel('No. of Components')
plt.ylabel('Total Variance')
plt.title('Variance vs No. of Components')
plt.show()


# num of component tha less or equal than (0.95)
idx = sum(cumsum<=0.95)
idx
 
# training
pca = PCA(n_components=153)
pca.fit(x)
# show the used ration from the dataset
sum(pca.explained_variance_ratio_)


##############################################################################
####### recovering with 95% info.

x_pca = pca.fit_transform(x)
x_pca.shape

x_recover = pca.inverse_transform(x_pca)
x_recover.shape

