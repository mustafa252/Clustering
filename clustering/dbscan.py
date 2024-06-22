
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# generate moon data
from sklearn.datasets import make_moons

X, Y = make_moons(n_samples=200, noise=0.1 , random_state=0)
X

plt.rcParams['figure.figsize'] = [9,6]
plt.rcParams['figure.dpi'] = 300
plt.scatter(X[:, 0], X[:, 1])
plt.show()



# train
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
clusters = dbscan.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.title('DBSCAN Clustering')
















