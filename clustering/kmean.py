
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# load data 
df = pd.read_csv('mall_customer.csv')
df.columns


######################## ####################################################################
############ data anlaysis

# sample
df.sample(5)
# info
df.info()
# check null values
df.isnull().sum()
# statistical analysis
df.describe()


######################## ####################################################################
############ feature scaling

from sklearn.preprocessing import StandardScaler

#  get 2 column
gender = df['Gender']
gender

# get 3,4,5 columns
x = df.iloc[:, 2:]
x

# standardisation
scaler = StandardScaler()
# apply scaler
x = scaler.fit_transform(x)


data = pd.DataFrame(x, columns = ['age', 'income', 'score'])
data['Gender'] = gender
data


######################## ####################################################################
############ data visualization

# visualize the normalize data
sns.pairplot(data, hue='Gender')



######################## ####################################################################
############ segmentation with age vs spending score

#  try to see the number of clusters by eyes
sns.scatterplot(data=data, x='age', y='score', hue='Gender')

# import model
from sklearn.cluster import KMeans

# get the needed the columns + convert to array
X = data[['age', 'score']].to_numpy()
X

# create model
kmean = KMeans(n_clusters=2, random_state=42)
# fit
kmean.fit(X)

# show labels
labels = kmean.labels_
# show centers axis
centers = kmean.cluster_centers_
# show inertia
inertia = kmean.inertia_




######################## ####################################################################
############ visualize Clustes & Centroids


# get cluster indexes (C0,C1)
C0 = labels == 0
C1 = labels == 1

# plot all 
plt.scatter(X[:,0], X[:,1])


plt.scatter(X[C0,0], X[C0,1], c='red', label='Cluster 0') # plot cluster0
plt.scatter(X[C1,0], X[C1,1], c='blue', label='Cluster 1') # plot cluster1
plt.scatter(centers[:,0], centers[:,1], s=70, c='black', label='Centroid')  # plot centroids
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()

######################## ####################################################################
############ Decision boundry visualization


# create meshgrid

x_min = X[:, 0].min()-1
x_max = X[:, 0].max()+1

y_min = X[:, 1].min()-1
y_max = X[:, 1].max()+1

x_min,x_max,y_min, y_max
 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), 

                     np.arange(y_min, y_max, 0.01))
xx, yy

region_points = np.concatenate((xx.ravel().reshape(-1,1),
                               yy.ravel().reshape(-1,1)), axis=1)

region_points

z = kmean.predict(region_points)
z = z.reshape(xx.shape)

# plot all 
plt.scatter(X[:,0], X[:,1])


plt.scatter(X[C0,0], X[C0,1], c='red', label='Cluster 0') # plot cluster0
plt.scatter(X[C1,0], X[C1,1], c='blue', label='Cluster 1') # plot cluster1
plt.scatter(centers[:,0], centers[:,1], s=70, c='black', label='Centroid')  # plot centroids
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()

# create contourf
plt.contourf(xx, yy, z, alpha=0.4)




######################## ####################################################################
############ more than two clusters


#  number of clusters
k = 2

# create model
kmean = KMeans(n_clusters=k, random_state=42)
# fit
kmean.fit(X)

    
# show labels
labels = kmean.labels_ 
# show centers axis
centers = kmean.cluster_centers_
# show inertia
inertia = kmean.inertia_


# create meshgrid

x_min = X[:, 0].min()-1
x_max = X[:, 0].max()+1

y_min = X[:, 1].min()-1
y_max = X[:, 1].max()+1

x_min,x_max,y_min, y_max
 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), 

                     np.arange(y_min, y_max, 0.01))
xx, yy

region_points = np.concatenate((xx.ravel().reshape(-1,1),
                               yy.ravel().reshape(-1,1)), axis=1)

region_points

z = kmean.predict(region_points)
z = z.reshape(xx.shape)

# loop for plotting clusters
for idx in range(k):
    
    clust_i = labels == idx
    print("Cluster {}: {} Customers".format(idx, sum(clust_i)))
    plt.scatter(X[clust_i, 0], X[clust_i, 1],label='Cluster {}'.format(idx)) # plot cluster0

# plotting centers
plt.scatter(centers[:,0], centers[:,1], s=70, c='black', label='Centroid')  # plot centroids
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()

# create contourf
plt.contourf(xx, yy, z, alpha=0.4)



######################## ####################################################################
############ deciding the optimal number of clusters



# import slihoute
from sklearn.metrics import silhouette_score

# create lists
inertia_score = []
sil_score = []

# create loop 
max_k = 15
for k in range(2, max_k):
    # train
    kmean = KMeans(n_clusters=k)
    kmean.fit(X)

    # inertia scores
    inertia_score.append(kmean.inertia_)
    # silhoute scores
    cluster_labels = kmean.labels_
    s_score = silhouette_score(X, cluster_labels)
    sil_score.append(s_score)


inertia_score, sil_score



# plotting inertia, silhoute 

plt.rcParams['figure.figsize'] = [9,6]
plt.rcParams['figure.dpi'] = 300

plt.subplot(2,1,1)
plt.plot(np.arange(2, max_k), inertia_score, 'o-')
plt.title('Cluster Inertia')

plt.subplot(2,1, 2)
plt.plot(np.arange(2, max_k), sil_score, 'o-')
plt.title('Cluster silhoute')

plt.tight_layout()




######################## ####################################################################
############ income + age + spending score 


k = 6
# get the needed the columns + convert to array
X = data[['age', 'income','score']].to_numpy()
X


# train
from sklearn.cluster import KMeans
# create model
kmean = KMeans(n_clusters=k, random_state=42)
# fit
kmean.fit(X)
   
# show labels
labels = kmean.labels_ 
# show centers axis
centers = kmean.cluster_centers_
# show inertia
inertia = kmean.inertia_

# clusters 
data['cluster'] = labels
data

# plotting 3D clustering

import plotly as py
import plotly.graph_objects as go


trace = go.Scatter3d(
    x= data['age'], y= data['score'], z= data['income'],
    mode='markers',
    marker=dict(
        color = data['cluster']
    )
)

trace

data2 = [trace]
data2
 
layout = go.Layout(
    title='3D Clustering',
    scene = dict(
        xaxis = dict(title='Age'), 
        yaxis = dict(title='Score'),
        zaxis = dict(title='Income')))

plt.rcParams['figure.figsize'] = [9,6]
plt.rcParams['figure.dpi'] = 300
fig = go.Figure(data=data2, layout=layout)
py.offline.iplot(fig)












