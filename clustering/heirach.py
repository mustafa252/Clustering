

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


 
# load data
data = pd.read_csv('sp500_data.csv', index_col=0)
data

# define needed columns
stocks = ['AAPL', 'AMZN', 'AXP', 'COP', 'CSCO', 'CVX',
          'GOOGL', 'HD', 'INTC', 'JPM', 'MSFT', 'SLB', 'TGT',
          'USB', 'WFC', 'WMT', 'XOM']

# define dataFrame
df = data[stocks].T
df


# plotting hierarchical diagrame
Linkage = linkage(df, 'complete')
plt.figure(figsize=(12,5), dpi=200)
dendrogram(Linkage, labels=df.index, leaf_rotation=90, leaf_font_size=16)
plt.title('Hierarichal Clustering')
plt.xlabel('Sample index')
plt.ylabel('distance')
plt.show()











