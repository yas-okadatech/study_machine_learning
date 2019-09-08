from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
# print(data)

features = data['data']
#print(features)
feature_names = data['feature_names']
#print(feature_names)
target=data['target']
#print(target)
target_names=data['target_names']
#print(target_names)
labels=target_names[target]
#print(labels)

for t,marker,c in zip(range(3),'>ox','rgb'):
  plt.scatter(features[target == t,0],
              features[target == t,1],
              marker=marker,
              c=c)
  print('marker: %s, target_name: %s' % (marker, target_names[t]))

plt.show()
