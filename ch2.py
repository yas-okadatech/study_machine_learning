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

# Petal length
plength=features[:,2]
is_setosa = (labels == 'setosa')

max_setosa=plength[is_setosa].max()
min_non_setosa=plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))

def apply_model( example ):
  if example[2] < 2: print('Iris Setosa')
  else: print('Iris Virginica or Iris Versicolor')

features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')

print(virginica)

for t,marker,c in zip(range(3),'>ox','rgb'):
  plt.scatter(features[target == t,0],
              features[target == t,1],
              marker=marker,
              c=c)

plt.show()
