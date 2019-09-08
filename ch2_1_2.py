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

def apply_model1( example ):
  if example[2] < 2: print('Iris Setosa')
  else: print('Iris Virginica or Iris Versicolor')

features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')

# print(labels)
# print(virginica)

best_acc = -1.0
best_fi = -1.0
best_t = -1.0

for fi in range(features.shape[1]):
  thresh = features[:, fi].copy()
  thresh.sort()

  for t in thresh:
    pred = (features[:,fi] > t)
    acc = (labels[pred] == 'virginica').mean()
    # print('acc: %f, fi: %f, t: %f' % (acc, fi, t))
    if acc > best_acc:
      best_acc = acc
      best_fi = fi
      best_t = t

print('best_acc: %f' % best_acc)
print('best_fi: %f' % best_fi)
print('best_t: %f' % best_t)

def apply_model2( example ):
  if example[best_fi] > best_t: print('virginica')
  else: print('versicolor')


