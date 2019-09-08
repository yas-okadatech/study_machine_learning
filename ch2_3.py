import numpy as np


def distance(p0, p1):
  'ユークリッド距離'
  return np.sum( (p0-p1) ** 2 )

# 距離が一番近いものを探す
def nn_classify(training_set, training_labels, new_example):
  dists = np.array([distance(t, new_example) for t in training_set])
  nearest = dists.argmin() # 最小要素のindex
  return training_labels[nearest]

