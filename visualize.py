"""
Visualize examples from the scikit-learn digits dataset before and after
binarization.
"""

from sklearn.datasets import load_digits
from sklearn.preprocessing import binarize

import matplotlib.pyplot as plt

dataset = load_digits()
for i, image in enumerate(dataset.data[:10]):
    plt.subplot(2, 10, i + 1)
    plt.axis('off')
    plt.imshow(image.reshape((8, 8)), cmap=plt.cm.gray_r)
for i, image in enumerate(binarize(dataset.data, threshold=8)[:10]):
    plt.subplot(2, 10, i + 11)
    plt.axis('off')
    plt.imshow(image.reshape((8, 8)), cmap=plt.cm.gray_r)
plt.show()
