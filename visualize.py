"""
One-off scripts for making visualizations.
"""

import re

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import binarize

import matplotlib.pyplot as plt


def visualize_dataset():
    """
    Visualize examples from the scikit-learn digits dataset before and after
    binarization.
    """
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


def visualize_regexes(regexes):
    """
    Visualize regex match locations.
    """
    def to_str(arr):
        return ''.join(map(str, np.ravel(arr.astype(int))))

    dataset = load_digits()
    for i, regex in enumerate(regexes):
        for j, image in enumerate(binarize(dataset.data, threshold=8)[:10]):
            mask = np.zeros(64)
            for match in re.finditer(regex, to_str(image)):
                mask[match.start():match.end()] += 1

            ax = plt.subplot(len(regexes), 10, i * 10 + j + 1)
            if j == 0:
                ax.set_title(regex, size='small', family='monospace')
            plt.axis('off')
            plt.imshow(image.reshape((8, 8)), cmap=plt.cm.gray_r)
            plt.imshow(mask.reshape((8, 8)), cmap=plt.cm.Greens, alpha=0.7)
    plt.show()

if __name__ == '__main__':
    regexes = ['()?110011(00)?', '()?110011(()?)?', '000000100001',
               '000000000000', '(10)*00100001', '(11)*00100001',
               '(1)*000000000', '(1)?000000000', '()?0000000000',
               '1(101)?100()?1']
    visualize_regexes(regexes)
