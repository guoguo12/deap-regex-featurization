"""
Finds a regex featurization for binarized rows of images of handwritten digits
and evaluates the featurization by training an SVM (for various values of C).

The regex featurization is compared against the identity featurization, i.e.,
using the raw pixels.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
import sklearn.svm

from models import EvolutionaryRegexFeaturizer

TEST_SIZE = 0.33

FEATURE_SIZE = 8
REGEX_SIZE = 3
NUM_GENERATIONS = 10
POP_SIZE = 512


def get_binarized_data(stringify=True):
    """Loads and returns the binarized images dataset."""
    def to_str(row):
        return ''.join(map(str, np.ravel(row.astype(int))))

    dataset = load_digits()
    data = binarize(dataset.data, threshold=8)
    if stringify:
        data = np.apply_along_axis(to_str, 2, data.reshape((-1, 8, 8)))

    return train_test_split(
        data, dataset.target, test_size=TEST_SIZE, random_state=0)


def svm_acc(X_train, y_train, X_test, y_test, C, kernel='rbf'):
    """Trains and evaluates an SVM with the given hyperparameters and data."""
    clf = sklearn.svm.SVC(C=C, kernel=kernel)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


X_train, X_test, y_train, y_test = get_binarized_data()

erfs = []  # Train one model per row
for i in range(8):
    regex_comps = ['0', '1', '(', ')?', ')*']
    init_set = range(4)  # Initialize with first four kinds of regex characters

    erf = EvolutionaryRegexFeaturizer(regex_comps, init_set, X_train[:,i], y_train, 10)
    erf.train(FEATURE_SIZE, REGEX_SIZE, NUM_GENERATIONS, POP_SIZE)
    erfs.append(erf)

X_train_f = np.hstack(erfs[i].featurize(X_train[:,i]) for i in range(8))
X_test_f = np.hstack(erfs[i].featurize(X_test[:,i]) for i in range(8))
X_train_raw, X_test_raw, _, _ = get_binarized_data(stringify=False)

for C in np.logspace(-2, 5, 8):
    raw_acc = svm_acc(X_train_raw, y_train, X_test_raw, y_test, C)
    featurized_acc = svm_acc(X_train_f, y_train, X_test_f, y_test, C)
    print('C={:.2f}, raw accuracy={:.3f}, featurized accuracy={:.3f}'.format(
        C, raw_acc, featurized_acc))
