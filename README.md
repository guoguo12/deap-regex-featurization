# Image Classification with Regex-based Features

I used regular expressions to featurize images of handwritten digits.
To find good regexes, I used an evolutionary algorithm implemented using [DEAP](https://deap.readthedocs.io/).

To evaluate my features, I trained an SVM on the featurized dataset.
I was ultimately able to reach ~64% classification accuracy using regex-based features, which is better than guessing.
That said, it's not great&mdash;training an SVM on the raw image pixels gives ~95% accuracy.

Not bad for a weekend project though!
