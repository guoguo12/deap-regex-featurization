# Image Classification with Regex-based Features

I wrote a [blog post](http://aguo.us/writings/regex-features.html) on this project. 

## Experiment 1
*Source: `train_digits.py`.*

I used regular expressions to featurize images of handwritten digits.
To find good regexes, I used an evolutionary algorithm implemented using [DEAP](https://deap.readthedocs.io/).

To evaluate my features, I trained an SVM on the featurized dataset.
I was ultimately able to reach ~64% classification accuracy using regex-based features, which is significantly better than guessing.
That said, it's not great&mdash;training an SVM on the raw image pixels gives ~95% accuracy.

## Experiment 2
*Source: `train_digits_rows.py`.*

I divided each image into rows and repeated Experiment 1, training one model per row. I then featurized each image by featurizing its rows and concatenating the row features. This gave me ~88.7% classification accuracy.

What irks me is that I achieved optimal performance with a regex size of only 3. Most of the top-fitness regexes were simple strings like `101`, `111`, `000`, etc.
This actually makes a lot of sense, but it kind of defeats the purpose of using regex. Or maybe it shows that using regex is simply unnecessary for this particular task.

## Instructions

First, install the required libraries: `pip install -r requirements.txt`.

`models.py` contains the main model class, `EvolutionaryRegexFeaturizer`.
See `train_digits.py` and `train_digits_rows.py` for example usage.
