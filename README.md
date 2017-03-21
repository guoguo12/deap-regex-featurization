# Image Classification with Regex-based Features

I wrote a blog post on this project: ["Image Classification with Regex-based Features"](http://aguo.us/writings/regex-features.html). Here's a summary:

* I used regular expressions to featurize images of handwritten digits.
To find good regexes, I used an evolutionary algorithm implemented using [DEAP](https://deap.readthedocs.io/).
* To evaluate my features, I trained an SVM on the featurized dataset.
I was ultimately able to reach ~64% classification accuracy using regex-based features, which is significantly better than guessing.
That said, it's not great&mdash;training an SVM on the raw image pixels gives ~95% accuracy.
