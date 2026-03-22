# This script will simply predict the majority class given the model and test
# data as a method of baseline comparison.
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
