from sklearn.dummy import DummyClassifier
import numpy as np

# "Train" majority classifier. It will actually just ignore X data and pick
# majority class all the time.
def train_majority_classifier(X_train, y_train):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model
