# This script will simply predict the majority class given the model and test
# data as a method of baseline comparison. For now, prints predictions to
# screen.
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
# Using the breast cancer dataset as a toy dataset for now
from sklearn.datasets import load_breast_cancer

# Load toy data
toy_data = load_breast_cancer()
X, y = toy_data.data, toy_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# "Train" the majority class model
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

predictions = dummy.predict(X_test)
print(predictions)
