#we use linear regression and the column 'pedal width' to determine if the flower is of tyoe 'setosa' or not

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random


iris = load_iris()

X = iris.data[:, 3].reshape(-1, 1)
y = (iris.target == 0).astype(int)  # 1 if setosa, 0 otherwise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= int(random.random()))
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = np.mean(y_pred_class == y_test)

#accuracy should be 1.0 since correlation between input and output is very high
print("Accuracy:", accuracy)