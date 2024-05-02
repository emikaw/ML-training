# ================================== BIBLIOTEKI ==================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, KFold
from sklearn.linear_model import LinearRegression
from sklearn import datasets, svm, metrics, linear_model
import warnings
import matplotlib.pyplot as plt

# ================================== PROBLEMY REGRESYJNE  ==================================
# ZAŁADOWANIE PRZYKŁADOWYCH DANYCH dla regresji z 'sklearn datasets'
X, y = datasets.load_iris(return_X_y=True)
print(X.shape, y.shape)

# STWORZENIE MODELU
modelRegress = LinearRegression()

# PODZIAŁ DANYCH
# Ustalona z góry proporcja -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)
modelRegress.fit(X_train, y_train)
train_score = modelRegress.score(X_train, y_train)
test_score = modelRegress.score(X_test, y_test)
print(f"Train_score = {train_score}\nTest score = {test_score}")

# Walidacja krzyżowa --------------------------------------------------------------
kfold = KFold(n_splits=6, random_state=42, shuffle=True)

train_scores = []
test_scores = []


for (train_index, test_index) in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    modelRegress.fit(X_train, y_train)

    # Ocena zbioru treningowego
    train_score2 = modelRegress.score(X_train, y_train)
    train_scores.append(train_score2)

    # Ocena na zbiorze testowym
    test_score2 = modelRegress.score(X_test, y_test)
    test_scores.append(test_score2)

# Wszystkie wyniki CV
print(f"\nTrain scores = {train_scores}\nTest scores = {test_scores}")

# Średni wynik CV
avg_train_score = np.mean(train_scores)
avg_test_score = np.mean(test_scores)

print(f"\nAverage train score = {avg_train_score}\nAverage test score = {avg_test_score}")

# EWALUACJA MODELU (MSE, RMSE)

y_pred = modelRegress.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMSE = {mse}\nRMSE = {rmse}")

# ================================== PROBLEMY KLASYFIKACYJNE  ==================================
# ZAŁADOWANIE PRZYKŁADOWYCH DANYCH dla klasyfikacji z 'sklearn datasets'
X, y = datasets.load_wine(return_X_y=True)
print(X.shape, y.shape)

# STWORZENIE MODELU
modelClass = svm.SVC(kernel='linear', C=1)

# PODZIAŁ DANYCH
# Z góry ustalony podział
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)
modelClass.fit(X_train, y_train)
train_score = modelClass.score(X_train, y_train)
test_score = modelClass.score(X_test, y_test)
print(f"Train_score = {train_score}\nTest score = {test_score}")

# Walidacja krzyżowa
kfold = KFold(n_splits=6, random_state=42, shuffle=True)

train_scores = []
test_scores = []

for (train_index, test_index) in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    modelRegress.fit(X_train, y_train)

    # Ocena zbioru treningowego
    train_score2 = modelRegress.score(X_train, y_train)
    train_scores.append(train_score2)

    # Ocena na zbiorze testowym
    test_score2 = modelRegress.score(X_test, y_test)
    test_scores.append(test_score2)

# Wszystkie wyniki CV
print(f"\nTrain scores = {train_scores}\nTest scores = {test_scores}")

# Średni wynik CV
avg_train_score = np.mean(train_scores)
avg_test_score = np.mean(test_scores)

print(f"\nAverage train score = {avg_train_score}\nAverage test score = {avg_test_score}")

# EWALUACJA MODELU
y_pred = modelClass.predict(X_test)

# Obliczenie TP, TN, FP i FN
tp = 0
tn = 0
fp = 0
fn = 0

for y_value in range(len(y_pred)):
    if y_pred[y_value] == 1 and y_test[y_value] == 1:
        tp += 1
    elif y_pred[y_value] == 1 and y_test[y_value] == 0:
        fp += 1
    elif y_pred[y_value] == 0 and y_test[y_value] == 1:
        fn += 1
    elif y_pred[y_value] == 0 and y_test[y_value] == 0:
        tn += 1

# Metryki
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
fscore = (2 * precision * recall) / (precision + recall)

print(f"\nAccuracy = {accuracy}\nPrecision = {precision}\nRecall = {recall}\nF-score = {fscore}")

