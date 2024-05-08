# ================================== LIBRARIES ==================================

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ================================== REGRESSION  ==================================
print("-"*30,"REGRESSION","-"*30)
iris = datasets.load_iris()
X = iris.data[:, 2].reshape(-1, 1)
y = iris.data[:, 3]                

# MODEL
print("[INFO] Creation a model...")
modelRegress = LinearRegression()

# DATA SPLIT
# Fixed proportion (here: 80%-20%)
print("[INFO] Data splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)
modelRegress.fit(X_train, y_train)
train_score = modelRegress.score(X_train, y_train)
test_score = modelRegress.score(X_test, y_test)
print(f"[INFO] Calculation scores for fixed ratio...\n\nTrain_score = {train_score}\nTest score = {test_score}")

# Cross-validation
print("\n[INFO] Cross-validation initiation...")
kfold = KFold(n_splits=6, random_state=42, shuffle=True)

train_scores = []
test_scores = []


for (train_index, test_index) in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    modelRegress.fit(X_train, y_train)

    # Score for training set
    train_score2 = modelRegress.score(X_train, y_train)
    train_scores.append(train_score2)

    # Score for test set
    test_score2 = modelRegress.score(X_test, y_test)
    test_scores.append(test_score2)

# All CV scores
print(f"[INFO] Calculation scores for cross-validation...\n\nTrain scores = {train_scores}\nTest scores = {test_scores}\n")

# Mean CV scores
print(f"[INFO] Calculation mean scores for cross-validation...")
avg_train_score = np.mean(train_scores)
avg_test_score = np.mean(test_scores)

print(f"\nAverage train score = {avg_train_score}\nAverage test score = {avg_test_score}")

# MODEL EVALUATION (MSE, RMSE)
print("\n[INFO] Model evaluation initiation...")
y_pred = modelRegress.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"[INFO] Evaluating...\n\nMSE = {mse}\nRMSE = {rmse}")

# Actual vs predicted plot
print("\n[INFO] Generating 'Actual vs predicted plot'...")
plt.scatter(y_test, y_pred)
plt.title('Actual vs Predicted Model')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m*y_test+b, color = 'r')
plt.show()


# Residual plot
print("[INFO] Generating 'Residual plot'...\n")
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.show()

# ================================== CLASSIFICATION PROBLEMS  ==================================
print("-"*30,"CLASSIFICATION","-"*30)
X, y = datasets.load_breast_cancer(return_X_y=True)

# MODEL
print("[INFO] Creation a model...")
modelClass = svm.SVC(kernel='linear', C=1)

# DATA SPLIT
# Fixed ratio
print("[INFO] Data splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)
modelClass.fit(X_train, y_train)
train_score = modelClass.score(X_train, y_train)
test_score = modelClass.score(X_test, y_test)
print(f"\n[INFO] Calculation scores for fixed ratio...\n\nTrain_score = {train_score}\nTest score = {test_score}")

# Cross-validation
print("\n[INFO] Cross-validation initiation...")
kfold = KFold(n_splits=6, random_state=42, shuffle=True)

train_scores = []
test_scores = []

for (train_index, test_index) in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    modelClass.fit(X_train, y_train)

    # Score for training set
    train_score2 = modelClass.score(X_train, y_train)
    train_scores.append(train_score2)

    # Score for test set
    test_score2 = modelClass.score(X_test, y_test)
    test_scores.append(test_score2)

# All CV scores
print(f"[INFO] Calculation scores for cross-validation...\n\nTrain scores = {train_scores}\nTest scores = {test_scores}")

# Mean CV scores
print(f"\n[INFO] Calculation mean scores for cross-validation...")
avg_train_score = np.mean(train_scores)
avg_test_score = np.mean(test_scores)

print(f"\nAverage train score = {avg_train_score}\nAverage test score = {avg_test_score}")

# MODEL EVALUATION
print("\n[INFO] Model evaluation initiation...")
y_pred = modelClass.predict(X_test)

# tp, tn, fp, fn calculation
print("[INFO] Calculation of tp, tn, fp and fn...")
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

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot(cmap="Blues")
plt.title("Confusion matrix")
plt.show()

# Metrics
print("[INFO] Calculation of metrics...")
try:
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    print(f"\nAccuracy = {accuracy}\nPrecision = {precision}\nRecall = {recall}\nF-score = {fscore}")
except ZeroDivisionError:
    print("[ERROR] Division by zero is not possible!")

