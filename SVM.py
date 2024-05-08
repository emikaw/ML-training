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
print("[INFO] Model generation...")
modelRegress = LinearRegression()

# DATA SPLIT
# Fixed proportion
print("[INFO] Data splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)
modelRegress.fit(X_train, y_train)
y_pred = modelRegress.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"[INFO] Calculation scores for fixed ratio...\n\n\nMSE = {mse}\nRMSE = {rmse}")
y_pred = modelRegress.predict(X_test)

# Cross-validation
print("\n[INFO] Cross-validation initiation...")
kfold = KFold(n_splits=6, random_state=42, shuffle=True)

mse_scores = []
rmse_scores = []

for (train_index, test_index) in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    modelRegress.fit(X_train, y_train)

    # MSE
    y_pred = modelRegress.predict(X_test)
    mse2 = metrics.mean_squared_error(y_test, y_pred)
    mse_scores.append(mse2)
    
    # RMSE 
    rmse2 = np.sqrt(mse2)
    rmse_scores.append(rmse2)

# All CV scores
print(f"[INFO] Calculation scores for cross-validation...\n\nMSE = {mse_scores}\nRMSE = {rmse_scores}")

# Mean CV scores
print(f"[INFO] Calculation mean scores for cross-validation...")
avg_mse_score = np.mean(mse_scores)
avg_rmse_score = np.mean(rmse_scores)

print(f"\nMSE = {avg_mse_score}\nRMSE = {avg_rmse_score}")

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
def PosNegCalculation(y_pred):
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
    
    return tp, tn, fp, fn

def Metrics(tp, tn, fp, fn):
    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = (2 * precision * recall) / (precision + recall)
        return accuracy, precision, recall, fscore
    except ZeroDivisionError:
        print("[ERROR] Division by zero is not possible!")
    
print("-"*30,"CLASSIFICATION","-"*30)
X, y = datasets.load_breast_cancer(return_X_y=True)

# MODEL
print("[INFO] Model generation...")
modelClass = svm.SVC(kernel='linear', C=1)

# DATA SPLIT
# Fixed ratio
print("[INFO] Data splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)
modelClass.fit(X_train, y_train)
y_pred = modelClass.predict(X_test)
print("[INFO] Metrics calculation...\n")
tp, tn, fp, fn = PosNegCalculation(y_pred)
ac, prec, rec, fs = Metrics(tp, tn, fp, fn)
print(f"Accuracy = {ac}\nPrecision = {prec}\nRecall = {rec}\nFscore = {fs}") 

# Confusion matrix
print("[INFO] Confusion matrix generation...")
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot(cmap="Blues")
plt.title("Confusion matrix")
plt.show()


# Cross-validation
print("\n[INFO] Cross-validation initiation...")
kfold = KFold(n_splits=6, random_state=42, shuffle=True)

accuarcy_scores = []
precision_scores = []
recall_scores = []
fscores_scores = []

for (train_index, test_index) in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    modelClass.fit(X_train, y_train)
    y_pred = modelClass.predict(X_test)

    tp, tn, fp, fn = PosNegCalculation(y_pred)
    ac, prec, rec, fs = Metrics(tp, tn, fp, fn)
    accuarcy_scores.append(ac)
    precision_scores.append(prec)
    recall_scores.append(rec)
    fscores_scores.append(fs)

avg_accuarcy_scores = np.mean(accuarcy_scores)
avg_precision_scores = np.mean(precision_scores)
avg_recall_scores = np.mean(recall_scores)
avg_fscores_scores = np.mean(fscores_scores)

print(f"Accuracy = {avg_accuarcy_scores}\nPrecision = {avg_precision_scores}\nRecall = {avg_recall_scores}\nFscore = {avg_fscores_scores}")
