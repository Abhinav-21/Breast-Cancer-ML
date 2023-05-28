import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

data = pd.read_csv("datasets/obesity/obesity.csv")
data = data.dropna()

data['Gender'] = data['Gender'].map({"Male":1, "Female":0})

x = np.array(data[['Age','Gender','Height','Weight','BMI']])
y = np.array(data[['Label']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10,random_state=42)

model = SVC()
filename = "model_obesity.joblib"

# ! uncomment this to train model
# model.fit(x_train,y_train)
# joblib.dump(model, filename)

model = joblib.load(filename)

# ! you can edit this feature array to use custom data.
# feature = np.array([[20, 1, 170, 59, 20.41]])
# print(model.predict(feature)[0])

y_predict = model.predict(x_test)
# print(y_predict)

confusion_matrix = confusion_matrix(y_test,y_predict)
true_positive = confusion_matrix[0][0]
false_positive = confusion_matrix[0][1]
false_negative = confusion_matrix[1][0]
true_negative = confusion_matrix[1][1]

accuracy = (true_positive + true_negative) / (true_positive + false_positive+false_negative + true_negative)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1Score = 2 * (recall * precision) / (recall + precision)

print('true_positive: ', true_positive, '\ntrue_negative: ', true_negative, '\nfalse_positive: ', false_positive, '\nfalse_negative: ', false_negative, sep = "")
print()
print('Accuracy: ',round(accuracy, 2),'\nPrecision: ', round(precision,2),'\nRecall: ', round(recall,2),'\nF1_val: ', round(f1Score,2), sep="")