import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

data = pd.read_csv("datasets/diabetes/diabetes.csv")
data = data.dropna()

data['gender'] = data['gender'].map({"Male":1, "Female":0})
data['smoking_history'] = data['smoking_history'].map({"No Info": 0, "never": 1, "former": 2, "current": 3, "not current": 4})
data = data.dropna()
x = np.array(data[['gender','age','hypertension','heart_disease','smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
y = np.array(data[['diabetes']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10,random_state=42)

model = SVC()
filename = "model.joblib"

# ! uncomment this to train model
# model.fit(x_train,y_train)
# joblib.dump(model, filename)

model = joblib.load(filename)
y_predict = model.predict(x_test)
z = np.array([[1, 20, 0, 0, 1, 20, 5.4, 95]])
print(model.predict(z))
count = 0
ycount = 0
ncount = 0
for i in y_predict:
    count += 1
    if i == 1:
        ycount += 1
    else:
        ncount+= 1

confusion_matrix = confusion_matrix(y_test,y_predict)
true_positive = confusion_matrix[0][0]
false_positive = confusion_matrix[0][1]
false_negative = confusion_matrix[1][0]
true_negative = confusion_matrix[1][1]

accuracy = (true_positive + true_negative) / (true_positive + false_positive+false_negative + true_negative)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1Score = 2 * (recall * precision) / (recall + precision)
print('total: ',count, ", Yes: ",ycount,', No: ',ncount, sep="")
print('true_positive: ', true_positive, ', true_negative: ', true_negative, ', false_positive: ', false_positive, ', false_negative: ', false_negative, sep = "")
print('Accuracy: ',round(100*accuracy, 2),', Precision: ', round(precision,2),', Recall: ', round(recall,2),', F1_val: ', round(f1Score,2), sep="")