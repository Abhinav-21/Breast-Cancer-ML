import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

data = pd.read_csv("datasets/breast_cancer/breast_cancer.csv")
data = data.dropna()

data["Tumour_Stage"] = data["Tumour_Stage"].map({"I": 1, "II": 2, "III": 3})
data["Histology"] = data["Histology"].map({"Infiltrating Ductal Carcinoma": 1,"Infiltrating Lobular Carcinoma": 2,"Mucinous Carcinoma": 3})
data["ER status"] = data["ER status"].map({"Positive": 1})
data["PR status"] = data["PR status"].map({"Positive": 1})
data["HER2 status"] = data["HER2 status"].map({"Positive": 1, "Negative": 2})
data["Gender"] = data["Gender"].map({"MALE": 0, "FEMALE": 1})
data["Surgery_type"] = data["Surgery_type"].map({"Other": 1, "Modified Radical␣Mastectomy": 2,"Lumpectomy": 3, "Simple␣Mastectomy": 4})
data = data.dropna()
x = np.array(data[['Age', 'Gender', 'Protein1', 'Protein2','Protein3','Protein4','Tumour_Stage', 'Histology', 'ER status', 'PR status','HER2 status', 'Surgery_type']])
y = np.array(data[['Patient_Status']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10,random_state=42)

model = SVC()
model.fit(x_train,y_train)
features = np.array([[76.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 3, 1, 1, 2, 4,]])
print(model.predict(features))
y_predict = model.predict(x_test)
print(y_predict)
confusion_matrix = confusion_matrix(y_test,y_predict)
true_positive = confusion_matrix[0][0]
false_positive = confusion_matrix[0][1]
false_negative = confusion_matrix[1][0]
true_negative = confusion_matrix[1][1]

accuracy = (true_positive + true_negative) / (true_positive + false_positive+false_negative + true_negative)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1Score = 2 * (recall * precision) / (recall + precision)
print(accuracy, precision, recall, f1Score)