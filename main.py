import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/BRCA.csv")
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

model=SVC()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(y_pred)
conf_mat = confusion_matrix(y_test,y_pred)
true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]

Accuracy = (true_positive + true_negative) / (true_positive +false_positive+false_negative + true_negative)
Precision = true_positive/(true_positive+false_positive)
Recall = true_positive/(true_positive+false_negative)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print(Accuracy, Precision, Recall, F1_Score)