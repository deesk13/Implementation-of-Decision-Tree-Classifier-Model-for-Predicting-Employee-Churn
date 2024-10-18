# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DEVA DHARSHINI.I
RegisterNumber: 212223240026
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
HEAD() AND INFO():
![Screenshot 2024-10-18 173909](https://github.com/user-attachments/assets/f19d6f68-5e76-41cb-9f0d-e73440f58d09)

NULL & COUNT:
![Screenshot 2024-10-18 173920](https://github.com/user-attachments/assets/293b284b-479e-45bf-ac98-2f18b7845156)
![Screenshot 2024-10-18 173937](https://github.com/user-attachments/assets/ff3a8e59-23f2-49fa-9ed7-4b76a17a3764)

ACCURACY SCORE:
![Screenshot 2024-10-18 173950](https://github.com/user-attachments/assets/949aea66-1763-4f87-90e4-37b965e40429)

DECISION TREE CLASSIFIER MODEL:
![Screenshot 2024-10-18 174004](https://github.com/user-attachments/assets/3de77a04-d29c-43c5-8041-5d00e07f8c08)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
