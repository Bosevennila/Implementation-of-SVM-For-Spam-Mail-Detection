# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step-1 : Import the required packages.

Step-2 : Import the dataset to operate on.

Step-3 : Split the dataset.

Step-4 : Predict the required output. 

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: CHANDRAPRIYADHARSHINI C
RegisterNumber:  212223240019
```
```
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:

### data
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/5a5d0c80-e8c1-4cff-9105-4c8a688c6e6b)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
