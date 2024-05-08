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
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: CHANDRAPRIYADHARSHINI C
RegisterNumber:  212223240019
*/
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

### Data
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/5a5d0c80-e8c1-4cff-9105-4c8a688c6e6b)

### Data.shape
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/11200aa0-1b4a-4b20-a775-1312edadcaae)

### x.shape and y.shape
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/ed46172d-4d90-406e-a52b-b98bcc653e2a)

### x_train
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/2879f5f8-3489-4c7b-8bc4-df9e179bcaab)

### x_train.shape
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/a98ac44a-d6e9-4f72-8d45-81529fdd0bba)

### SVC
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/651a7650-9e8f-4cd5-803a-07e7c9bf5b28)

### y_pred
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/642f034c-534f-45d8-9766-7d09e618e7a4)

### Accuracy score
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/6e766951-9e49-43c9-9808-ffe20072b093)

### Confusion matrix
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/535f130f-8b19-4bbf-b965-e9e4b0d83b97)

### Classification report
![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/ff633782-30ba-4ff3-8048-82420e560730)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
