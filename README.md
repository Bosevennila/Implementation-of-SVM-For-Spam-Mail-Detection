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

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.tail()

data.isnull().sum()

x=data['v1'].values

y=data['v2'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## data.head()

![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/1bb3899c-3dd8-4bb7-a7d2-cd264fb7953a)

## data.info()

![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/aff70fd5-c06a-4dfe-ae72-97b7c583dbc3)

## data.tail()

![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/8185c7cb-53a4-4d50-a3c5-dbcea6a5faf3)

## data.isnull().sum()

![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/45fb4e71-7231-4a3c-ac41-6f91095c2392)

## y_pred

![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/bfa86388-0e87-4c56-8a66-ecbc5f67b56e)

## Accuracy

![image](https://github.com/Bosevennila/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870486/45bca7f0-571d-4a3c-8f1b-ab8897f287f1)


## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
