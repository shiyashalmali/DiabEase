import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from sklearn.metrics import accuracy_score

diabetes_data=pd.read_csv('diabetes.csv')

x=diabetes_data.drop(columns='Outcome',axis=1)
y=diabetes_data['Outcome']
scaler = StandardScaler()
scaler.fit(x)
standard_data = scaler.transform(x)
x=standard_data
y=diabetes_data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# x_train_predict = classifier.predict(x_train)
# training_data_accuracy = accuracy_score(x_train_predict, y_train)
#
# print('Accuracy: ', training_data_accuracy)

pickle.dump(classifier, open("diabetic.pkl", "wb"))