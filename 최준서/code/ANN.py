import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import os

# dataset
os.chdir("..")  # code 폴더에서 메인 폴더로 접근하기 위해서
dataset = pd.read_csv("dataset/Churn_Modelling.csv")  # 데이터셋 불러오기


# 데이터 전처리
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# ANN 구현
ANN = tf.keras.models.Sequential()
ANN.add(tf.keras.layers.Dense(units=6, activation='relu'))  # 은닉층 1
ANN.add(tf.keras.layers.Dense(units=6, activation='relu'))  # 은닉층 2
ANN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # 결과층

# ANN 훈련
ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ANN.fit(x=x_train, y=y_train, batch_size=32, epochs=100)

# ANN 검증
y_pred = ANN.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
