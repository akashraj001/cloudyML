import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import shutil
image_dir = r'D:\Spyder\hacker-earth-dance-form-prediction'
os.chdir(image_dir)
os.listdir(image_dir)

train_csv=pd.read_csv("train.csv")
train_csv['target'].value_counts()

X=train_csv['Image']
y=train_csv['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)

y_train.value_counts()
y_test.value_counts()



os.mkdir('train_dir')
for i in train_csv['target'].unique():
    os.mkdir('train_dir\\'+i)

os.mkdir('validation_dir')
for i in train_csv['target'].unique():
    os.mkdir('validation_dir\\'+i)
    
for i in train_csv['target'].unique():
    for j in X_train[y_train==i]:
        copyfile('train\\'+j, 'train_dir\\'+i+'\\'+j)
        
for i in train_csv['target'].unique():
    for j in X_test[y_test==i]:
        copyfile('train\\'+j, 'validation_dir\\'+i+'\\'+j)
