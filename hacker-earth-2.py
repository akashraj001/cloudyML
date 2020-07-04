import os
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import shutil
image_dir = r'D:\Spyder\hacker-earth-dance-form-prediction' #Give your directory path here after extracting files from zip file provided by hackerearth
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
        
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
train_datagen=ImageDataGenerator(rescale=1/255)
validation_datagen=ImageDataGenerator(rescale=1/255)

image_size=50
train_generator=train_datagen.flow_from_directory(
        r"D:\Spyder\hacker-earth-dance-form-prediction\train_dir",
        target_size=(image_size,image_size),
#        batch_size=128,
        class_mode='sparse'
        )

validation_generator=validation_datagen.flow_from_directory(
        r'D:\Spyder\hacker-earth-dance-form-prediction\validation_dir',
        target_size=(image_size,image_size),
#        batch_size=32,
        class_mode='sparse'
        )



model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(image_size,image_size,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
#        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#        tf.keras.layers.MaxPooling2D(2,2),
#        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dense(8,activation=tf.nn.softmax)
        ])

#from tensorflow.keras.optimizers import RMSprop
    

    
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.optimizers.Adam(),metrics=['accuracy'])

history=model.fit(train_generator,
#                  steps_per_epoch=8,
                  epochs=15,
                  verbose=1,
                  validation_data=validation_generator,
#                  callbacks=[metrics]
#                  validation_steps=8
                  )
