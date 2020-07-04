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

os.mkdir('final_train_dir')
for i in train_csv['target'].unique():
    os.mkdir('final_train_dir\\'+i)
    
for i in train_csv['target'].unique():
    for j in X[y==i]:
        copyfile('train\\'+j, 'final_train_dir\\'+i+'\\'+j)
        

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

final_train_datagen=ImageDataGenerator(rescale=1/255)

image_size=50
train_generator=final_train_datagen.flow_from_directory(
        r"D:\Spyder\hacker-earth-dance-form-prediction\final_train_dir",
        target_size=(image_size,image_size),
#        batch_size=128,
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
    
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.optimizers.Adam(),metrics=['accuracy'])

history=model.fit(train_generator,
#                  steps_per_epoch=8,
                  epochs=15,
                  verbose=1,
#                  validation_data=validation_generator,
#                  callbacks=[metrics]
#                  validation_steps=8
                  )

test_datagen=ImageDataGenerator(rescale=1/255)
test_generator=test_datagen.flow_from_directory(
        r'D:\Spyder\hacker-earth-dance-form-prediction\test',
        target_size=(image_size,image_size),
#       color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
        )

import numpy as np
pred=model.predict_generator(test_generator,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results["Filename"]=results["Filename"].apply(lambda x:x[7:])

test_csv=pd.read_csv("test.csv")

results.set_index(["Filename"],inplace=True)
test_csv.set_index(["Image"],inplace=True)

output=test_csv.merge(results,left_index=True,right_index=True)
output.index.name='Image'
output.rename(columns={'Predictions':'target'},inplace=True)
output.to_csv('submission12.csv')
