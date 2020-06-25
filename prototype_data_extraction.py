# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:04:13 2020

@author: Akash Raj
"""
import pandas as pd
import os

os.chdir(r'D:\Machine Learning Training Youtube\Basic Prototype')

data_from_csv=pd.read_csv('data_prototype_csv.csv')
data_from_excel=pd.read_excel('data_prototype_excel.xlsx')

print(data_from_csv['x-axis'])
data_from_csv['y-axis']

data_from_csv.isnull()

data_from_csv.isnull().sum()
data_from_csv.shape


data_from_csv['z-axis']=data_from_csv['x-axis']+data_from_csv['y-axis']

data_from_csv['z-axis'].value_counts()
data_from_csv['z-axis'].unique()
data_from_csv['z-axis'].nunique()

data_from_csv['a-axis']=100

data_from_csv['a-axis'].value_counts()
data_from_csv['a-axis'].unique()
data_from_csv['a-axis'].nunique()
