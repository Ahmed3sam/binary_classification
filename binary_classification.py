# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:18:18 2019

@author: Ahmed Essam Adel
"""

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%
# read the training dataset and take some indication
df = pd.read_csv('training.csv', sep = ';')
df1 = pd.read_csv('validation.csv', sep = ';')

print(df.info())

'''
the dataset has 3700 rows and 19 columns (18 features & 1 class label)
first thing we will see if there is duplicates rows
'''

#%%

#drop duplicates rows
df = df.drop_duplicates()
print(df.info())
print(df.head(5))

'''
there are many duplicates. We have now only 490 rows out of 3700
and there is the problems we have to get answers:
    1 - nulls
    2 - is there duplicate columns 
    3 - is there bias columns
    4 - the columns has 2 values in a row
    5 - convert the characters to numbers
    
so we need to make more analysis on the data
'''
#%%



#count the values of the columns
print(df['variable1'].value_counts())
print(df['variable4'].value_counts())
print(df['variable5'].value_counts())
print(df['variable6'].value_counts())
print(df['variable7'].value_counts())
print(df['variable9'].value_counts())
print(df['variable10'].value_counts())
print(df['variable11'].value_counts())
print(df['variable12'].value_counts())
print(df['variable13'].value_counts())
print(df['variable14'].value_counts())
print(df['variable17'].value_counts())
print(df['variable18'].value_counts())
print(df['variable19'].value_counts())
print(df['classLabel'].value_counts())

'''
We saw that:
    1 - variable2 & variable3 & variable8 have 2 values in a row
    2 - variable1 has 2 characters.
    3 - variable4 has 3 characters.
    4 - variable5 has 3 charchters.
    note that variable4 and variable5 are the same where
    u=g, y=p, l=gg
    
    5 - variable6 has 14 characters.
    6 - variable7 has 9 characters
    7 - variable9 has 2 characters.
    8 - variable10 has 2 characters.
    note the f values in variable10 equal 0 in variable11
    
    9 - variable12 has 2 characters
    10 - variable13 has 3 characters
    11 - variable17 = variable14 * 10000
    12 - variable18 has 2 characters
    13 - variable19 equal classLabel
    
    14 - we have all the alphabets :) and (aa,bb,cc,dd,ff,gg)
    15 - Null problem of course
'''
'''
solutions:
    1 - we will drop duplicates columns (variable5 - variable10 - variable17)
    2 - we will drop bias columns (variable19)
    
    3 - we will take the mean of the 2 values in variable2 & variable8
    4 - the values in variable3 is very small so we will add them
    
    5 - we will change the characters to ascii (and douple character will multiply by 2)
    6 - we will fill nulls with 0 because the dataset became small after dropping the duplicates
    7 - we will apply that in the validation set either

'''

#%%

#drop duplicates and bias columns
df = df.drop(columns=['variable5', 'variable10', 'variable17', 'variable19'])
df1 = df1.drop(columns=['variable5', 'variable10', 'variable17', 'variable19'])

#%%

# the mean of variable2 & variable8
def take_mean(var):
    new = var.str.split(",", n = 1, expand = True)
    new[0] =  pd.to_numeric(new[0])
    new[1] =  pd.to_numeric(new[1])
    new[0] = new[0].fillna(0)
    new[1] = new[1].fillna(0)
    var = new.mean(axis=1)
    return var
    
df['variable2'] = take_mean(df['variable2'])
df['variable8'] = take_mean(df['variable8'])
df1['variable2'] = take_mean(df1['variable2'])
df1['variable8'] = take_mean(df1['variable8'])

#%%

# add variable3 values
def take_sum(var):
    new = var.str.split(",", n = 1, expand = True)
    new1 = new[0].str.cat(new[1],sep=".")
    var = pd.to_numeric(new1)
    return var

df['variable3'] = take_sum(df['variable3'])
df1['variable3'] = take_sum(df1['variable3'])

#%%

#change characters to ascii
def take_ascii(var):
    new =  var.unique()
    for i in new:
        if isinstance(i, str):
            if len(i)==1:
                var = var.replace(i , ord(i)) 
            else:
                y = [c for c in i]
                var = var.replace(i , 2*ord(y[0]))
    return var

df['variable1'] = take_ascii(df['variable1'])
df['variable4'] = take_ascii(df['variable4'])
df['variable6'] = take_ascii(df['variable6'])
df['variable7'] = take_ascii(df['variable7'])
df['variable8'] = take_ascii(df['variable8'])
df['variable9'] = take_ascii(df['variable9'])
df['variable12'] = take_ascii(df['variable12'])
df['variable13'] = take_ascii(df['variable13'])
df['variable18'] = take_ascii(df['variable18'])
df1['variable1'] = take_ascii(df1['variable1'])
df1['variable4'] = take_ascii(df1['variable4'])
df1['variable6'] = take_ascii(df1['variable6'])
df1['variable7'] = take_ascii(df1['variable7'])
df1['variable8'] = take_ascii(df1['variable8'])
df1['variable9'] = take_ascii(df1['variable9'])
df1['variable12'] = take_ascii(df1['variable12'])
df1['variable13'] = take_ascii(df1['variable13'])
df1['variable18'] = take_ascii(df1['variable18'])

#%%
# fill nulls with 0

df = df.fillna(0)
df1 = df1.fillna(0)

#%%

#make svm model

X_train = df.drop('classLabel', axis=1)  
y_train = df['classLabel']  
X_test = df1.drop('classLabel', axis=1)  
y_test = df1['classLabel']  



from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train) 

y_pred = svclassifier.predict(X_test)  



from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  

    





