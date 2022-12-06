# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import irdatacleaning as ird
import islanders as ir
# if .csv=xls
#df= read_csv("C:/Users/asus/Downloads/DT/assign/Diabetes.csv")

olympic = pd.read_csv("C:/Users/L E N O V O/Documents/ai day 5/Winter_Olympic_Medals.csv")
    
olympic.info()

olympic.head()
olympic
#
olympic.drop(columns =["Year", "Host_country", "Host_city", "Country_Code" ], inplace=True)
#reshape data
new_data = pd.DataFrame(olympic.groupby(by="Country_Name").sum().sort_values(by="Gold", ascending=False))
new_data
#last 5 rows
gets_gold = []
for i in new_data.Gold:
    if i >0:
        gets_gold.append(1)
    else:
        gets_gold.append(0)
        
        gets_gold
        #gets new data that drops Gold column
new_data["gets_gold"] = gets_gold        
new_data.drop(columns="Gold", inplace=True)
new_data
#counrty name is not part of the data
new_data.columns
Index(["Silver", "Bronze", "gets_gold"], dtype="object")
#set features
X = np.array(new_data.iloc[:,:-1].values)
y = np.array(new_data.iloc[:,-1].values)
predict = np.array(new_data.iloc[-5:,:-1].values)
predict
#desc tree build the model
#islander ir dec
#===
#DT
#this class is designed to work with scikit-learns DecisionTreeClassifier by making it so that all you have to do when you initiate this class is 
#set the X and Y values DT(X values,Y values, test_size=.2) once you have initialized the class you can build the model by calling the 
#build merthod there are two ways to call this methodthe first way is dt = self.build() which will just return the model itself, 
#the next way is a bit more useful I think, dt,X_test,y_test = dec.build(True) this method will return not just the model but the X_test data 
#and the y_test allowing you to run test on the model. This class is designed to return the most optimized model you can possible have the
# last method allows you to see what your tree looks like you can call this method by running self.show()
#======

dec = ir.DT(X,y,test_size=0.25) # check accuracy
#dt=dec.build(test = True)
dt,X_test,y_test = dec.build(test=True)
#desc classfier
dt.score(X_test,y_test)
dec.show()
#probability sklearn: which counrty wil get their 1st gold in olympic
dt.predict_proba(predict)
new_data
