# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:41:38 2020

@author: DELL
"""
import pandas as pd
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



df =  pd.read_csv("D:\Prabhaker\Data Science\Keras\RA0034_predectors.csv")

X = df.iloc[:,:4].astype(float) # here we have 2 input variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df.iloc[:,4]# output variable (what we are trying to predict)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, Y)

# Make predictions using the testing set
y_pred = regr.predict(X)


print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# tkinter GUI
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()
attributes = ['CHOKE','WHP','FLP','WHT', "OIL_RATE"]
# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)

# CHOKE label and input box
label1 = tk.Label(root, text='CHOKE: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# WHP label and input box
label2 = tk.Label(root, text='WHP: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

# FLP label and input box
label3 = tk.Label(root, text='FLP: ')
canvas1.create_window(140, 140, window=label3)

entry3 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 140, window=entry3)

# WHT label and input box
label4 = tk.Label(root, text='WHT: ')
canvas1.create_window(160, 160, window=label4)

entry4 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 160, window=entry4)

def values(): 
    global New_CHOKE #our 1st input variable
    New_CHOKE = float(entry1.get()) 
    
    global New_WHP #our 2nd input variable
    New_WHP = float(entry2.get()) 
    
    global New_FLP #our 3rd input variable
    New_FLP = float(entry3.get()) 
    
    global New_WHT #our 4th input variable
    New_WHT = float(entry4.get()) 
    
    Prediction_result  = ('Predicted OIL_RATE: ', regr.predict([[New_CHOKE ,New_WHP, New_FLP, New_WHT]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict OIL_RATE',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 180, window=button1)
 
#plot 1st scatter 
figure3 = plt.Figure(figsize=(4,3), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df['CHOKE'].astype(float),df['OIL_RATE'].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend() 
ax3.set_xlabel('CHOKE')
ax3.set_title('CHOKE Vs. OIL_RATE')

#plot 2nd scatter 
figure4 = plt.Figure(figsize=(4,3), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['WHP'].astype(float),df['OIL_RATE'].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend() 
ax4.set_xlabel('WHP')
ax4.set_title('WHP Vs. SOIL_RATE')

#plot 2nd scatter 
figure5 = plt.Figure(figsize=(4,3), dpi=100)
ax5 = figure5.add_subplot(111)
ax5.scatter(df['FLP'].astype(float),df['OIL_RATE'].astype(float), color = 'b')
scatter5 = FigureCanvasTkAgg(figure5, root) 
scatter5.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax5.legend() 
ax5.set_xlabel('FLP')
ax5.set_title('FLP Vs. SOIL_RATE')

#plot 2nd scatter 
figure6 = plt.Figure(figsize=(4,3), dpi=100)
ax6 = figure6.add_subplot(111)
ax6.scatter(df['WHT'].astype(float),df['OIL_RATE'].astype(float), color = 'y')
scatter6 = FigureCanvasTkAgg(figure6, root) 
scatter6.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax6.legend() 
ax6.set_xlabel('WHT')
ax6.set_title('WHT Vs. SOIL_RATE')

root.mainloop()