# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 20:26:48 2021

@author: SAI RAKSHITHA
"""
from tkinter.constants import BOTTOM, TOP
import pandas as pd
import numpy as np
import sklearn as sk
import tkinter as tk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



def Weather():
# read the cleaned data
    data = pd.read_csv("austin_final_final.csv")

# the features or the 'x' values of the data
# these columns are used to train the model
# the last column, i.e, precipitation column
# will serve as the label
    X = data.drop(['PrecipitationSumInches'], axis = 1)

# the output or the label.
    Y = data['PrecipitationSumInches']
# reshaping it into a 2-D vector
    Y = Y.values.reshape(-1, 1)
    
    
# consider a random day in the dataset
# we shall plot a graph and observe this
# day
    day_index = 400
    days = [i for i in range(Y.size)]
    
# initialize a linear regression classifier
    clf = LinearRegression()
# train the classifier with our
# input data.
    clf.fit(X,Y)

# give a sample input to test our model
# this is a 2-D vector that contains values
# for each column in the dataset.


    inp = np.array([[TempHighF.get()], [TempAvgF.get()], [TempLowF.get()], [DewPointHighF.get()], [DewPointAvgF.get()], [DewPointLowF.get()], [HumidityHighPercent.get()], [HumidityAvgPercent.get()],
                [HumidityLowPercent.get()], [SeaLevelPressureAvgInches.get()], [VisibilityHighMiles.get()], [VisibilityAvgMiles.get()], [VisibilityLowMiles.get()],[0],[WindHighMPH.get()], [WindAvgMPH.get()],[WindGustMPH.get()]])
    inp = inp.reshape(1, -1)
    inp=inp.astype(np.float64)
# print the output.  
    op=clf.predict(inp)
    #print('The precipitation in inches for the input is: ', op[0][0])
    strn="The precipitation in inches for the input is:"
    newstr=strn+str(op[0][0])
    result1.configure(text=newstr)
    #result2.configure(text="High Rainfall to be Expected..")
    if(op>0.5):
       # print("High Rainfall to be Expected..")
        result2.configure(text="    High Rainfall to be Expected..")
    elif(op>0.3 and op<0.5):
       # print("Moderate Rainfall to be expected..")
        result2.configure(text="Moderate Rainfall to be expected..")
    else:
       # print("Low Rainfall to be...")
        result2.configure(text="             Low Rainfall to be expected...")
    
# plot a graph of the precipitation levels
# versus the total number of days.
# one day, which is in red, is
# tracked here.
    print("the precipitation trend graph: ")
    plt.scatter(days, Y, color = 'b')
    plt.scatter(days[day_index], Y[day_index], color ='r')
    plt.title("Precipitation level")
    plt.xlabel("Days")
    plt.ylabel("Precipitation in inches")


    plt.show()
    x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
                  'SeaLevelPressureAvgInches', 'VisibilityAvgMiles',
                  'WindAvgMPH'], axis = 1)

# plot a graph with a few features (x values)
# against the precipitation or rainfall to observe
# the trends

    print("Precipitation vs selected attributes graph: ")

    for i in range(x_vis.columns.size):
        plt.subplot(3, 2, i + 1)
        plt.scatter(days, x_vis[x_vis.columns.values[i][:100]],
                                               color = 'r')

    plt.scatter(days[day_index],
                x_vis[x_vis.columns.values[i]][day_index],
                color ='b')

    plt.title(x_vis.columns.values[i])

    plt.show()
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(Y,op)

mainroot=tk.Tk()
#mainroot.maxsize(width=800,height=500)mainroot.minsize(width=800,height=500)
root=tk.Frame(mainroot)


heading=tk.Label(mainroot,text="Weather prediction using time series analysis",font=("Arial",30,"bold"))

TempHighLa=tk.Label(root,text="TempHigh(F)",font=("Arial",15,"bold"))
TempHighF=tk.Entry(root,width=10,font=("Arial",15,"bold"))
TempAvgLa=tk.Label(root,text="TempAvg(F)",font=("Arial",15,"bold"))
TempAvgF=tk.Entry(root,width=10,font=("Arial",15,"bold"))
TempLowLa=tk.Label(root,text="TempLow(F)",font=("Arial",15,"bold"))
TempLowF=tk.Entry(root,width=10,font=("Arial",15,"bold"))
DewHighLa=tk.Label(root,text="DewHigh(F)",font=("Arial",15,"bold"))
DewPointHighF=tk.Entry(root,width=10,font=("Arial",15,"bold"))
DewAvgLa=tk.Label(root,text="DewAvg(F)",font=("Arial",15,"bold"))
DewPointAvgF=tk.Entry(root,width=10,font=("Arial",15,"bold"))
DewLowLa=tk.Label(root,text="DewLow(F)",font=("Arial",15,"bold"))
DewPointLowF=tk.Entry(root,width=10,font=("Arial",15,"bold"))
HumidityHighLa=tk.Label(root,text="HumidityHigh(%)",font=("Arial",15,"bold"))
HumidityHighPercent=tk.Entry(root,width=10,font=("Arial",15,"bold"))
HumidityAvgLa=tk.Label(root,text="HumidityAvg(%)",font=("Arial",15,"bold"))
HumidityAvgPercent=tk.Entry(root,width=10,font=("Arial",15,"bold"))
HumidityLowLa=tk.Label(root,text="HumidityLow(%)",font=("Arial",15,"bold"))
HumidityLowPercent=tk.Entry(root,width=10,font=("Arial",15,"bold"))
SeaLevelPressureLa=tk.Label(root,text="SeaLevelPressure(Inches)",font=("Arial",15,"bold"))
SeaLevelPressureAvgInches=tk.Entry(root,width=10,font=("Arial",15,"bold"))
VisibilityHighLa=tk.Label(root,text="VisibilityHigh(Miles)",font=("Arial",15,"bold"))
VisibilityHighMiles=tk.Entry(root,width=10,font=("Arial",15,"bold"))
VisibilityAvgLa=tk.Label(root,text="VisibilityAvg(Miles)",font=("Arial",15,"bold"))
VisibilityAvgMiles=tk.Entry(root,width=10,font=("Arial",15,"bold"))
VisibilityLowLa=tk.Label(root,text="VisibilityLow(Miles)",font=("Arial",15,"bold"))
VisibilityLowMiles=tk.Entry(root,width=10,font=("Arial",15,"bold"))
WindHighLa=tk.Label(root,text="WindHigh(MPH)",font=("Arial",15,"bold"))
WindHighMPH=tk.Entry(root,width=10,font=("Arial",15,"bold"))
WindAvgLa=tk.Label(root,text="WindAvg(MPH)",font=("Arial",15,"bold"))
WindAvgMPH=tk.Entry(root,width=10,font=("Arial",15,"bold"))
WindGustLa=tk.Label(root,text="WindGust(MPH)",font=("Arial",15,"bold"))
WindGustMPH=tk.Entry(root,width=10,font=("Arial",15,"bold"))

but=tk.Button(root,text="Click",command=Weather,font=("Arial",15,"bold"))

result1=tk.Label(mainroot,text="The precipitation in inches for the input is:--.-- ",font=("Arial",30,"bold"))
result2=tk.Label(mainroot,text="  ",font=("Arial",20,"bold"))


root.place(x=150,y=200)
heading.place(x=300,y=50)
result1.place(x=200,y=400)
result2.place(x=425,y=500)


TempHighLa.grid(row=1,column=0)
TempHighF.grid(row=1,column=1)
TempAvgLa.grid(row=2,column=0)
TempAvgF.grid(row=2,column=1)
TempLowLa.grid(row=3,column=0)
TempLowF.grid(row=3,column=1)
DewHighLa.grid(row=4,column=0)
DewPointHighF.grid(row=4,column=1)


DewAvgLa.grid(row=1,column=3)
DewPointAvgF.grid(row=1,column=4)
DewLowLa.grid(row=2,column=3)
DewPointLowF.grid(row=2,column=4)
HumidityHighLa.grid(row=3,column=3)
HumidityHighPercent.grid(row=3,column=4)
HumidityAvgLa.grid(row=4,column=3)
HumidityAvgPercent.grid(row=4,column=4)

HumidityLowLa.grid(row=1,column=6)

HumidityLowPercent.grid(row=1,column=7)
SeaLevelPressureLa.grid(row=2,column=6)
SeaLevelPressureAvgInches.grid(row=2,column=7)
VisibilityHighLa.grid(row=3,column=6)
VisibilityHighMiles.grid(row=3,column=7)
VisibilityAvgLa.grid(row=4,column=6)
VisibilityAvgMiles.grid(row=4,column=7)

VisibilityLowLa.grid(row=1,column=9)
VisibilityLowMiles.grid(row=1,column=10)
WindHighLa.grid(row=2,column=9)
WindHighMPH.grid(row=2,column=10)
WindAvgLa.grid(row=3,column=9)
WindAvgMPH.grid(row=3,column=10)
WindGustLa.grid(row=4,column=9)
WindGustMPH.grid(row=4,column=10)

but.grid(row=5,column=5)

tk.mainloop()
