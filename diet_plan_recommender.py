import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import Tkinter as tk
from tkinter import *
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score

data=pd.read_csv('food.csv')

BreakfastFoodData=data['Breakfast']
BreakfastDataNumpy=BreakfastFoodData.to_numpy()

LunchFoodData=data['Lunch']
LunchFoodDataNumpy=LunchFoodData.to_numpy()

DinnerFoodData=data['Dinner']
DinnerFoodDataNumpy=DinnerFoodData.to_numpy()

FoodItemsData=data['Food_items']

def print_user_input():
    print("\n Age: %s\n Gender: %s\n Weight: %s kg\n Height: %s cm\n Activity level: %s\n Veg or NonVeg: %s\n " % (e1.get(), e2.get(),e3.get(), e4.get(), e5.get(), e6.get()))

def Weight_Loss_Plan():
    print_user_input()
    
    BreakfastFoodItem=[]
    LunchFoodItem=[]
    DinnerFoodItem=[]
        
    BreakfastFoodItemID=[]
    LunchFoodItemID=[]
    DinnerFoodItemID=[]

    # tach thuc an cho 3 buoi sang, trua, toi
    for i in range(len(BreakfastFoodData)):
        if BreakfastDataNumpy[i]==1:
            BreakfastFoodItem.append(FoodItemsData[i] )
            BreakfastFoodItemID.append(i)
        if LunchFoodDataNumpy[i]==1:
            LunchFoodItem.append(FoodItemsData[i])
            LunchFoodItemID.append(i)
        if LunchFoodDataNumpy[i]==1:
            DinnerFoodItem.append(FoodItemsData[i])
            DinnerFoodItemID.append(i)

    # lay food nutrition cho cac mon an buoi sang 
    BreakfastFoodItemIDData = data.iloc[BreakfastFoodItemID]
    BreakfastFoodItemIDData=BreakfastFoodItemIDData.T # dao nguoc row->col va col->row
    val=list(np.arange(5,16)) #[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    Valapnd=[0]+[4]+val # [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    BreakfastFoodItemIDData=BreakfastFoodItemIDData.iloc[Valapnd] # drop column Breakfast,Lunch,Dinner
    BreakfastFoodItemIDData=BreakfastFoodItemIDData.T

    # lay food nutrition cho cac mon an buoi trua
    LunchFoodItemIDdata = data.iloc[LunchFoodItemID]
    LunchFoodItemIDdata = LunchFoodItemIDdata.T
    val=list(np.arange(5,16))
    Valapnd=[0]+[4]+val
    LunchFoodItemIDdata=LunchFoodItemIDdata.iloc[Valapnd]
    LunchFoodItemIDdata=LunchFoodItemIDdata.T 

    # lay food nutrition cho cac mon an buoi toi
    DinnerFoodItemIDdata = data.iloc[DinnerFoodItemID]
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.T
    val=list(np.arange(5,16))
    Valapnd=[0]+[4]+val
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.iloc[Valapnd]
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.T

    # tinh BMI
    age=int(e1.get())
    weight=float(e3.get())
    height=float(e4.get())
    bmi = weight/((height/100)**2)

    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                print('age is between',str(lp),str(lp+10))
                agecl=round(lp/20) # [0,1,2,3,4]

    print("Your body mass index is: ", bmi)
    if ( bmi < 16):
        print("severely underweight")
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        print("underweight")
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        print("Healthy")
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        print("overweight")
        clbmi=1
    elif ( bmi >=30):
        print("severely overweight")
        clbmi=0   

    val1=DinnerFoodItemIDdata.describe()
    # print (val1)
    # valTog=val1.T
    # print (valTog.shape)
    # print (valTog)
    abc = BreakfastFoodItemIDData
    BreakfastFoodItemIDData=BreakfastFoodItemIDData.to_numpy()
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.to_numpy()
    LunchFoodItemIDdata=LunchFoodItemIDdata.to_numpy()
    ti=(clbmi+agecl)/2
  
    ## K-Means Based  Breakfast Food
    #Importing the standard scaler module and applying it on continuous variables
    Datacalorie=BreakfastFoodItemIDData[0:,1:len(BreakfastFoodItemIDData)] #nutrion data
    # print(Datacalorie)
    S = StandardScaler()
    scaled_data = S.fit_transform(Datacalorie)
    print(scaled_data)

    k_means = KMeans(n_clusters=3)
    k_means.fit(scaled_data)
    print(k_means.labels_)

    #To determine the optimum number of clusters, check the wss score for a given range of k
    wss =[] 
    for i in range(1,11):
        KM = KMeans(n_clusters=i)
        KM.fit(scaled_data)
        wss.append(KM.inertia_)
    print(wss)
    # plt.plot(range(1,11), wss, marker = '*')
    # plt.show()

    #Checking for n-clusters=3
    k_means_three = KMeans(n_clusters = 3)
    y_kmeans = k_means_three.fit(scaled_data)
    print('WSS for K=3:', k_means_three.inertia_)
    labels_three = k_means_three.labels_
    print(labels_three)
    #Calculating silhouette_score for k=3
    print(silhouette_score(scaled_data, labels_three))

    length = len(BreakfastFoodItemIDData) + 2
    abc['KMCluster'] = k_means.labels_
    # abc = abc.iloc[:,2:length].astype(float)
    # abc["KMCluster"] = pd.to_numeric(abc["KMCluster"], downcast="float")
    # print(abc.iloc[:,1:length])
    clust_profile=abc.iloc[:,1:length].astype(float).groupby(abc['KMCluster']).mean()
    clust_profile['KMFrequency']=abc.KMCluster.value_counts().sort_index()
    print(clust_profile)

    # Datacalorie=BreakfastFoodItemIDData[0:,1:len(BreakfastFoodItemIDData)] #nutrion data
    # X = np.array(Datacalorie)
    # print(X)
    # kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    # XValu=np.arange(0,len(kmeans.labels_))
    # brklbl=kmeans.labels_
    # print ('## Prediction Result ##')
    # print(brklbl)

    # kmeanss = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
    # y_kmeans = kmeanss.fit_predict(X)
    # arr=y_kmeans
    # abc['cluster'] = arr.tolist()
    # print(abc)

    # XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    # print(len(brklbl))
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    # plt.show()

if __name__ == '__main__':
    main_win = Tk()
    
    Label(main_win,text="Age").grid(row=0,column=0,sticky=W,pady=4)
    Label(main_win,text="Gender (1/0)").grid(row=1,column=0,sticky=W,pady=4)
    Label(main_win,text="Weight (in kg)").grid(row=2,column=0,sticky=W,pady=4)
    Label(main_win,text="Height (in cm)").grid(row=3,column=0,sticky=W,pady=4)
    Label(main_win,text="Activity level").grid(row=4,column=0,sticky=W,pady=4)
    Label(main_win,text="Veg/Non-veg (1/0)").grid(row=5,column=0,sticky=W,pady=4)

    e1 = Entry(main_win)
    e2 = Entry(main_win)
    e3 = Entry(main_win)
    e4 = Entry(main_win)
    e5 = Entry(main_win)
    e6 = Entry(main_win)

    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    e3.grid(row=2, column=1)
    e4.grid(row=3, column=1)
    e5.grid(row=4, column=1)
    e6.grid(row=5, column=1)

    # Button(main_win,text='Quit',command=main_win.quit).grid(row=5,column=0,sticky=W,pady=4)
    # Button(main_win,text='Weight Loss',command=Weight_Loss).grid(row=1,column=4,sticky=W,pady=4)
    # Button(main_win,text='Weight Gain',command=Weight_Gain).grid(row=2,column=4,sticky=W,pady=4)
    # Button(main_win,text='Healthy',command=Healthy).grid(row=3,column=4,sticky=W,pady=4)
    Button(main_win,text='Weight Loss',command=Weight_Loss_Plan).grid(row=3,column=4,sticky=W,pady=4)
    main_win.geometry("400x200")
    main_win.wm_title("DIET RECOMMENDATION SYSTEM")

    main_win.mainloop()


