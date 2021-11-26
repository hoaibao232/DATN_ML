import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score

# read dataset
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
        if DinnerFoodDataNumpy[i]==1:
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

    # tinh BMR
    # Nam - BMR = 10W + 6.25H – 5A + 5
    # Nu - BMR = 10W + 6.25H – 5A - 161
    gender=int(e2.get())
    activity_level=float(e5.get())
    if (gender == 1):
        bmr = 10*weight + 6.25*height - 5*age + 5
    elif (gender == 0):
        bmr = 10*weight + 6.25*height - 5*age - 161

    tdee = bmr * activity_level
    print(bmr)
    print(tdee)

    # val1=DinnerFoodItemIDdata.describe()
    # print (val1)
    # valTog=val1.T
    # print (valTog.shape)
    # print (valTog)

    BreakfastFoodItem_Test = BreakfastFoodItemIDData
    LunchFoodItem_Test = LunchFoodItemIDdata
    DinnerFoodItem_Test = DinnerFoodItemIDdata
    BreakfastFoodItemIDData=BreakfastFoodItemIDData.to_numpy()
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.to_numpy()
    LunchFoodItemIDdata=LunchFoodItemIDdata.to_numpy()
    ti=(clbmi+agecl)/2
  
    ###### K-MEANS FOR BREAKFAST FOOD

    #Importing the standard scaler module and applying it on continuous variables
    BreakfastDatacalorie=BreakfastFoodItemIDData[0:,2:len(BreakfastFoodItemIDData)] #nutrion data
    S = StandardScaler()
    breakfast_scaled_data = S.fit_transform(BreakfastDatacalorie)
    # print(breakfast_scaled_data)

    # First, test Kmeans with clusters=3
    k_means_breakfast = KMeans(n_clusters=3)
    k_means_breakfast.fit(breakfast_scaled_data)
    brklbl=k_means_breakfast.labels_
    print(brklbl)

    #To determine the optimum number of clusters, check the wss score for a given range of k
    wss =[] 
    for i in range(1,11):
        KM_Breakfast = KMeans(n_clusters=i)
        KM_Breakfast.fit(breakfast_scaled_data)
        wss.append(KM_Breakfast.inertia_)
    print(wss)
    plt.plot(range(1,11), wss, marker = '*')
    # plt.show()

    #Checking for n-clusters=3
    k_means_three_breakfast = KMeans(n_clusters = 3)
    k_means_three_breakfast.fit(breakfast_scaled_data)
    print('WSS for K=3:', k_means_three_breakfast.inertia_)
    labels_three = k_means_three_breakfast.labels_
    # print(labels_three)
    #Calculating silhouette_score for k=3
    print(silhouette_score(breakfast_scaled_data, labels_three))

    # Overview data in clusters
    length = len(BreakfastFoodItemIDData) + 2
    BreakfastFoodItem_Test['KMCluster'] = brklbl
    clust_profile=BreakfastFoodItem_Test.iloc[:,2:length].astype(float).groupby(BreakfastFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=BreakfastFoodItem_Test.KMCluster.value_counts().sort_index()
    print(clust_profile)

    # XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    # print(len(brklbl))
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    # plt.show()
    print("--------------------------------------------------------------------")

    ####### K-MEANS FOR LUNCH FOOD
    LunchDatacalorie=LunchFoodItemIDdata[0:,2:len(LunchFoodItemIDdata)]
    L = StandardScaler()
    lunch_scaled_data = L.fit_transform(LunchDatacalorie)
    # print(lunch_scaled_data)

    k_means_lunch = KMeans(n_clusters=3)
    k_means_lunch.fit(lunch_scaled_data)
    lnchlbl=k_means_lunch.labels_
    # print(k_means_lunch.labels_)

    wss =[] 
    for i in range(1,11):
        KM_Lunch = KMeans(n_clusters=i)
        KM_Lunch.fit(lunch_scaled_data)
        wss.append(KM_Lunch.inertia_)
    print(wss)
    plt.plot(range(1,11), wss, marker = '*')
    # plt.show()

    k_means_three_lunch = KMeans(n_clusters = 3)
    k_means_three_lunch.fit(lunch_scaled_data)
    print('WSS for K=3:', k_means_three_lunch.inertia_)
    labels_three = k_means_three_lunch.labels_
    # print(labels_three)
    print(silhouette_score(lunch_scaled_data, labels_three))

    length = len(LunchFoodItemIDdata) + 2
    LunchFoodItem_Test['KMCluster'] = lnchlbl
    clust_profile=LunchFoodItem_Test.iloc[:,2:length].astype(float).groupby(LunchFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=LunchFoodItem_Test.KMCluster.value_counts().sort_index()
    print(clust_profile)
    print("--------------------------------------------------------------------")

    ####### K-MEANS FOR DINNER FOOD
    DinnerDatacalorie=DinnerFoodItemIDdata[0:,2:len(DinnerFoodItemIDdata)] #nutrion data
    D = StandardScaler()
    dinner_scaled_data = D.fit_transform(DinnerDatacalorie)
    # print(dinner_scaled_data)

    k_means_dinner = KMeans(n_clusters=3)
    k_means_dinner.fit(dinner_scaled_data)
    dnrlbl=k_means_dinner.labels_
    print(k_means_dinner.labels_)

    wss =[] 
    for i in range(1,11):
        KM_Dinner = KMeans(n_clusters=i)
        KM_Dinner.fit(dinner_scaled_data)
        wss.append(KM_Dinner.inertia_)
    print(wss)
    plt.plot(range(1,11), wss, marker = '*')
    # plt.show()

    k_means_three_dinner = KMeans(n_clusters=3)
    k_means_three_dinner.fit(dinner_scaled_data)
    print('WSS for K=3:', k_means_three_dinner.inertia_)
    labels_three = k_means_three_dinner.labels_
    print(labels_three)
    print(silhouette_score(dinner_scaled_data, labels_three))

    length = len(DinnerFoodItemIDdata) + 2
    DinnerFoodItem_Test['KMCluster'] = dnrlbl
    clust_profile=DinnerFoodItem_Test.iloc[:,2:length].astype(float).groupby(DinnerFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=DinnerFoodItem_Test.KMCluster.value_counts().sort_index()
    print(clust_profile)

    ### TRAIN SET / TEST SET
    dataset=pd.read_csv('train_data.csv')
    print(dataset.head())
    datasetT=dataset.T
    print(datasetT.head())

    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightloss_nutrion=datasetT.iloc[[1,2,7,8]]
    print(weightloss_nutrion.head())

    weightloss_nutrion=weightloss_nutrion.T
    print(weightloss_nutrion.head())

    weightloss_nutrion_numpy = weightloss_nutrion.to_numpy()
    print(weightloss_nutrion_numpy)

    weightloss_zeros = np.zeros((len(weightloss_nutrion_numpy)*5,6),dtype=np.float32)
    print(weightloss_zeros)

    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightloss_nutrion_numpy)):
            valloc=list(weightloss_nutrion_numpy[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            # print(valloc)
            weightloss_zeros[t]=np.array(valloc)
            # print(weightloss_zeros)
            yt.append(brklbl[jj])
            # print(yt)
            t+=1
            
        for jj in range(len(weightloss_nutrion_numpy)):
            valloc=list(weightloss_nutrion_numpy[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightloss_zeros[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
            
        for jj in range(len(weightloss_nutrion_numpy)):
            valloc=list(weightloss_nutrion_numpy[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightloss_zeros[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    X_test=np.zeros((len(weightloss_nutrion_numpy),6),dtype=np.float32)
    
    for jj in range(len(weightloss_nutrion_numpy)):
        valloc=list(weightloss_nutrion_numpy[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti

    print(X_test)

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


