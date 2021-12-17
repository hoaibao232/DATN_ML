from contextlib import nullcontext
from typing import TypedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydot
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models import DataTable, TableColumn
from streamlit_bokeh_events import streamlit_bokeh_events
import streamlit.components.v1 as components
import jinja2


# read dataset
data=pd.read_csv('food.csv')
BreakfastFoodData=data['Breakfast']
BreakfastDataNumpy=BreakfastFoodData.to_numpy()
LunchFoodData=data['Lunch']
LunchFoodDataNumpy=LunchFoodData.to_numpy()
DinnerFoodData=data['Dinner']
DinnerFoodDataNumpy=DinnerFoodData.to_numpy()
FoodItemsData=data['Food_items']

images = [
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737354/Asparagus_Cooked_jh671t.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737354/Avocados_xh7avy.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737766/Bananas_l4fgxw.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737766/Bagels_made_in_wheat_wgv3xe.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737768/Berries_tzjk9f.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737768/Brocolli_ekgofj.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737768/Brown_Rice_iirip7.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737768/Cauliflower_xrxdxs.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737768/American_cheese_mwgoot.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737767/Coffee_l3rzsd.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737767/Corn_csfhlk.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737767/Dark_chocolates_xg5fge.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737768/Grapes_fm13pq.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737767/Milk_av1bih.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737767/Cashew_Nuts_lusyt6.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737766/Onions_cf304k.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737765/Orange_j3agvs.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737766/Pasta_canned_with_tomato_sauce_m28p2g.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639737765/Pears_zychen.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738561/Peas_kcxpul.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738560/Protein_Powder_hyzmyw.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738560/Pumpkin_zyivzx.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738560/Tuna_Salad_iaf2jy.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738560/Tuna_Fish_znza6j.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738559/Peproni_Pizza_vgpfic.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738559/Cheese_Pizza_g67piw.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738559/French_Fries_mnqjcw.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738559/Chicken_Burger_fecrh2.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738560/Cheese_Burger_rbar2b.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738558/Chicken_Sandwich_wvewjt.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738559/Sugar_Doughnuts_l4iejk.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738558/Chocolate_Doughnuts_zwo7je.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738558/Pop_Corn_-_Caramel_khljyq.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738558/Pop_Corn_dpzuah.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738558/Dosa_tjeime.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738557/Idli_tb6adq.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738558/Poha_g63hie.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738558/Chappati_waz4rn.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639738558/Tomato_sf3vyz.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741014/Yogurt_rlf2rk.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741014/Brownie_ykpxvq.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741014/Noodles_ikm59c.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741014/Uttapam_muoad3.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741014/Bhaji_Pav_datcj6.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741014/Dal_Makhani_uript4.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741013/Almonds_hundni.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741013/mushrooms-in-a-bowel-on-a-dark-table_bzl18x.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741013/Egg_Yolk_cooked_boauwl.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741014/Sweet_Potatoes_cooked_j5os8i.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741013/Boiled_Potatoes_yhzhrt.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741013/White_Rice_x8lakj.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741012/Orange_juice_s1g3mb.webp',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741013/Greek_yogurt_plain_adpop7.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741013/Oat_Bran_Cooked_ibkhfa.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741011/Green_Tea_aa3rx4.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741011/Chia_seeds_zvpgsy.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741011/Cottage_cheese_with_vegetables_t8uji6.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741011/Salmon_ipefts.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639741011/Cereals-Corn_Flakes_ndxzqh.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742985/Beans_emwgo9.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742985/Lentils_d87twv.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742985/Pasta_with_corn_homemade_zmqjfk.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742984/Tea_neqm1i.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742984/Apples_jp2q7q.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742985/Strawberries_x9p2bt.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742984/Quninoa_i0smgs.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742984/Goat_meat_imrpzu.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742984/Rabbit_meat_ajq8xd.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639753678/Chicken_Breast_kkn69p.png',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742983/Steak_Fries_ohnsrk.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742983/Mexican_Rice_dx4lw8.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742983/Fried_Shrimp_g38o5q.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742984/Spaghetti_and_meatballs_qhj8ah.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742985/Pasta_with_corn_homemade_zmqjfk.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742983/Pork_cooked_lrcw40.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742983/Bacon_cooked_cbvrxv.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742983/Nachos_ay8ex0.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742983/Chicken_Popcorn_mxx8qx.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639742982/Turkey_cooked_sgxvb1.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743600/Oyster_cooked_cq6bi1.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743600/Beef_sticks_kkzo8o.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743600/Banana_Chips_alivwt.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743600/Honey_anjya2.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743601/Chocolate_Icecream_poonsd.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743599/Vanilla_Ice_cream_en5pd1.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743599/Strawberry_Icecream_xvhscw.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743599/Marshmallows_yosje3.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743598/Chocolate_milk_fh2iet.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743599/Rice_Pudding_o49uef.jpg' 
          ] 

data['Image'] = pd.Series(images)

def path_to_image_html(path):
    return '<img src="'+ path + '" width="60" >'

def food_data():
    FoodsData = data
    FoodsData=FoodsData.T 
    val=list(np.arange(5,16))
    Valapnd=[0]+[4]+val 
    FoodItemIDData=FoodsData.iloc[Valapnd]
    FoodItemIDData=FoodItemIDData.T

    return FoodItemIDData

def print_user_input():
    print("\n Age: %s\n Gender: %s\n Weight: %s kg\n Height: %s cm\n Activity level: %s\n" % (age, gender, weight, height, activity_level))

def print_prediction_input():
    print("\n Calories: %s\n Fat: %s g\n Protein: %s g\n Carb: %s g " % (age,age, age, age))

def calc_BMI():
    # tinh BMI
    Age=int(age)
    Weight=float(weight)
    Height=float(height)
    BMI = Weight/((Height/100)**2)

    BMI = float("{:.1f}".format(BMI))

    return BMI

def calc_TDEE():
    # tinh BMR
    # Nam - BMR = 10W + 6.25H – 5A + 5
    # Nu - BMR = 10W + 6.25H – 5A - 161
    Age=int(age)
    Weight=float(weight)
    Height=float(height)
    Gender=gender
    
    if activity_level == 'Sedentary':
        Activity_Level1 = 1.2
    if activity_level == 'Lightly active':
        Activity_Level1 = 1.375
    if activity_level == 'Moderately active':
        Activity_Level1 = 1.55
    if activity_level == 'Very active':
        Activity_Level1 = 1.725
    if activity_level == 'Extra active':
        Activity_Level1 = 1.9

    if (gender == 'Male'):
        BMR = 10*Weight + 6.25*Height - 5*Age + 5
    elif (gender == 'Female'):
        BMR = 10*Weight + 6.25*Height - 5*Age - 161

    BMI = calc_BMI()
    if BMI >= 30:
        sub_calories = 500
    else:
        sub_calories = 300

    TDEE = BMR * Activity_Level1
    total_calo = float("{:.0f}".format(TDEE - sub_calories)) 
    total_protein = float("{:.0f}".format((total_calo * 0.4)/4))
    total_carb = float("{:.0f}".format((total_calo * 0.3)/4)) 
    total_fat = float("{:.0f}".format((total_calo - total_protein*4 - total_carb*4)/9)) 

    my_expander = st.expander(label='Health Check!')
    with my_expander:
        str_bmi = "Your body mass index is **{}**".format(BMI)
        str_bmr = "Your Basal metabolic rate is **{} calories**".format(BMR)
        str_tdee = "Your Total Daily Energy Expenditure is **{} calories**".format(TDEE)
        str_calories = "Our Recommend Total Daily Intake Calories is **{} calories**".format(total_calo)
        str_protein = "Protein intake should be **{} gam per day**".format(total_protein)
        str_fat = "Fat intake should be **{} gam per day**".format(total_fat)
        str_carb = "Carbohydrate intake should be **{} gam per day**".format(total_carb)
        
        
        
        if ( BMI < 16):
            str_health = "Your body condition is **Severely Underweight**"
        elif ( BMI >= 16 and BMI < 18.5):
            str_health = "Your body condition is **Underweight**"
        elif ( BMI >= 18.5 and BMI < 25):
            str_health = "Your body condition is **Healthy**"
        elif ( BMI >= 25 and BMI < 30):
            str_health = "Your body condition is **overweight**"
        elif ( BMI >=30):
            str_health = "Your body condition is Severely Overweight"

        st.info(str_bmi + " - " + str_health)
        st.info(str_bmr)
        st.info(str_tdee)
        st.success(str_calories)
        st.success(str_protein)
        st.success(str_fat)
        st.success(str_carb)

        st.balloons()
    # st.info("**Your Total Daily Energy Expenditure is: ", TDEE , 'calories**')
    # st.info("**Our recommend Total Daily Intake Calories is: ", total_calo , 'calories**')
    # st.info("**Total Protein is: ", total_protein , 'g**')
    # st.info("**Total Carbohydrates is: ", total_carb , 'g**')
    # st.info("**Total Fats is: ", total_fat , 'g**')

    return TDEE,total_calo,total_protein,total_carb,total_fat

def meal_food_data():
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
    val=list(np.arange(5,17)) #[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    Valapnd=[0]+[4]+val # [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    BreakfastFoodItemIDData=BreakfastFoodItemIDData.iloc[Valapnd] # drop column Breakfast,Lunch,Dinner
    BreakfastFoodItemIDData=BreakfastFoodItemIDData.T
    print(BreakfastFoodItemIDData)

    # lay food nutrition cho cac mon an buoi trua
    LunchFoodItemIDdata = data.iloc[LunchFoodItemID]
    LunchFoodItemIDdata = LunchFoodItemIDdata.T
    val=list(np.arange(5,17))
    Valapnd=[0]+[4]+val
    LunchFoodItemIDdata=LunchFoodItemIDdata.iloc[Valapnd]
    LunchFoodItemIDdata=LunchFoodItemIDdata.T 

    # lay food nutrition cho cac mon an buoi toi
    DinnerFoodItemIDdata = data.iloc[DinnerFoodItemID]
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.T
    val=list(np.arange(5,17))
    Valapnd=[0]+[4]+val
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.iloc[Valapnd]
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.T

    return BreakfastFoodItemIDData, LunchFoodItemIDdata, DinnerFoodItemIDdata

def breakfast_cluster_food(BreakfastFoodItemIDData, BreakfastFoodItem_Test):
    ###### K-MEANS FOR BREAKFAST FOOD

    #Importing the standard scaler module and applying it on continuous variables
    BreakfastDatacalorie=BreakfastFoodItemIDData[0:,2:len(BreakfastFoodItemIDData)] #nutrion data
    BreakfastDatacalorie=BreakfastDatacalorie[:, :-1]
    print(BreakfastDatacalorie)

    S = StandardScaler()
    breakfast_scaled_data = S.fit_transform(BreakfastDatacalorie)
    breakfast_scaled_data1 = breakfast_scaled_data
    breakfast_scaled_data2=breakfast_scaled_data

    # First, test Kmeans with clusters=3
    k_means_breakfast = KMeans(n_clusters=4, random_state=0)
    k_means_breakfast.fit(breakfast_scaled_data)
    brklbl=k_means_breakfast.labels_

    #To determine the optimum number of clusters, check the wss score for a given range of k
    wss =[] 
    for i in range(1,11):
        KM_Breakfast = KMeans(n_clusters=i)
        KM_Breakfast.fit(breakfast_scaled_data1)
        wss.append(KM_Breakfast.inertia_)
    # st.write(wss)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(1,11), wss, marker = '*')
    st.pyplot(fig)

    #Checking for n-clusters=3
    k_means_three_breakfast = KMeans(n_clusters = 3)
    k_means_three_breakfast.fit(breakfast_scaled_data2)
    print('WSS for K=3:', k_means_three_breakfast.inertia_)
    labels_three = k_means_three_breakfast.labels_
    #Calculating silhouette_score for k=3
    st.write(silhouette_score(breakfast_scaled_data2, labels_three))

    # Overview data in clusters
    length = len(BreakfastFoodItemIDData) + 2
    BreakfastFoodItem_Test['KMCluster'] = brklbl
    clust_profile=BreakfastFoodItem_Test.iloc[:,[2,3,4,9,10]].astype(float).groupby(BreakfastFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=BreakfastFoodItem_Test.KMCluster.value_counts().sort_index()
    clust = pd.DataFrame(clust_profile)
    st.dataframe(clust)

    return brklbl

def lunch_cluster_food(LunchFoodItemIDdata, LunchFoodItem_Test):
    ####### K-MEANS FOR LUNCH FOOD
    LunchDatacalorie=LunchFoodItemIDdata[0:,2:len(LunchFoodItemIDdata)]
    LunchDatacalorie=LunchDatacalorie[:, :-1]
    L = StandardScaler()
    lunch_scaled_data = L.fit_transform(LunchDatacalorie)

    k_means_lunch = KMeans(n_clusters=4, random_state=0)
    k_means_lunch.fit(lunch_scaled_data)
    lnchlbl=k_means_lunch.labels_

    wss =[] 
    for i in range(1,11):
        KM_Lunch = KMeans(n_clusters=i)
        KM_Lunch.fit(lunch_scaled_data)
        wss.append(KM_Lunch.inertia_)
    # st.write(wss)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(1,11), wss, marker = '*')
    st.pyplot(fig)

    k_means_three_lunch = KMeans(n_clusters = 4)
    k_means_three_lunch.fit(lunch_scaled_data)
    print('WSS for K=3:', k_means_three_lunch.inertia_)
    labels_three = k_means_three_lunch.labels_
    st.write(silhouette_score(lunch_scaled_data, labels_three))

    length = len(LunchFoodItemIDdata) + 2
    LunchFoodItem_Test['KMCluster'] = lnchlbl
    clust_profile=LunchFoodItem_Test.iloc[:,[2,3,4,9,10]].astype(float).groupby(LunchFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=LunchFoodItem_Test.KMCluster.value_counts().sort_index()
    clust = pd.DataFrame(clust_profile)
    st.dataframe(clust)

    return lnchlbl

def dinner_cluster_food(DinnerFoodItemIDdata, DinnerFoodItem_Test):
    ####### K-MEANS FOR DINNER FOOD
    DinnerDatacalorie=DinnerFoodItemIDdata[0:,2:len(DinnerFoodItemIDdata)] #nutrion data
    DinnerDatacalorie=DinnerDatacalorie[:, :-1]
    D = StandardScaler()
    dinner_scaled_data = D.fit_transform(DinnerDatacalorie)

    k_means_dinner = KMeans(n_clusters=3, random_state=0)
    k_means_dinner.fit(dinner_scaled_data)
    dnrlbl=k_means_dinner.labels_

    wss =[] 
    for i in range(1,11):
        KM_Dinner = KMeans(n_clusters=i)
        KM_Dinner.fit(dinner_scaled_data)
        wss.append(KM_Dinner.inertia_)
    # st.write(wss)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(1,11), wss, marker = '*')
    st.pyplot(fig)

    k_means_three_dinner = KMeans(n_clusters=3)
    k_means_three_dinner.fit(dinner_scaled_data)
    print('WSS for K=3:', k_means_three_dinner.inertia_)
    labels_three = k_means_three_dinner.labels_
    st.write(silhouette_score(dinner_scaled_data, labels_three))

    length = len(DinnerFoodItemIDdata) + 2
    DinnerFoodItem_Test['KMCluster'] = dnrlbl
    clust_profile=DinnerFoodItem_Test.iloc[:,[2,3,4,9,10]].astype(float).groupby(DinnerFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=DinnerFoodItem_Test.KMCluster.value_counts().sort_index()
    clust = pd.DataFrame(clust_profile)
    st.dataframe(clust)

    return dnrlbl

def cluster_food(FoodItemIDData, FoodItem_Test):
    ###### K-MEANS FOR ALL FOOD
    
    #Importing the standard scaler module and applying it on continuous variables
    BreakfastDatacalorie=FoodItemIDData[0:,2:len(FoodItemIDData)] #nutrion data
    S = StandardScaler()
    breakfast_scaled_data = S.fit_transform(BreakfastDatacalorie)
    # print(breakfast_scaled_data)

    # First, test Kmeans with clusters=3
    k_means_breakfast = KMeans(n_clusters=3, random_state=0)
    k_means_breakfast.fit(breakfast_scaled_data)
    labels=k_means_breakfast.labels_
    # print(brklbl)

    #To determine the optimum number of clusters, check the wss score for a given range of k
    wss =[] 
    for i in range(1,11):
        KM_Breakfast = KMeans(n_clusters=i)
        KM_Breakfast.fit(breakfast_scaled_data)
        wss.append(KM_Breakfast.inertia_)
    print(wss)
    plt.plot(range(1,11), wss, marker = '*')
    plt.show()

    #Checking for n-clusters=3
    k_means_three_breakfast = KMeans(n_clusters = 3)
    k_means_three_breakfast.fit(breakfast_scaled_data)
    print('WSS for K=3:', k_means_three_breakfast.inertia_)
    labels_three = k_means_three_breakfast.labels_
    # print(labels_three)
    #Calculating silhouette_score for k=3
    print(silhouette_score(breakfast_scaled_data, labels_three))

    # Overview data in clusters
    length = len(FoodItemIDData) + 2
    FoodItem_Test['KMCluster'] = labels
    clust_profile=FoodItem_Test.iloc[:,2:length].astype(float).groupby(FoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=FoodItem_Test.KMCluster.value_counts().sort_index()
    print(clust_profile)

    return labels

def Weight_Loss_Plan():
    print_user_input()

    TDEE,total_calo,total_protein,total_carb,total_fat = calc_TDEE()

    BreakfastFoodItemIDData, LunchFoodItemIDdata, DinnerFoodItemIDdata = meal_food_data()

    BreakfastNutrition = BreakfastFoodItemIDData
    LunchNutrition = LunchFoodItemIDdata
    DinnerNutrition = DinnerFoodItemIDdata

    # BreakfastFoodItem_Test = BreakfastFoodItemIDData
    # LunchFoodItem_Test = LunchFoodItemIDdata
    # DinnerFoodItem_Test = DinnerFoodItemIDdata

    BreakfastFoodItemIDData=BreakfastFoodItemIDData.to_numpy()
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.to_numpy()
    LunchFoodItemIDdata=LunchFoodItemIDdata.to_numpy()
  
    brklbl = breakfast_cluster_food(BreakfastFoodItemIDData, BreakfastNutrition)

    # st.write("--------------------------------------------------------------------")

    lnchlbl = lunch_cluster_food(LunchFoodItemIDdata, LunchNutrition)

    # st.write("--------------------------------------------------------------------")

    dnrlbl = dinner_cluster_food(DinnerFoodItemIDdata, DinnerNutrition)
    
    # st.write("--------------------------------------------------------------------")

    ## CREATE TRAIN SET FOR WEIGHT LOSS
    # if meal_time=='Breakfast':
        # Breakfast
        # print(BreakfastNutrition)

    labels = np.array(BreakfastNutrition['KMCluster'])
    features= BreakfastNutrition.drop(['KMCluster','Image','Food_items','VegNovVeg','Iron', 'Calcium', 'Sodium', 'Potassium','VitaminD','Sugars'], axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

    # #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(train_features, train_labels)

    y_pred=clf.predict(test_features)

    print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
    # print(y_pred)

    rows_list = []
    # st.subheader('SUGGESTED FOOD ITEMS FOR WEIGHT LOSS (BREAKFAST)')
    for idx, row in BreakfastNutrition.iterrows():
        if row['KMCluster']==1:
            # row = row.drop(['KMCluster'])
            # print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'],row['Fibre'])
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre','KMCluster']]
            rows_list.append(row)
            # print(row.to_frame().T)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Select',value = '')
    df.insert(loc = 1,column = 'Volume (g)',value = '100')

    df.append(df, ignore_index = True, sort = False)

    array_test = df.to_numpy()
    
    breakfast_df = df
    # st.dataframe(df)

    # abc=clf.predict([[435,9.70,9.50,55.10,0]])
    # print(abc)

    lenn = len(rows_list)

    # Get numerical feature importances
    importances = list(clf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # Generate HTML from template.
    template = jinja2.Template(f"""<!DOCTYPE html>
        <html>

        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width">
            <title>Demo</title>
            <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
            <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>

                <style>
                h2 {{
                    font-family: "Source Sans Pro", sans-serif;
                    font-weight: 400;
                    color: rgb(49, 51, 63);
                    letter-spacing: -0.005em;
                    padding: 0.5rem 0px 1rem;
                    margin: 0px;
                    line-height: 1;
                    font-size: 15px;
                }}

                .alert {{
                    padding: 15px;
                    margin-bottom: 10px;
                    border: 1px solid transparent;
                    border-radius: 4px;
                }}

                .alert-success {{
                    background-color: #dff0d8;
                    border-color: #d6e9c6;
                    color: #3c763d;
                }}

                .alert-info {{
                    background-color: #d9edf7;
                    border-color: #bce8f1;
                    color: #31708f;
                }}
                .alert-warning {{
                    background-color: #fcf8e3;
                    border-color: #faebcc;
                    color: #8a6d3b;
                }}

                .table {{
                    width: 100%;
                    max-width: 100%;
                    margin-bottom: 1rem;

                    text-overflow: ellipsis;
                    white-space: nowrap;
                    display: table-cell;

                    font-family: "Source Sans Pro", sans-serif;
                    font-size: 14px;
                    color: rgb(49, 51, 63);
                }}

                .table th,
                .table td {{
                padding: 0.75rem;
                vertical-align: top;
                border-top: 1px solid #eceeef;
                data-sortable: false;

                }}

                .table thead tr th {{
                vertical-align: bottom;
                border-bottom: 2px solid #eceeef;
                text-align: center;
                color: rgba(49, 51, 63, 0.6);
                font-family: "Source Sans Pro", sans-serif;
                font-weight: 400;
                vertical-align: middle;
                }}

                    .table thead tr th {{
                vertical-align: bottom;
                border-bottom: 2px solid #eceeef;
                text-align: center;
                color: rgba(49, 51, 63, 0.6);
                font-family: "Source Sans Pro", sans-serif;
                font-weight: 400;
                vertical-align: middle;
                }}

                .dataTable-sorter::before,
                .dataTable-sorter::after {{
                    display: none;
                    
                }}

                .dataTable-sorter {{
                    pointer-events: none;
                    cursor: default;
                }}

                .table tbody + tbody {{
                border-top: 2px solid #eceeef;
                }}
                
                .table-striped tbody tr:nth-of-type(odd) {{
                background-color: rgba(0, 0, 0, 0.05);
                }}

                    .table_wrapper{{
                    display: block;
                    overflow-x: auto;
                    white-space: nowrap;
                }}
                .table {{
                    font-family: arial, sans-serif;
                    border-collapse: collapse;
                    width: 100%;
                    overflow-x: auto;
                    border: 1px solid black;
                    table-layout: fixed;
                    overflow: scroll;
                    overflow-y:scroll;
                    height: 400px;
                    display:block;
                }}
                td {{
                    border: 1px solid #dddddd;
                    text-align: center;
                    padding: 8px;
                    white-space: nowrap;
                    width: 100px;
                }}
                th {{
                    border: 1px solid #dddddd;
                    text-align: center;
                    padding: 8px;
                    white-space: nowrap;
                    width: 100px;
                }}
                div {{
                    overflow: auto;
                }}
            </style>
        </head>
        
            <div>
            <h2 class ="alert alert-info">Total calories is <strong><span id="calories"></span>/{total_calo}</strong> calories</h2>
            <h2 class ="alert alert-info">Total fats is <strong><span id="fats"></span>/{total_fat}</strong> g</h2>
            <h2 class ="alert alert-info">Total proteins is <strong><span id="proteins"></span>/{total_protein}</strong> g</h2>
            <h2 class ="alert alert-info">Total carbohydrates is <strong><span id="carbohydrates"></span>/{total_carb}</strong> g</h2>
            </div>
            
            <body>

                {{{{ dataframe }}}}
    
            </body>

            <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

            <script defer type="text/javascript">
                let myTable = new simpleDatatables.DataTable("#myTable", {{paging:false}});
            
                    var $rows = $('#myTable tr');
                    console.log($rows.length)
                    for (var i = 0; i < $rows.length; i++) {{
                        var checkbox = document.createElement("INPUT"); //Added for checkbox
                        checkbox.name = "case[]"
                        checkbox.type = "checkbox"; //Added for checkbox
                        
                        if(i == 0) {{
                            var br = document.createElement("br");
                            $rows[i].cells[1].appendChild(br);
                        }}
                        $rows[i].cells[1].appendChild(checkbox); //Added for checkbox
                        $rows[i].cells[2].contentEditable = "true";
                    }}
                    $('td[contenteditable]').addClass('volumn_editable');
            </script>

        


            <script defer type="text/javascript">
                function calc_new() {{
                    var valuess = new Array();
                    $.each($("input[name='case[]']:checked"), function() {{
                        var datas = $(this).parents('tr:eq(0)');
                        console.log(datas);
                        valuess.push({{ 'Volumn':$(datas).find('td:eq(1)').text(), 'Food_items':$(datas).find('td:eq(2)').text() , 'Calories':$(datas).find('td:eq(3)').text(),
                                        'Fats':$(datas).find('td:eq(4)').text(), 'Proteins':$(datas).find('td:eq(5)').text(),
                                        'Carbohydrates':$(datas).find('td:eq(6)').text(), 'Fibre':$(datas).find('td:eq(7)').text(),
                                        }});               
                    
                                    
                        console.log(valuess);
                        var total_calories = 0;
                        var total_fats = 0;
                        var total_proteins = 0;
                        var total_carbs = 0;
                
                        for(var i = 0; i < valuess.length; i++) {{
                            total_calories = total_calories + parseFloat(valuess[i]['Calories']);
                            total_fats = total_fats + parseFloat(valuess[i]['Fats']);
                            total_proteins = total_proteins + parseFloat(valuess[i]['Proteins']);
                            total_carbs = total_carbs + parseFloat(valuess[i]['Carbohydrates']);
                        }}

                        document.getElementById("calories").innerHTML = total_calories.toFixed(1).toString();
                        document.getElementById("fats").innerHTML = total_fats.toFixed(1).toString();
                        document.getElementById("proteins").innerHTML = total_proteins.toFixed(1).toString();
                        document.getElementById("carbohydrates").innerHTML = total_carbs.toFixed(1).toString();
                    }});
                }}
                $("input[name='case[]']").click(function(){{
                    calc_new();
                    var numberOfChecked = $("input[name='case[]']:checked").length;

                    if (numberOfChecked == 0) {{
                        document.getElementById("calories").innerHTML = '0';
                        document.getElementById("fats").innerHTML = '0';
                        document.getElementById("proteins").innerHTML = '0';
                        document.getElementById("carbohydrates").innerHTML = '0';
                    }}
                }});
            </script>

            <script defer type="text/javascript">
                var first_load = true;
                var ratio_old = 0;
                var calo_fixed = 0;
                var fats_fixed = 0;
                var proteins_fixed = 0;
                var carbohydrates_fixed = 0;
                var fibre_fixed = 0;

                var ratio = 0; 
                var calories = 0; 
                var fats = 0; 
                var proteins = 0; 
                var carbohydrates = 0; 
                var fibre = 0; 

                var new_ratio = 0;

                $("td[contenteditable]").on("focus", function() {{
                    var values = new Array();

                    var data = $(event.target).closest('tr');
                    
                    values.push({{ 'Volumn':$(data).find('td:eq(1)').text(), 'Food_items':$(data).find('td:eq(2)').text() , 'Calories':$(data).find('td:eq(3)').text(),
                                        'Fats':$(data).find('td:eq(4)').text(), 'Proteins':$(data).find('td:eq(5)').text(),
                                        'Carbohydrates':$(data).find('td:eq(6)').text(), 'Fibre':$(data).find('td:eq(7)').text(),
                                        }});    

                    ratio_old = parseFloat(values[0]['Volumn']);
                    console.log(ratio_old)
                                                
                }});
                
                $("td[contenteditable]").on("blur", function() {{
                    var values = new Array();

                    var data = $(event.target).closest('tr');
                    
                    values.push({{ 'Volumn':$(data).find('td:eq(1)').text(), 'Food_items':$(data).find('td:eq(2)').text() , 'Calories':$(data).find('td:eq(3)').text(),
                                        'Fats':$(data).find('td:eq(4)').text(), 'Proteins':$(data).find('td:eq(5)').text(),
                                        'Carbohydrates':$(data).find('td:eq(6)').text(), 'Fibre':$(data).find('td:eq(7)').text(),
                                        }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);
                        fibre_fixed = (parseFloat(values[0]['Fibre']) * ratio);

                        console.log(ratio);
                        console.log(new_ratio);
                    
                    
                        $(data).find('td:eq(3)').text(calo_fixed.toFixed(1));
                        $(data).find('td:eq(4)').text(fats_fixed.toFixed(1));
                        $(data).find('td:eq(5)').text(proteins_fixed.toFixed(1));
                        $(data).find('td:eq(6)').text(carbohydrates_fixed.toFixed(1));
                        $(data).find('td:eq(7)').text(fibre_fixed.toFixed(1));
                        console.log(proteins_fixed)
                        calc_new();
                    

                }});

            </script>
        </html>"""
                                )

    # output_html = template.render(dataframe=df.to_html(classes='table table-striped', header="true", table_id="myTable"))

    # components.html(output_html,720,1000)  # JavaScript works

    ###-------------------------------------------------------------------------------   
    # if meal_time=='Lunch':
        # Lunch
        # print(LunchNutrition)

    labels = np.array(LunchNutrition['KMCluster'])
    features= LunchNutrition.drop(['KMCluster','Image','Food_items','VegNovVeg','Iron', 'Calcium', 'Sodium', 'Potassium','VitaminD','Sugars'], axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

    # #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(train_features, train_labels)

    y_pred=clf.predict(test_features)

    print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
    # print(y_pred)
    
    rows_list = []
    # st.subheader('SUGGESTED FOOD ITEMS FOR WEIGHT LOSS (LUNCH)')
    for idx, row in LunchNutrition.iterrows():
        if row['KMCluster']==1:
            # print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'],row['Fibre'])
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre','KMCluster']]
            rows_list.append(row)

    # Get numerical feature importances
    importances = list(clf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    
    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Select',value = '')
    df.insert(loc = 1,column = 'Volume (g)',value = '100')

    df.append(df, ignore_index = True, sort = False)

    array_test = df.to_numpy()
    lunch_df = df
    # st.dataframe(df)

    # Generate HTML from template.
    template = jinja2.Template(f"""<!DOCTYPE html>
        <html>

        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width">
            <title>Demo</title>
            <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
            <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>

                <style>
                h2 {{
                    font-family: "Source Sans Pro", sans-serif;
                    font-weight: 400;
                    color: rgb(49, 51, 63);
                    letter-spacing: -0.005em;
                    padding: 0.5rem 0px 1rem;
                    margin: 0px;
                    line-height: 1;
                    font-size: 15px;
                }}

                .alert {{
                    padding: 15px;
                    margin-bottom: 10px;
                    border: 1px solid transparent;
                    border-radius: 4px;
                }}

                .alert-success {{
                    background-color: #dff0d8;
                    border-color: #d6e9c6;
                    color: #3c763d;
                }}

                .alert-info {{
                    background-color: #d9edf7;
                    border-color: #bce8f1;
                    color: #31708f;
                }}
                .alert-warning {{
                    background-color: #fcf8e3;
                    border-color: #faebcc;
                    color: #8a6d3b;
                }}

                .table {{
                    width: 100%;
                    max-width: 100%;
                    margin-bottom: 1rem;

                    text-overflow: ellipsis;
                    white-space: nowrap;
                    display: table-cell;

                    font-family: "Source Sans Pro", sans-serif;
                    font-size: 14px;
                    color: rgb(49, 51, 63);
                }}

                .table th,
                .table td {{
                padding: 0.75rem;
                vertical-align: top;
                border-top: 1px solid #eceeef;
                data-sortable: false;

                }}

                .table thead tr th {{
                vertical-align: bottom;
                border-bottom: 2px solid #eceeef;
                text-align: center;
                color: rgba(49, 51, 63, 0.6);
                font-family: "Source Sans Pro", sans-serif;
                font-weight: 400;
                vertical-align: middle;
                }}

                    .table thead tr th {{
                vertical-align: bottom;
                border-bottom: 2px solid #eceeef;
                text-align: center;
                color: rgba(49, 51, 63, 0.6);
                font-family: "Source Sans Pro", sans-serif;
                font-weight: 400;
                vertical-align: middle;
                }}

                .dataTable-sorter::before,
                .dataTable-sorter::after {{
                    display: none;
                    
                }}

                .dataTable-sorter {{
                    pointer-events: none;
                    cursor: default;
                }}

                .table tbody + tbody {{
                border-top: 2px solid #eceeef;
                }}
                
                .table-striped tbody tr:nth-of-type(odd) {{
                background-color: rgba(0, 0, 0, 0.05);
                }}

                    .table_wrapper{{
                    display: block;
                    overflow-x: auto;
                    white-space: nowrap;
                }}
                .table {{
                    font-family: arial, sans-serif;
                    border-collapse: collapse;
                    width: 100%;
                    overflow-x: auto;
                    border: 1px solid black;
                    table-layout: fixed;
                    overflow: scroll;
                    overflow-y:scroll;
                    height: 400px;
                    display:block;
                }}
                td {{
                    border: 1px solid #dddddd;
                    text-align: center;
                    padding: 8px;
                    white-space: nowrap;
                    width: 100px;
                }}
                th {{
                    border: 1px solid #dddddd;
                    text-align: center;
                    padding: 8px;
                    white-space: nowrap;
                    width: 100px;
                }}
                div {{
                    overflow: auto;
                }}
            </style>
        </head>
        
            <div>
            <h2 class ="alert alert-info">Total calories is <strong><span id="calories"></span>/{total_calo}</strong> calories</h2>
            <h2 class ="alert alert-info">Total fats is <strong><span id="fats"></span>/{total_fat}</strong> g</h2>
            <h2 class ="alert alert-info">Total proteins is <strong><span id="proteins"></span>/{total_protein}</strong> g</h2>
            <h2 class ="alert alert-info">Total carbohydrates is <strong><span id="carbohydrates"></span>/{total_carb}</strong> g</h2>
            </div>
            
            <body>

                {{{{ dataframe }}}}
                {{{{ dataframe1 }}}}
    
            </body>

            <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

            <script defer type="text/javascript">
                let myTable = new simpleDatatables.DataTable("#myTable", {{paging:false}});
            
                    var $rows = $('#myTable tr');
                    console.log($rows.length)
                    for (var i = 0; i < $rows.length; i++) {{
                        var checkbox = document.createElement("INPUT"); //Added for checkbox
                        checkbox.name = "case[]"
                        checkbox.type = "checkbox"; //Added for checkbox
                        
                        if(i == 0) {{
                            var br = document.createElement("br");
                            $rows[i].cells[1].appendChild(br);
                        }}
                        $rows[i].cells[1].appendChild(checkbox); //Added for checkbox
                        $rows[i].cells[2].contentEditable = "true";
                    }}
                    $('td[contenteditable]').addClass('volumn_editable');
            </script>

        


            <script defer type="text/javascript">
                function calc_new() {{
                    var valuess = new Array();
                    $.each($("input[name='case[]']:checked"), function() {{
                        var datas = $(this).parents('tr:eq(0)');
                        console.log(datas);
                        valuess.push({{ 'Volumn':$(datas).find('td:eq(1)').text(), 'Food_items':$(datas).find('td:eq(2)').text() , 'Calories':$(datas).find('td:eq(3)').text(),
                                        'Fats':$(datas).find('td:eq(4)').text(), 'Proteins':$(datas).find('td:eq(5)').text(),
                                        'Carbohydrates':$(datas).find('td:eq(6)').text(), 'Fibre':$(datas).find('td:eq(7)').text(),
                                        }});               
                    
                                    
                        console.log(valuess);
                        var total_calories = 0;
                        var total_fats = 0;
                        var total_proteins = 0;
                        var total_carbs = 0;
                
                        for(var i = 0; i < valuess.length; i++) {{
                            total_calories = total_calories + parseFloat(valuess[i]['Calories']);
                            total_fats = total_fats + parseFloat(valuess[i]['Fats']);
                            total_proteins = total_proteins + parseFloat(valuess[i]['Proteins']);
                            total_carbs = total_carbs + parseFloat(valuess[i]['Carbohydrates']);
                        }}

                        document.getElementById("calories").innerHTML = total_calories.toFixed(1).toString();
                        document.getElementById("fats").innerHTML = total_fats.toFixed(1).toString();
                        document.getElementById("proteins").innerHTML = total_proteins.toFixed(1).toString();
                        document.getElementById("carbohydrates").innerHTML = total_carbs.toFixed(1).toString();
                    }});
                }}
                $("input[name='case[]']").click(function(){{
                    calc_new();
                    var numberOfChecked = $("input[name='case[]']:checked").length;

                    if (numberOfChecked == 0) {{
                        document.getElementById("calories").innerHTML = '0';
                        document.getElementById("fats").innerHTML = '0';
                        document.getElementById("proteins").innerHTML = '0';
                        document.getElementById("carbohydrates").innerHTML = '0';
                    }}
                }});
            </script>

            <script defer type="text/javascript">
                var first_load = true;
                var ratio_old = 0;
                var calo_fixed = 0;
                var fats_fixed = 0;
                var proteins_fixed = 0;
                var carbohydrates_fixed = 0;
                var fibre_fixed = 0;

                var ratio = 0; 
                var calories = 0; 
                var fats = 0; 
                var proteins = 0; 
                var carbohydrates = 0; 
                var fibre = 0; 

                var new_ratio = 0;

                $("td[contenteditable]").on("focus", function() {{
                    var values = new Array();

                    var data = $(event.target).closest('tr');
                    
                    values.push({{ 'Volumn':$(data).find('td:eq(1)').text(), 'Food_items':$(data).find('td:eq(2)').text() , 'Calories':$(data).find('td:eq(3)').text(),
                                        'Fats':$(data).find('td:eq(4)').text(), 'Proteins':$(data).find('td:eq(5)').text(),
                                        'Carbohydrates':$(data).find('td:eq(6)').text(), 'Fibre':$(data).find('td:eq(7)').text(),
                                        }});    

                    ratio_old = parseFloat(values[0]['Volumn']);
                    console.log(ratio_old)
                                                
                }});
                
                $("td[contenteditable]").on("blur", function() {{
                    var values = new Array();

                    var data = $(event.target).closest('tr');
                    
                    values.push({{ 'Volumn':$(data).find('td:eq(1)').text(), 'Food_items':$(data).find('td:eq(2)').text() , 'Calories':$(data).find('td:eq(3)').text(),
                                        'Fats':$(data).find('td:eq(4)').text(), 'Proteins':$(data).find('td:eq(5)').text(),
                                        'Carbohydrates':$(data).find('td:eq(6)').text(), 'Fibre':$(data).find('td:eq(7)').text(),
                                        }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);
                        fibre_fixed = (parseFloat(values[0]['Fibre']) * ratio);

                        console.log(ratio);
                        console.log(new_ratio);
                    
                    
                        $(data).find('td:eq(3)').text(calo_fixed.toFixed(1));
                        $(data).find('td:eq(4)').text(fats_fixed.toFixed(1));
                        $(data).find('td:eq(5)').text(proteins_fixed.toFixed(1));
                        $(data).find('td:eq(6)').text(carbohydrates_fixed.toFixed(1));
                        $(data).find('td:eq(7)').text(fibre_fixed.toFixed(1));
                        console.log(proteins_fixed)
                        calc_new();
                    

                }});

            </script>
        </html>"""
                                )

    # output_html = template.render(dataframe=df.to_html(classes='table table-striped', header="true", table_id="myTable"), dataframe1=breakfast_df.to_html(classes='table table-striped', header="true", table_id="myTable"))

    # components.html(output_html,720,1000)  # JavaScript works

    ###------------------------------------------------------------------------------- 
    # if meal_time=='Dinner':
        # Dinner
        # print(DinnerNutrition)

    labels = np.array(DinnerNutrition['KMCluster'])
    features= DinnerNutrition.drop(['KMCluster','Image','Food_items','VegNovVeg','Iron', 'Calcium', 'Sodium', 'Potassium','VitaminD','Sugars'], axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

    # #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(train_features, train_labels)

    y_pred=clf.predict(test_features)

    print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
    # print(y_pred)

    rows_list = []
    st.subheader('SUGGESTED FOOD ITEMS FOR WEIGHT LOSS')
    for idx, row in DinnerNutrition.iterrows():
        if row['KMCluster']==1:
            # print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'],row['Fibre'])
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre','KMCluster']]
            rows_list.append(row)

    # Get numerical feature importances
    importances = list(clf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Select',value = '')
    df.insert(loc = 1,column = 'Volume (g)',value = '100')

    df.append(df, ignore_index = True, sort = False)

    dinner_df = df
    array_test = df.to_numpy()
    
    # st.dataframe(df)

    # Generate HTML from template.
    template = jinja2.Template(f"""<!DOCTYPE html>
        <html>

            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width">
                <title>Demo</title>
                <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
                <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>

                <style>
                    h2 {{
                        font-family: "Source Sans Pro", sans-serif;
                        font-weight: 400;
                        color: rgb(49, 51, 63);
                        letter-spacing: -0.005em;
                        padding: 0.5rem 0px 1rem;
                        margin: 0px;
                        line-height: 1;
                        font-size: 15px;
                    }}

                    h3 {{
                        font-family: "Source Sans Pro", sans-serif;
                        font-weight: 600;
                        color: rgb(49, 51, 63);
                        letter-spacing: -0.005em;
                        padding: 0.5rem 0px 1rem;
                        margin: 0px;
                        line-height: 1;
                        font-size: 20px;
                        text-align: center;
                    }}

                    .alert {{
                        padding: 15px;
                        margin-bottom: 10px;
                        border: 1px solid transparent;
                        border-radius: 4px;
                    }}

                    .alert-success {{
                        background-color: #dff0d8;
                        border-color: #d6e9c6;
                        color: #3c763d;
                    }}

                    .alert-info {{
                        background-color: #d9edf7;
                        border-color: #bce8f1;
                        color: #31708f;
                    }}
                    .alert-warning {{
                        background-color: #fcf8e3;
                        border-color: #faebcc;
                        color: #8a6d3b;
                    }}

                    .table {{
                        width: 100%;
                        max-width: 100%;
                        margin-bottom: 1rem;

                        text-overflow: ellipsis;
                        white-space: nowrap;
                        display: table-cell;

                        font-family: "Source Sans Pro", sans-serif;
                        font-size: 14px;
                        color: rgb(49, 51, 63);
                    }}

                    .table th,
                    .table td {{
                    padding: 0.75rem;
                    vertical-align: top;
                    border-top: 1px solid #eceeef;
                    data-sortable: false;

                    }}

                    .table thead tr th {{
                    vertical-align: bottom;
                    border-bottom: 2px solid #eceeef;
                    text-align: center;
                    color: rgba(49, 51, 63, 0.6);
                    font-family: "Source Sans Pro", sans-serif;
                    font-weight: 400;
                    vertical-align: middle;
                    }}

                        .table thead tr th {{
                    vertical-align: bottom;
                    border-bottom: 2px solid #eceeef;
                    text-align: center;
                    color: rgba(49, 51, 63, 0.6);
                    font-family: "Source Sans Pro", sans-serif;
                    font-weight: 400;
                    vertical-align: middle;
                    }}

                    .dataTable-sorter::before,
                    .dataTable-sorter::after {{
                        display: none;
                        
                    }}

                    .dataTable-sorter {{
                        pointer-events: none;
                        cursor: default;
                    }}

                    .table tbody + tbody {{
                    border-top: 2px solid #eceeef;
                    }}
                    
                    .table-striped tbody tr:nth-of-type(odd) {{
                    background-color: rgba(0, 0, 0, 0.05);
                    }}

                        .table_wrapper{{
                        display: block;
                        overflow-x: auto;
                        white-space: nowrap;
                    }}
                    .table {{
                        font-family: arial, sans-serif;
                        border-collapse: collapse;
                        width: 100%;
                        overflow-x: auto;
                        border: 1px solid black;
                        table-layout: fixed;
                        overflow: scroll;
                        overflow-y:scroll;
                        height: 400px;
                        display:block;
                    }}
                    td {{
                        border: 1px solid #dddddd;
                        text-align: center;
                        padding: 8px;
                        white-space: nowrap;
                        width: 100px;
                    }}
                    th {{
                        border: 1px solid #dddddd;
                        text-align: center;
                        padding: 8px;
                        white-space: nowrap;
                        width: 100px;
                    }}
                    div {{
                        overflow: auto;
                    }}
                </style>
            </head>
        
            <div>
                <h2 class ="alert alert-info">Total calories is <strong><span id="calories"></span>/{total_calo}</strong> calories</h2>
                <h2 class ="alert alert-info">Total fats is <strong><span id="fats"></span>/{total_fat}</strong> g</h2>
                <h2 class ="alert alert-info">Total proteins is <strong><span id="proteins"></span>/{total_protein}</strong> g</h2>
                <h2 class ="alert alert-info">Total carbohydrates is <strong><span id="carbohydrates"></span>/{total_carb}</strong> g</h2>
            </div>
            
            <body>
                <h3>BREAKFAST</h3>
                {{{{ breakfast_dataframe }}}}
                <h3>LUNCH</h3>
                {{{{ lunch_dataframe }}}}
                 <h3>DINNER</h3>
                {{{{ dinner_dataframe }}}}
    
            </body>
            <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

            <script defer type="text/javascript">
                let myTable = new simpleDatatables.DataTable("#myTable", {{paging:false}});
            
                    var $rows = $('#myTable tr');
                    console.log($rows.length)
                    for (var i = 0; i < $rows.length; i++) {{
                        var checkbox = document.createElement("INPUT"); //Added for checkbox
                        checkbox.name = "case[]"
                        checkbox.type = "checkbox"; //Added for checkbox
                        
                        if(i == 0) {{
                            var br = document.createElement("br");
                            $rows[i].cells[1].appendChild(br);
                        }}
                        $rows[i].cells[1].appendChild(checkbox); //Added for checkbox
                        $rows[i].cells[2].contentEditable = "true";
                    }}
                    $('td[contenteditable]').addClass('volumn_editable');
            </script>

            <script defer type="text/javascript">
                function calc_new() {{
                    var valuess = new Array();
                    $.each($("input[name='case[]']:checked"), function() {{
                        var datas = $(this).parents('tr:eq(0)');
                        console.log(datas);
                        valuess.push({{ 'Volumn':$(datas).find('td:eq(1)').text(), 'Food_items':$(datas).find('td:eq(3)').text() , 'Calories':$(datas).find('td:eq(4)').text(),
                                        'Fats':$(datas).find('td:eq(5)').text(), 'Proteins':$(datas).find('td:eq(6)').text(),
                                        'Carbohydrates':$(datas).find('td:eq(7)').text(), 'Fibre':$(datas).find('td:eq(8)').text(),
                                        }});               
                    
                                    
                        console.log(valuess);
                        var total_calories = 0;
                        var total_fats = 0;
                        var total_proteins = 0;
                        var total_carbs = 0;
                
                        for(var i = 0; i < valuess.length; i++) {{
                            total_calories = total_calories + parseFloat(valuess[i]['Calories']);
                            total_fats = total_fats + parseFloat(valuess[i]['Fats']);
                            total_proteins = total_proteins + parseFloat(valuess[i]['Proteins']);
                            total_carbs = total_carbs + parseFloat(valuess[i]['Carbohydrates']);
                        }}

                        document.getElementById("calories").innerHTML = total_calories.toFixed(1).toString();
                        document.getElementById("fats").innerHTML = total_fats.toFixed(1).toString();
                        document.getElementById("proteins").innerHTML = total_proteins.toFixed(1).toString();
                        document.getElementById("carbohydrates").innerHTML = total_carbs.toFixed(1).toString();
                    }});
                }}
                $("input[name='case[]']").click(function(){{
                    calc_new();
                    var numberOfChecked = $("input[name='case[]']:checked").length;

                    if (numberOfChecked == 0) {{
                        document.getElementById("calories").innerHTML = '0';
                        document.getElementById("fats").innerHTML = '0';
                        document.getElementById("proteins").innerHTML = '0';
                        document.getElementById("carbohydrates").innerHTML = '0';
                    }}
                }});
            </script>

            <script defer type="text/javascript">
                var first_load = true;
                var ratio_old = 0;
                var calo_fixed = 0;
                var fats_fixed = 0;
                var proteins_fixed = 0;
                var carbohydrates_fixed = 0;
                var fibre_fixed = 0;

                var ratio = 0; 
                var calories = 0; 
                var fats = 0; 
                var proteins = 0; 
                var carbohydrates = 0; 
                var fibre = 0; 

                var new_ratio = 0;

                $("td[contenteditable]").on("focus", function() {{
                    var values = new Array();

                    var data = $(event.target).closest('tr');
                    
                    values.push({{ 'Volumn':$(data).find('td:eq(1)').text(), 'Food_items':$(data).find('td:eq(3)').text() , 'Calories':$(data).find('td:eq(4)').text(),
                                        'Fats':$(data).find('td:eq(5)').text(), 'Proteins':$(data).find('td:eq(6)').text(),
                                        'Carbohydrates':$(data).find('td:eq(7)').text(), 'Fibre':$(data).find('td:eq(8)').text(),
                                        }});    

                    ratio_old = parseFloat(values[0]['Volumn']);
                    console.log(ratio_old)
                                                
                }});
                
                $("td[contenteditable]").on("blur", function() {{
                    var values = new Array();

                    var data = $(event.target).closest('tr');
                    
                    values.push({{ 'Volumn':$(data).find('td:eq(1)').text(), 'Food_items':$(data).find('td:eq(3)').text() , 'Calories':$(data).find('td:eq(4)').text(),
                                        'Fats':$(data).find('td:eq(5)').text(), 'Proteins':$(data).find('td:eq(6)').text(),
                                        'Carbohydrates':$(data).find('td:eq(7)').text(), 'Fibre':$(data).find('td:eq(8)').text(),
                                        }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);
                        fibre_fixed = (parseFloat(values[0]['Fibre']) * ratio);

                        console.log(ratio);
                        console.log(new_ratio);
                    
                    
                        $(data).find('td:eq(4)').text(calo_fixed.toFixed(1));
                        $(data).find('td:eq(5)').text(fats_fixed.toFixed(1));
                        $(data).find('td:eq(6)').text(proteins_fixed.toFixed(1));
                        $(data).find('td:eq(7)').text(carbohydrates_fixed.toFixed(1));
                        $(data).find('td:eq(8)').text(fibre_fixed.toFixed(1));
                        console.log(proteins_fixed)
                        calc_new();
                    

                }});

            </script>
        </html>"""
                                )

    output_html = template.render(lunch_dataframe=lunch_df.to_html(classes='table table-striped', header="true", table_id="myTable", escape=False ,formatters=dict(Image=path_to_image_html)),
                breakfast_dataframe=breakfast_df.to_html(classes='table table-striped', header="true", table_id="myTable", escape=False ,formatters=dict(Image=path_to_image_html)),
                dinner_dataframe=dinner_df.to_html(classes='table table-striped', header="true", table_id="myTable", escape=False ,formatters=dict(Image=path_to_image_html)))

    components.html(output_html,720,1700)  # JavaScript works

def Weight_Gain_Plan():
    print_user_input()

    BMI = calc_BMI()
    TDEE = calc_TDEE()

    BreakfastFoodItemIDData, LunchFoodItemIDdata, DinnerFoodItemIDdata = meal_food_data()

    BreakfastNutrition = BreakfastFoodItemIDData
    LunchNutrition = LunchFoodItemIDdata
    DinnerNutrition = DinnerFoodItemIDdata

    BreakfastFoodItemIDData=BreakfastFoodItemIDData.to_numpy()
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.to_numpy()
    LunchFoodItemIDdata=LunchFoodItemIDdata.to_numpy()

    brklbl = breakfast_cluster_food(BreakfastFoodItemIDData, BreakfastNutrition)
    print("--------------------------------------------------------------------")

    lnchlbl = lunch_cluster_food(LunchFoodItemIDdata, LunchNutrition)
    print("--------------------------------------------------------------------")

    dnrlbl = dinner_cluster_food(DinnerFoodItemIDdata, DinnerNutrition)
    print("--------------------------------------------------------------------")

    ## CREATE TRAIN SET FOR WEIGHT GAIN
    if meal_time=='Breakfast':
        # Breakfast
        # print(BreakfastNutrition)
        labels = np.array(BreakfastNutrition['KMCluster'])
        features= BreakfastNutrition.drop(['KMCluster','Food_items','VegNovVeg','Sodium','Potassium','Fibre'], axis = 1)
        feature_list = list(features.columns)
        features = np.array(features)

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        # #Create a Gaussian Classifier
        clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

        # #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(train_features, train_labels)

        y_pred=clf.predict(test_features)

        print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
        # print(y_pred)

        print ('SUGGESTED FOOD ITEMS FOR WEIGHT GAIN (BREAKFAST)')
        for idx, row in BreakfastNutrition.iterrows():
            if row['KMCluster']==1:
                print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'])

        # abc=clf.predict([[435,9.70,9.50,55.10,0]])
        # print(abc)

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    if meal_time=='Lunch':
        # Lunch
        # print(LunchNutrition)

        labels = np.array(LunchNutrition['KMCluster'])
        features= LunchNutrition.drop(['KMCluster','Food_items','VegNovVeg','Sodium','Potassium','Fibre'], axis = 1)
        feature_list = list(features.columns)
        features = np.array(features)

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        # #Create a Gaussian Classifier
        clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

        # #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(train_features, train_labels)

        y_pred=clf.predict(test_features)

        print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
        # print(y_pred)

        print ('SUGGESTED FOOD ITEMS FOR WEIGHT GAIN (LUNCH)')
        for idx, row in LunchNutrition.iterrows():
            if row['KMCluster']==2:
                print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'])

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    if meal_time=='Dinner':
        # Dinner
        # print(DinnerNutrition)

        labels = np.array(DinnerNutrition['KMCluster'])
        features= DinnerNutrition.drop(['KMCluster','Food_items','VegNovVeg','Sodium','Potassium','Fibre'], axis = 1)
        feature_list = list(features.columns)
        features = np.array(features)

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        # #Create a Gaussian Classifier
        clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

        # #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(train_features, train_labels)

        y_pred=clf.predict(test_features)

        print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
        # print(y_pred)

        print ('SUGGESTED FOOD ITEMS FOR WEIGHT GAIN (DINNER)')
        for idx, row in DinnerNutrition.iterrows():
            if row['KMCluster']==0 or row['KMCluster']==1:
                print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'])

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

def Maintenance_Plan():
    print_user_input()

    BMI = calc_BMI()
    TDEE = calc_TDEE()

    BreakfastFoodItemIDData, LunchFoodItemIDdata, DinnerFoodItemIDdata = meal_food_data()

    BreakfastNutrition = BreakfastFoodItemIDData
    LunchNutrition = LunchFoodItemIDdata
    DinnerNutrition = DinnerFoodItemIDdata

    BreakfastFoodItemIDData=BreakfastFoodItemIDData.to_numpy()
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.to_numpy()
    LunchFoodItemIDdata=LunchFoodItemIDdata.to_numpy()

    brklbl = breakfast_cluster_food(BreakfastFoodItemIDData, BreakfastNutrition)
    print("--------------------------------------------------------------------")

    lnchlbl = lunch_cluster_food(LunchFoodItemIDdata, LunchNutrition)
    print("--------------------------------------------------------------------")

    dnrlbl = dinner_cluster_food(DinnerFoodItemIDdata, DinnerNutrition)
    print("--------------------------------------------------------------------")

    ## CREATE TRAIN SET FOR MAINTENANCE
    if meal_time=='Breakfast':
        # Breakfast
        # print(BreakfastNutrition)

        labels = np.array(BreakfastNutrition['KMCluster'])
        features= BreakfastNutrition.drop(['KMCluster','Food_items','VegNovVeg','Sodium','Fibre','Sugars'], axis = 1)
        feature_list = list(features.columns)
        features = np.array(features)

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        # #Create a Gaussian Classifier
        clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

        # #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(train_features, train_labels)

        y_pred=clf.predict(test_features)

        print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
        # print(y_pred)

        print ('SUGGESTED FOOD ITEMS FOR MAINTENANCE (BREAKFAST)')
        for idx, row in BreakfastNutrition.iterrows():
            if row['KMCluster']==2:
                print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'])

        # abc=clf.predict([[435,9.70,9.50,55.10,0]])
        # print(abc)

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    if meal_time=='Lunch':
        # Lunch
        # print(LunchNutrition)

        labels = np.array(LunchNutrition['KMCluster'])
        features= LunchNutrition.drop(['KMCluster','Food_items','VegNovVeg','Sodium','Fibre','Sugars'], axis = 1)
        feature_list = list(features.columns)
        features = np.array(features)

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        # #Create a Gaussian Classifier
        clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

        # #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(train_features, train_labels)

        y_pred=clf.predict(test_features)

        print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
        # print(y_pred)

        print ('SUGGESTED FOOD ITEMS FOR MAINTENANCE (LUNCH)')
        for idx, row in LunchNutrition.iterrows():
            if row['KMCluster']==0 or row['KMCluster']==1:
                print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'])

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    if meal_time=='Dinner':
        # Dinner
        # print(DinnerNutrition)

        labels = np.array(DinnerNutrition['KMCluster'])
        features= DinnerNutrition.drop(['KMCluster','Food_items','VegNovVeg','Sodium','Fibre','Sugars'], axis = 1)
        feature_list = list(features.columns)
        features = np.array(features)

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        # #Create a Gaussian Classifier
        clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

        # #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(train_features, train_labels)

        y_pred=clf.predict(test_features)

        print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
        # print(y_pred)

        print ('SUGGESTED FOOD ITEMS FOR MAINTENANCE (DINNER)')
        for idx, row in DinnerNutrition.iterrows():
            if row['KMCluster']==1 or row['KMCluster']==2:
                print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'])

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

def Predict():
    # print_prediction_input()

    FoodItemIDData = food_data()

    FoodNutrion = FoodItemIDData

    FoodItemIDData=FoodItemIDData.to_numpy()
  
    foodlbs = cluster_food(FoodItemIDData, FoodNutrion)

    labels = np.array(FoodNutrion['KMCluster'])
    features= FoodNutrion.drop(['KMCluster','Food_items','VegNovVeg','Iron', 'Calcium', 'Sodium', 'Potassium','Fibre','VitaminD','Sugars'], axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators = 100, random_state = 42)

    # #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(train_features, train_labels)

    y_pred=clf.predict(test_features)

    st.write("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
    print(y_pred)

    y_pred=clf.predict([[float(food_calories),float(food_fat), float(food_protein), float(food_carb)]])

    print('PREDICTION RESULT :: ')
    if y_pred==0:
        st.info('FIT FOR WEIGHT GAIN PLAN')
    if y_pred==1 or y_pred==2:
        st.info('FIT FOR WEIGHT LOSS or MAINTENANCE PLAN')
    
    st.balloons()

    # Get numerical feature importances
    importances = list(clf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

 

st.set_page_config(layout="centered")
    
header = st.container()
user_input = st.container()
table_result = st.container()


st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
        
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .css-1d391kg {
        padding-left: 52px;
        padding-right: 52px;
        # padding-top: 43px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .css-177yq5e ul {
        margin-bottom: 0px;
        margin-left: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with header:
    st.sidebar.title('Eat Better Daily')

with user_input:
    choice = st.sidebar.radio("You want",['Food recommendation', "Predict food for diet plan"])
    
    if choice == 'Food recommendation':
        # sel_col, disp_col = st.columns([6,1])
        st.sidebar.subheader("How old are you?")
        age = st.sidebar.text_input('Enter your age!', '20')

        st.sidebar.subheader("What is your gender?")
        gender = st.sidebar.selectbox('Choose your gender!', options=['Male','Female'])

        st.sidebar.subheader("How much do you weigh?")
        weight = st.sidebar.text_input('Enter your weight in kg!', '60')

        st.sidebar.subheader("How tall are you?")
        height = st.sidebar.text_input('Enter your height in cm!', '170')

        st.sidebar.subheader("What is your activity level?")

        st.sidebar.markdown('* **Sedentary** (little or no exercise, desk job)')
        st.sidebar.markdown('* **Lightly active** (light exercise/sports 1-3 days/week)')
        st.sidebar.markdown('* **Moderately active** (moderate exercise 6-7 days)')
        st.sidebar.markdown('* **Very active** (hard exercise every day, or 2 xs/day)')
        st.sidebar.markdown('* **Extra active** (hard exercise 2 or more times per day)')

        activity_level = st.sidebar.select_slider('Choose your activity level!', options=[
            'Sedentary',
            'Lightly active',
            'Moderately active',
            'Very active',
            'Extra active'])

        st.sidebar.subheader("What is your favorite meal?")
        meal_time = st.sidebar.selectbox('Choose your desired meal time!', options=[
            'Breakfast',
            'Lunch',
            'Dinner',])

        st.sidebar.subheader("What is your diet plan?")
        diet_plan = st.sidebar.radio(
        "Choose your diet plan!",
        ('Weight Loss', 'Weight Gain', 'Maintenance'))

        st.sidebar.subheader("Are you ready?")
        if diet_plan == 'Weight Loss':
            button = st.sidebar.button('Do it now!', on_click=Weight_Loss_Plan)
        elif diet_plan == 'Weight Gain':
            button = st.sidebar.button('Do it now!', on_click=Weight_Gain_Plan)
        elif diet_plan == 'Maintenance':
            button = st.sidebar.button('Do it now!', on_click=Maintenance_Plan)

    if choice == 'Predict food for diet plan':
        st.sidebar.subheader("What is the name of the food?")
        food_name = st.sidebar.text_input("Enter the food's name!'", 'Banana')

        st.sidebar.subheader("Enter calories in 100g of food!")
        food_calories = st.sidebar.text_input("Enter the amount of calories in the food!'", '89')

        st.sidebar.subheader("Enter the amount of fat in 100g of food!")
        food_fat = st.sidebar.text_input("Enter the grams of fat!", '0.3')

        st.sidebar.subheader("Enter the amount of protein in 100g of food!")
        food_protein = st.sidebar.text_input("Enter the grams of protein!", '1.1')

        st.sidebar.subheader("Enter the amount of carbohydrate in 100g of food!")
        food_carb = st.sidebar.text_input("Enter the grams of carbohydrate!", '23')

        st.sidebar.button('Do it now!', on_click=Predict)

