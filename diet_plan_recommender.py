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

result_df = pd.DataFrame()

def show():
    print(result_df)
    # create plot
    cds = ColumnDataSource(result_df)
    columns = [
    TableColumn(field="Food_items"),
    TableColumn(field="Calories"),
    TableColumn(field="Fats"),
    TableColumn(field="Proteins"),
    TableColumn(field="Carbohydrates"),
    TableColumn(field="Fibre"),
    TableColumn(field="KMCluster"),
    ]

    # define events
    cds.selected.js_on_change(
    "indices",
    CustomJS(
            args=dict(source=cds),
            code="""
            document.dispatchEvent(
            new CustomEvent("INDEX_SELECT", {detail: {data: source.selected.indices}})
            )
            """
    )
    )
    p = DataTable(source=cds, columns=columns)
    result = streamlit_bokeh_events(bokeh_plot=p, events="INDEX_SELECT", key="foo", refresh_on_update=False, debounce_time=0, override_height=100)
    if result:
            if result.get("INDEX_SELECT"):
                    st.write(result_df.iloc[result.get("INDEX_SELECT")["data"]])      

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

    st.write("Your body mass index is: ", BMI)
    if ( BMI < 16):
        st.write("Your body condition is **Severely Underweight**")
    elif ( BMI >= 16 and BMI < 18.5):
        st.write("Your body condition is **Underweight**")
    elif ( BMI >= 18.5 and BMI < 25):
        st.write("Your body condition is **Healthy**")
    elif ( BMI >= 25 and BMI < 30):
        st.write("**overweight**")
    elif ( BMI >=30):
        st.write("Your body condition is Severely Overweight")

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

    TDEE = BMR * Activity_Level1
    st.write("Your Basal metabolic rate is: ", BMR, 'calories')
    st.write("**Your Total Daily Energy Expenditure is: ", TDEE , 'calories**')

    return TDEE

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

    return BreakfastFoodItemIDData, LunchFoodItemIDdata, DinnerFoodItemIDdata

def breakfast_cluster_food(BreakfastFoodItemIDData, BreakfastFoodItem_Test):
    ###### K-MEANS FOR BREAKFAST FOOD
    
    #Importing the standard scaler module and applying it on continuous variables
    BreakfastDatacalorie=BreakfastFoodItemIDData[0:,2:len(BreakfastFoodItemIDData)] #nutrion data
    # BreakfastDatacalorie=BreakfastDatacalorie[:, [0,1,2,7,8]]
    S = StandardScaler()
    breakfast_scaled_data = S.fit_transform(BreakfastDatacalorie)

    # First, test Kmeans with clusters=3
    k_means_breakfast = KMeans(n_clusters=3, random_state=0)
    k_means_breakfast.fit(breakfast_scaled_data)
    brklbl=k_means_breakfast.labels_
    # print(brklbl)

    #To determine the optimum number of clusters, check the wss score for a given range of k
    wss =[] 
    for i in range(1,11):
        KM_Breakfast = KMeans(n_clusters=i)
        KM_Breakfast.fit(breakfast_scaled_data)
        wss.append(KM_Breakfast.inertia_)
    st.write(wss)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(1,11), wss, marker = '*')
    # plt.show()
    st.pyplot(fig)

    #Checking for n-clusters=3
    k_means_three_breakfast = KMeans(n_clusters = 3)
    k_means_three_breakfast.fit(breakfast_scaled_data)
    print('WSS for K=3:', k_means_three_breakfast.inertia_)
    labels_three = k_means_three_breakfast.labels_
    # print(labels_three)
    #Calculating silhouette_score for k=3
    st.write(silhouette_score(breakfast_scaled_data, labels_three))

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
    L = StandardScaler()
    lunch_scaled_data = L.fit_transform(LunchDatacalorie)
    # print(lunch_scaled_data)

    k_means_lunch = KMeans(n_clusters=3, random_state=0)
    k_means_lunch.fit(lunch_scaled_data)
    lnchlbl=k_means_lunch.labels_
    # print(k_means_lunch.labels_)

    wss =[] 
    for i in range(1,11):
        KM_Lunch = KMeans(n_clusters=i)
        KM_Lunch.fit(lunch_scaled_data)
        wss.append(KM_Lunch.inertia_)
    st.write(wss)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(1,11), wss, marker = '*')
    st.pyplot(fig)

    k_means_three_lunch = KMeans(n_clusters = 3)
    k_means_three_lunch.fit(lunch_scaled_data)
    print('WSS for K=3:', k_means_three_lunch.inertia_)
    labels_three = k_means_three_lunch.labels_
    # print(labels_three)
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
    D = StandardScaler()
    dinner_scaled_data = D.fit_transform(DinnerDatacalorie)
    # print(dinner_scaled_data)

    k_means_dinner = KMeans(n_clusters=3, random_state=0)
    k_means_dinner.fit(dinner_scaled_data)
    dnrlbl=k_means_dinner.labels_
    # print(k_means_dinner.labels_)

    wss =[] 
    for i in range(1,11):
        KM_Dinner = KMeans(n_clusters=i)
        KM_Dinner.fit(dinner_scaled_data)
        wss.append(KM_Dinner.inertia_)
    st.write(wss)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(1,11), wss, marker = '*')
    st.pyplot(fig)

    k_means_three_dinner = KMeans(n_clusters=3)
    k_means_three_dinner.fit(dinner_scaled_data)
    print('WSS for K=3:', k_means_three_dinner.inertia_)
    labels_three = k_means_three_dinner.labels_
    # print(labels_three)
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
    global result_df

    BMI = calc_BMI()
    TDEE = calc_TDEE()

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

    st.write("--------------------------------------------------------------------")

    lnchlbl = lunch_cluster_food(LunchFoodItemIDdata, LunchNutrition)

    st.write("--------------------------------------------------------------------")

    dnrlbl = dinner_cluster_food(DinnerFoodItemIDdata, DinnerNutrition)
    
    st.write("--------------------------------------------------------------------")

    ## CREATE TRAIN SET FOR WEIGHT LOSS
    if meal_time=='Breakfast':
        # Breakfast
        # print(BreakfastNutrition)

        labels = np.array(BreakfastNutrition['KMCluster'])
        features= BreakfastNutrition.drop(['KMCluster','Food_items','VegNovVeg','Iron', 'Calcium', 'Sodium', 'Potassium','VitaminD','Sugars'], axis = 1)
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
        st.subheader('SUGGESTED FOOD ITEMS FOR WEIGHT LOSS (BREAKFAST)')
        for idx, row in BreakfastNutrition.iterrows():
            if row['KMCluster']==0:
                # row = row.drop(['KMCluster'])
                # print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'],row['Fibre'])
                row = row[['Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre','KMCluster']]
                rows_list.append(row)
                # print(row.to_frame().T)

        df = pd.DataFrame(rows_list)
        df.insert(loc = 0,column = 'Select',value = '')
        df.insert(loc = 1,column = 'Volume (g)',value = '100')

        array_test = df.to_numpy()
        
        result_df = df
        st.dataframe(df)

        # abc=clf.predict([[435,9.70,9.50,55.10,0]])
        # print(abc)

        html = df.to_html(classes=["table-bordered", "table-striped", "table-hover"])

        lenn = len(rows_list)

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
        
        i = 0

        html_string = f'''
        <center><h1>GeeksforGeeks</h1></center>
        <ul id="myList"></ul>
        
        <script language="javascript">

            var list = document.getElementById("myList");
            
            for({i}=0; i<{lenn}; {i}++) {{
                var li = document.createElement("li");
                li.innerHTML = {array_test[i]};
                console.log(item);
                list.appendChild(li);
            }}
        </script>
       '''

        # Generate HTML from template.
        template = jinja2.Template("""<!DOCTYPE html>
            <html>

            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width">
                <title>Demo</title>
                <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
                <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>
                </head>


                <h2>Total calories is <span id="calories">calories</span> g</h2>
                <h2>Total fats is <span id="fats">fats</span> g</h2>
                <h2>Total proteins is <span id="proteins">proteins</span> g</h2>
                <h2>Total carbohydrates is <span id="carbohydrates">carbohydrates</span> g</h2>

                <body>

                    {{ dataframe }}
        
                </body>


                <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

                <script defer type="text/javascript">
                    let myTable = new simpleDatatables.DataTable("#myTable", {paging:false});
              
                   
                        var $rows = $('#myTable tr');
                        console.log($rows.length)
                        for (var i = 0; i < $rows.length; i++) {
                            var checkbox = document.createElement("INPUT"); //Added for checkbox
                            checkbox.name = "case[]"
                            checkbox.type = "checkbox"; //Added for checkbox
                            $rows[i].cells[1].appendChild(checkbox); //Added for checkbox
                            $rows[i].cells[2].contentEditable = "true";
                        }
                </script>

                <script defer type="text/javascript">
                    $("input[name='case[]']").click(function(){
                        var values = new Array();
                        $.each($("input[name='case[]']:checked"), function() {
                            var data = $(this).parents('tr:eq(0)');
                            values.push({ 'Volumn':$(data).find('td:eq(1)').text(), 'Food_items':$(data).find('td:eq(2)').text() , 'Calories':$(data).find('td:eq(3)').text(),
                                            'Fats':$(data).find('td:eq(4)').text(), 'Proteins':$(data).find('td:eq(5)').text(),
                                            'Carbohydrates':$(data).find('td:eq(6)').text(), 'Fibre':$(data).find('td:eq(7)').text(),
                                            });               
                        
                                        
                            console.log(values);
                            var total_calories = 0;
                            var total_fats = 0;
                            var total_proteins = 0;
                            var total_carbs = 0;
                    
                            for(var i = 0; i < values.length; i++) {
                                total_calories = total_calories + parseFloat(values[i]['Calories']);
                                total_fats = total_fats + parseFloat(values[i]['Fats']);
                                total_proteins = total_proteins + parseFloat(values[i]['Proteins']);
                                total_carbs = total_carbs + parseFloat(values[i]['Carbohydrates']);
                            }

                            document.getElementById("calories").innerHTML = total_calories.toString();
                            document.getElementById("fats").innerHTML = total_fats.toString();
                            document.getElementById("proteins").innerHTML = total_proteins.toString();
                            document.getElementById("carbohydrates").innerHTML = total_carbs.toString();
                        });
                    });
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
                    
                   $("td[contenteditable]").on("input", function() {
                        var values = new Array();
                        
                        var data = $(event.target).closest('tr');
                        values.push({ 'Volumn':$(data).find('td:eq(1)').text(), 'Food_items':$(data).find('td:eq(2)').text() , 'Calories':$(data).find('td:eq(3)').text(),
                                            'Fats':$(data).find('td:eq(4)').text(), 'Proteins':$(data).find('td:eq(5)').text(),
                                            'Carbohydrates':$(data).find('td:eq(6)').text(), 'Fibre':$(data).find('td:eq(7)').text(),
                                            });     

                        
                        console.log(first_load);
                        
                        if(first_load==true) {
                            ratio_old = parseFloat(values[0]['Volumn']) / 100;
                            calo_fixed = parseFloat(values[0]['Calories']);
                            fats_fixed = parseFloat(values[0]['Fats']);
                            proteins_fixed = parseFloat(values[0]['Proteins']);
                            carbohydrates_fixed = parseFloat(values[0]['Carbohydrates']);
                            fibre_fixed = parseFloat(values[0]['Fibre']);

                            ratio = parseFloat(values[0]['Volumn']) / 100;
                            calories = calo_fixed * ratio;
                            fats = fats_fixed * ratio;
                            proteins = proteins_fixed * ratio;
                            carbohydrates = carbohydrates_fixed * ratio;
                            fibre = fibre_fixed * ratio;
                        }
                        else {
                            ratio_old1= ratio_old;
                            calo_fixed1= calo_fixed;
                            fats_fixed1= fats_fixed;
                            proteins_fixed1= proteins_fixed;
                            carbohydrates_fixed1= carbohydrates_fixed;
                            fibre_fixed1= fibre_fixed;

                            ratio = parseFloat(values[0]['Volumn']) / 100;
                            calories = calo_fixed1 * ratio;
                            fats = fats_fixed1 * ratio;
                            proteins = proteins_fixed1 * ratio;
                            carbohydrates = carbohydrates_fixed1 * ratio;
                            fibre = fibre_fixed1 * ratio;
                        }
                        
                        $("td[contenteditable]").on("blur", function() {
                            $(data).find('td:eq(3)').text(calories.toFixed(1));
                            $(data).find('td:eq(4)').text(fats.toFixed(1));
                            $(data).find('td:eq(5)').text(proteins.toFixed(1));
                            $(data).find('td:eq(6)').text(carbohydrates.toFixed(1));
                            $(data).find('td:eq(7)').text(fibre.toFixed(1));
                        });

                        first_load = false;
                        console.log(first_load);

                    });

                </script>
            </html>"""
                                    )

        output_html = template.render(dataframe=df.to_html(table_id="myTable"))

        components.html(output_html,800,1300)  # JavaScript works

        

        # components.html(html,700,1300)   

        # st.markdown(df.to_html(classes='table table-striped'), unsafe_allow_html=True)  # JavaScript doesn't work

        
    if meal_time=='Lunch':
        # Lunch
        # print(LunchNutrition)

        labels = np.array(LunchNutrition['KMCluster'])
        features= LunchNutrition.drop(['KMCluster','Food_items','VegNovVeg','Iron', 'Calcium', 'Sodium', 'Potassium','VitaminD','Sugars'], axis = 1)
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
        st.subheader('SUGGESTED FOOD ITEMS FOR WEIGHT LOSS (LUNCH)')
        for idx, row in LunchNutrition.iterrows():
            if row['KMCluster']==1:
                # print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'],row['Fibre'])
                row = row[['Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre','KMCluster']]
                rows_list.append(row)

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

        df = pd.DataFrame(rows_list)

        array_test = df.to_numpy()
        
        result_df = df
        st.dataframe(df)

    if meal_time=='Dinner':
        # Dinner
        # print(DinnerNutrition)

        labels = np.array(DinnerNutrition['KMCluster'])
        features= DinnerNutrition.drop(['KMCluster','Food_items','VegNovVeg','Iron', 'Calcium', 'Sodium', 'Potassium','VitaminD','Sugars'], axis = 1)
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
        st.subheader('SUGGESTED FOOD ITEMS FOR WEIGHT LOSS (DINNER)')
        for idx, row in DinnerNutrition.iterrows():
            if row['KMCluster']==1:
                # print(row['Food_items'],row['Calories'],row['Fats'],row['Proteins'],row['Carbohydrates'],row['Fibre'])
                row = row[['Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre','KMCluster']]
                rows_list.append(row)

        # Get numerical feature importances
        importances = list(clf.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        # Print out the feature and importances 
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

        df = pd.DataFrame(rows_list)

        array_test = df.to_numpy()
        
        result_df = df
        st.dataframe(df)

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
    print_prediction_input()

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

    print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))
    print(y_pred)

    y_pred=clf.predict([[float(e8.get()),float(e9.get()), float(e10.get()), float(e11.get())]])

    # print("Accuracy:",metrics.accuracy_score(test_labels, y_pred))

    print('PREDICTION RESULT :: ')
    if y_pred==0:
        print('FIT FOR WEIGHT GAIN PLAN')
    if y_pred==1 or y_pred==2:
        print('FIT FOR WEIGHT LOSS or MAINTENANCE PLAN')

    # Get numerical feature importances
    importances = list(clf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

 

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
    st.sidebar.title('Diet Food Recommendation System')

with user_input:
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

  

    # if button == 1:
    #     selected_indices = st.multiselect('Select rows:', result.index)   
    #     selected_rows = result.loc[selected_indices]
    #     print('### Selected Rows', selected_rows)

    # placeholder_c = st.empty()
    # userNumCopper = placeholder_c.number_input('Enter number of Copper: ', min_value= 0)

# with table_result:
#     if button == 1:
#         # show()
#         print(result_df)
#         # create plot
#         cds = ColumnDataSource(result_df)
#         columns = [
#         TableColumn(field="Food_items"),
#         TableColumn(field="Calories"),
#         TableColumn(field="Fats"),
#         TableColumn(field="Proteins"),
#         TableColumn(field="Carbohydrates"),
#         TableColumn(field="Fibre"),
#         TableColumn(field="KMCluster"),
#         ]

#         # define events
#         cds.selected.js_on_change(
#         "indices",
#         CustomJS(
#                 args=dict(source=cds),
#                 code="""
#                 document.dispatchEvent(
#                 new CustomEvent("INDEX_SELECT", {detail: {data: source.selected.indices}})
#                 )
#                 """
#         )
#         )
#         p = DataTable(source=cds, columns=columns)
#         result = streamlit_bokeh_events(bokeh_plot=p, events="INDEX_SELECT", key="foo", refresh_on_update=False, debounce_time=0, override_height=100)
#         if result:
#                 if result.get("INDEX_SELECT"):
#                         st.write(result_df.iloc[result.get("INDEX_SELECT")["data"]])     
    
# html_string = '''
# <h1>HTML string in RED</h1>

# <script language="javascript">
#   document.querySelector("h1").style.color = "red";
#   console.log("Streamlit runs JavaScript");
#   alert("Streamlit runs JavaScript");
# </script>
# '''

# components.html(html_string)  # JavaScript works

# st.markdown(html_string, unsafe_allow_html=True)  # JavaScript doesn't work


# if __name__ == '__main__':
    # main_win = Tk()
    
    # Label(main_win,text="Age").grid(row=0,column=0,sticky=W,pady=4)
    # Label(main_win,text="Gender (1/0)").grid(row=1,column=0,sticky=W,pady=4)
    # Label(main_win,text="Weight (in kg)").grid(row=2,column=0,sticky=W,pady=4)
    # Label(main_win,text="Height (in cm)").grid(row=3,column=0,sticky=W,pady=4)
    # Label(main_win,text="Activity level").grid(row=4,column=0,sticky=W,pady=4)
    # Label(main_win,text="Veg/Non-veg (1/0)").grid(row=5,column=0,sticky=W,pady=4)
    # Label(main_win,text="Breakfast/Lunch/Dinner (1/2/3)").grid(row=6,column=0,sticky=W,pady=4)
    # Label(main_win,text="Calories").grid(row=7,column=0,sticky=W,pady=4)
    # Label(main_win,text="Fat").grid(row=8,column=0,sticky=W,pady=4)
    # Label(main_win,text="Protein").grid(row=9,column=0,sticky=W,pady=4)
    # Label(main_win,text="Carb").grid(row=10,column=0,sticky=W,pady=4)

    # e1 = Entry(main_win)
    # e2 = Entry(main_win)
    # e3 = Entry(main_win)
    # e4 = Entry(main_win)
    # e5 = Entry(main_win)
    # e6 = Entry(main_win)
    # e7 = Entry(main_win)
    # e8 = Entry(main_win)
    # e9 = Entry(main_win)
    # e10 = Entry(main_win)
    # e11 = Entry(main_win)

    # e1.grid(row=0, column=1)
    # e2.grid(row=1, column=1)
    # e3.grid(row=2, column=1)
    # e4.grid(row=3, column=1)
    # e5.grid(row=4, column=1)
    # e6.grid(row=5, column=1)
    # e7.grid(row=6, column=1)
    # e8.grid(row=7, column=1)
    # e9.grid(row=8, column=1)
    # e10.grid(row=9, column=1)
    # e11.grid(row=10, column=1)

    # # Button(main_win,text='Quit',command=main_win.quit).grid(row=5,column=0,sticky=W,pady=4)
    # Button(main_win,text='Maintenance',command=Maintenance_Plan).grid(row=1,column=4,sticky=W,pady=4)
    # Button(main_win,text='Weight Loss',command=Weight_Loss_Plan).grid(row=3,column=4,sticky=W,pady=4)
    # Button(main_win,text='Weight Gain',command=Weight_Gain_Plan).grid(row=2,column=4,sticky=W,pady=4)
    # Button(main_win,text='Predict',command=Predict).grid(row=4,column=4,sticky=W,pady=4)

    # main_win.geometry("550x360")
    # main_win.wm_title("DIET RECOMMENDATION SYSTEM")

    # main_win.mainloop()
