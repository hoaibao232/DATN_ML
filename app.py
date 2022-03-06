from contextlib import nullcontext
from typing import TypedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydot
import streamlit as st
import streamlit.components.v1 as components
import jinja2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score,classification_report
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
import os
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import pickle
import glob
import yaml
import configs
from streamlit.legacy_caching.hashing import _CodeHasher
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)

def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session

def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state

def autosave_session(state):
   # The session file is saved in the path saved in the state key called state.session_autosave_file_abs, which is declared in the beginning of the script
    with open(str(state.session_autosave_file_abs), 'wb') as outf:
        pickle.dump(state._state['data'], outf)
        
def is_shutdown_line(shutdown_line):
    return "Shutting down" in shutdown_line

def get_last_n_lines_of_file(file, n):
    with open(file, "r") as file:
        lines = file.readlines()
    return lines[-n:]

def was_session_shutdown(state):
    # If an unexpected shutdown happened and the session was restarted, the debug log 24th or 25th lines will have a shutdown message (at least in my case)
    list_of_files = glob.glob(
        configs['APP_BASE_DIR'] + '/logs/streamlit_logs/*')
    state.streamlit_log = max(list_of_files, key=os.path.getmtime)
    last_25_streamlit_log_lines = get_last_n_lines_of_file(state.streamlit_log, 25)
    shutdown_session_line = last_25_streamlit_log_lines[0]
    shutdown_session_line_after = last_25_streamlit_log_lines[1]
    session_was_shutdown = is_shutdown_line(shutdown_session_line) or is_shutdown_line(shutdown_session_line_after)
    return session_was_shutdown

def load_autosaved_session(state, login=False):
    try:
            with open(str(state.session_autosave_file_abs), 'rb') as inf:
                state._state['data'] = pickle.load(inf)
                if not login:
                    # logout user if security policy requires
                    state.user = ''
                    state.password = ''
                    state.authenticated = False
    except FileNotFoundError:  # someone deleted the sessions file
        pass

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
    'https://res.cloudinary.com/hoaibao232/image/upload/v1644552047/Wheat_Noodles_eswiv1.jpg',
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
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639743599/Rice_Pudding_o49uef.jpg', 
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835620/Beef_sqg7lp.jpg',   
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835620/Shrimp_nlzxtj.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835620/Squid_wftq6h.png',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835620/Egg_wn1tmx.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835620/White_Egg_yyk3vo.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835619/Bread_nut_w7jgny.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835619/Bread_white_bhigwr.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835620/Scallop_id5gzm.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835618/Lobster_yxalt7.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835619/Venison_xdnuwo.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835618/Halibut_vbv1n5.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835619/Tofu_uzzanp.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835617/Chicken_ez5fea.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835620/Sardines_cooked_yh61ee.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835617/Duck_Breast_ojfawz.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835617/Pork_Chops_m84ilu.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835617/Fish_x6obrq.png',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835615/Powdered_Peanut_Butter_vg9h4z.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835616/Turkey_breast_gu61fm.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835615/Carrots_ljgysk.png',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835615/Snail_w1qlxg.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835617/Mango_hv0r22.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835616/Cabbage_green_eusuna.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835614/Soy_boiled_yyys7j.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835614/Coca-Cola_Zero_calorie_bbx1ye.jpg',
    'https://res.cloudinary.com/hoaibao232/image/upload/v1639835614/Sliced_bell_peppers_yufu4f.webp'      
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
    if diet_plan == 'Weight Loss':
        TDEE = BMR * Activity_Level1
        calorie_deficit = 0.15 * TDEE
        total_calo = float("{:.0f}".format(TDEE - calorie_deficit)) 
        total_protein = float("{:.0f}".format((total_calo * 0.4)/4))
        total_carb = float("{:.0f}".format((total_calo * 0.3)/4)) 
        total_fat = float("{:.0f}".format((total_calo - total_protein*4 - total_carb*4)/9))

        labels = 'Protein', 'Carbohydrate', 'Fat'
        sizes = [40, 30, 30]
        explode = (0.1,0, 0)
    if diet_plan == 'Weight Gain':
        TDEE = BMR * Activity_Level1
        calorie_surplus = 0.15 * TDEE
        total_calo = float("{:.0f}".format(TDEE + calorie_surplus)) 
        total_protein = float("{:.0f}".format((total_calo * 0.3)/4))
        total_carb = float("{:.0f}".format((total_calo * 0.5)/4)) 
        total_fat = float("{:.0f}".format((total_calo - total_protein*4 - total_carb*4)/9))

        labels = 'Protein', 'Carbohydrate', 'Fat'
        sizes = [30, 50, 20]
        explode = (0,0.1, 0)
    if diet_plan == 'Maintenance':
        TDEE = BMR * Activity_Level1
        total_calo = float("{:.0f}".format(TDEE)) 
        total_protein = float("{:.0f}".format((total_calo * 0.3)/4))
        total_carb = float("{:.0f}".format((total_calo * 0.4)/4)) 
        total_fat = float("{:.0f}".format((total_calo - total_protein*4 - total_carb*4)/9))

        labels = 'Protein', 'Carbohydrate', 'Fat'
        sizes = [30, 40, 30]
        explode = (0,0.1, 0)

    my_expander = st.expander(label='HEALTH CHECK!')
    with my_expander:
        str_bmi = "Your body mass index is **{}**".format(BMI)
        str_bmr = "Your Basal metabolic rate is **{:.0f} calories**".format(BMR)
        str_tdee = "Your Total Daily Energy Expenditure is **{:.0f} calories**".format(TDEE)
        str_calories = "Our Recommend Total Daily Intake Calories is **{:.0f} calories**".format(total_calo)
        str_protein = "Protein intake should be **{:.0f} gam per day**".format(total_protein)
        str_fat = "Fat intake should be **{:.0f} gam per day**".format(total_fat)
        str_carb = "Carbohydrate intake should be **{:.0f} gam per day**".format(total_carb)
         
        if ( BMI < 18.5):
            str_health = "Your body condition is **Underweight**"
        elif ( BMI >= 18.5 and BMI < 25):
            str_health = "Your body condition is **Healthy**"
        elif ( BMI >= 25 and BMI < 30):
            str_health = "Your body condition is **Overweight**"
        elif ( BMI >=30):
            str_health = "Your body condition is **Obese**"

        st.info(str_bmi + " - " + str_health)
        st.info(str_bmr)
        st.info(str_tdee)
        st.success(str_calories)
        st.success(str_protein)
        st.success(str_fat)
        st.success(str_carb)

        fig1, ax1 = plt.subplots(figsize=(8, 3))
        patches, texts, pcts = ax1.pie(sizes, explode=explode, colors = ['#555658','#41B6EF','#CDCED0'] ,labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.setp(pcts, color='white')
        ax1.axis('equal') 

        st.pyplot(fig1)
        
        st.balloons()

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

    BreakfastDatacalorie=BreakfastFoodItemIDData[0:,2:len(BreakfastFoodItemIDData)] #nutrion data
    BreakfastDatacalorie=BreakfastDatacalorie[:, :-1]
    BreakfastDatacalorie = BreakfastDatacalorie[:, [0,1,2,7,8]]

    S = StandardScaler()
    breakfast_scaled_data = S.fit_transform(BreakfastDatacalorie)

    k_means_breakfast = KMeans(init="k-means++", n_clusters=3, n_init=50, max_iter=500, random_state=42)
    k_means_breakfast.fit(breakfast_scaled_data)
    brklbl=k_means_breakfast.labels_

    # #To determine the optimum number of clusters, check the wss score for a given range of k
    # wss =[] 
    # for i in range(1,11):
    #     KM_Breakfast = KMeans(init="k-means++", n_clusters=i, n_init=50, max_iter=500, random_state=42)
    #     KM_Breakfast.fit(breakfast_scaled_data)
    #     wss.append(KM_Breakfast.inertia_)
    # st.write(wss)
    # fig = plt.figure(figsize = (10, 5))
    # plt.plot(range(1,11), wss, marker = '*')
    # st.pyplot(fig)

    # #Checking for n-clusters=3
    # k_means_three_breakfast = KMeans(init="k-means++", n_clusters=3, n_init=50, max_iter=500, random_state=42)
    # k_means_three_breakfast.fit(breakfast_scaled_data)
    # print('WSS for K=3:', k_means_three_breakfast.inertia_)
    # labels_three = k_means_three_breakfast.labels_
    # #Calculating silhouette_score for k=3
    # st.write('silhouette_score for k=3', silhouette_score(breakfast_scaled_data, labels_three))

    # Overview data in clusters
    length = len(BreakfastFoodItemIDData) + 2
    BreakfastFoodItem_Test['KMCluster'] = brklbl
    clust_profile=BreakfastFoodItem_Test.iloc[:,[2,3,4,9,10]].astype(float).groupby(BreakfastFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=BreakfastFoodItem_Test.KMCluster.value_counts().sort_index()
    clust = pd.DataFrame(clust_profile)
    # st.dataframe(clust)

    return brklbl

def lunch_cluster_food(LunchFoodItemIDdata, LunchFoodItem_Test):
    ####### K-MEANS FOR LUNCH FOOD
    LunchDatacalorie=LunchFoodItemIDdata[0:,2:len(LunchFoodItemIDdata)]
    LunchDatacalorie=LunchDatacalorie[:, :-1]
    LunchDatacalorie = LunchDatacalorie[:, [0,1,2,7,8]]
    L = StandardScaler()
    lunch_scaled_data = L.fit_transform(LunchDatacalorie)

    k_means_lunch = KMeans(init="k-means++", n_clusters=3, n_init=50, max_iter=500, random_state=42)
    k_means_lunch.fit(lunch_scaled_data)
    lnchlbl=k_means_lunch.labels_

    # wss =[] 
    # for i in range(1,11):
    #     KM_Lunch = KMeans(init="k-means++", n_clusters=i, n_init=50, max_iter=500, random_state=42)
    #     KM_Lunch.fit(lunch_scaled_data)
    #     wss.append(KM_Lunch.inertia_)
    # st.write(wss)
    # fig = plt.figure(figsize = (10, 5))
    # plt.plot(range(1,11), wss, marker = '*')
    # st.pyplot(fig)

    # k_means_three_lunch = KMeans(init="k-means++", n_clusters=3, n_init=50, max_iter=500, random_state=42)
    # k_means_three_lunch.fit(lunch_scaled_data)
    # print('WSS for K=3:', k_means_three_lunch.inertia_)
    # labels_three = k_means_three_lunch.labels_
    # st.write('silhouette_score for k=3', silhouette_score(lunch_scaled_data, labels_three))

    length = len(LunchFoodItemIDdata) + 2
    LunchFoodItem_Test['KMCluster'] = lnchlbl
    clust_profile=LunchFoodItem_Test.iloc[:,[2,3,4,9,10]].astype(float).groupby(LunchFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=LunchFoodItem_Test.KMCluster.value_counts().sort_index()
    clust = pd.DataFrame(clust_profile)
    # st.dataframe(clust)

    return lnchlbl

def dinner_cluster_food(DinnerFoodItemIDdata, DinnerFoodItem_Test):
    ####### K-MEANS FOR DINNER FOOD
    DinnerDatacalorie=DinnerFoodItemIDdata[0:,2:len(DinnerFoodItemIDdata)] #nutrion data
    DinnerDatacalorie=DinnerDatacalorie[:, :-1]
    DinnerDatacalorie = DinnerDatacalorie[:, [0,1,2,7,8]]
    D = StandardScaler()
    dinner_scaled_data = D.fit_transform(DinnerDatacalorie)

    k_means_dinner = KMeans(init="k-means++", n_clusters=3, n_init=50, max_iter=500, random_state=42)
    k_means_dinner.fit(dinner_scaled_data)
    dnrlbl=k_means_dinner.labels_

    # wss =[] 
    # for i in range(1,11):
    #     KM_Dinner = KMeans(init="k-means++", n_clusters=i, n_init=50, max_iter=500, random_state=42)
    #     KM_Dinner.fit(dinner_scaled_data)
    #     wss.append(KM_Dinner.inertia_)
    # st.write(wss)
    # fig = plt.figure(figsize = (10, 5))
    # plt.plot(range(1,11), wss, marker = '*')
    # st.pyplot(fig)

    # k_means_three_dinner = KMeans(init="k-means++", n_clusters=3, n_init=50, max_iter=500, random_state=42)
    # k_means_three_dinner.fit(dinner_scaled_data)
    # print('WSS for K=3:', k_means_three_dinner.inertia_)
    # labels_three = k_means_three_dinner.labels_
    # st.write('silhouette_score for k=3', silhouette_score(dinner_scaled_data, labels_three))

    length = len(DinnerFoodItemIDdata) + 2
    DinnerFoodItem_Test['KMCluster'] = dnrlbl
    clust_profile=DinnerFoodItem_Test.iloc[:,[2,3,4,9,10]].astype(float).groupby(DinnerFoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=DinnerFoodItem_Test.KMCluster.value_counts().sort_index()
    clust = pd.DataFrame(clust_profile)
    # st.dataframe(clust)

    return dnrlbl

def cluster_food(FoodItemIDData, FoodItem_Test):
    ###### K-MEANS FOR ALL FOOD
    
    MealDatacalorie=FoodItemIDData[0:,2:len(FoodItemIDData)] #nutrion data
    MealDatacalorie = MealDatacalorie[:, [0,1,2,7,8]]
    S = StandardScaler()
    foods_scaled_data = S.fit_transform(MealDatacalorie)

    k_means_meals = KMeans(init="k-means++", n_clusters=3, n_init=50, max_iter=500, random_state=42)
    k_means_meals.fit(foods_scaled_data)
    labels=k_means_meals.labels_

    FoodItem_Test['KMCluster'] = labels

    # Check Elbow plot
    wss =[] 
    for i in range(1,11):
        KM_Meals = KMeans(init="k-means++", n_clusters=i, n_init=50, max_iter=500, random_state=42)
        KM_Meals.fit(foods_scaled_data)
        wss.append(KM_Meals.inertia_)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(1,11), wss, marker = '*')
    # st.pyplot(fig)

    # # Check silhouette score
    # for i in range(2,10):
    #     k_means_three_meals = KMeans(init="k-means++", n_clusters=i, n_init=50, max_iter=500, random_state=42)
    #     k_means_three_meals.fit(foods_scaled_data)
    #     print('WSS for K=:',k_means_three_meals.inertia_)
    #     labels_three = k_means_three_meals.labels_
    #     st.write(silhouette_score(foods_scaled_data, labels_three))
    
    # Overview data in clusters
    length = len(FoodItemIDData) + 2
    FoodItem_Test['KMCluster'] = labels
    clust_profile=FoodItem_Test.iloc[:,[2,3,4,9,10]].astype(float).groupby(FoodItem_Test['KMCluster']).mean()
    clust_profile['KMFrequency']=FoodItem_Test.KMCluster.value_counts().sort_index()
    clust = pd.DataFrame(clust_profile)
    # st.dataframe(clust)

    # c_data_path = "/Users/hoaibao/DATN/DATN_ML/image"
    # L = FoodItem_Test['Food_items']
    # n_row = 6
    # n_col=6
    # for i in range(3):
    #     fig1 = plt.figure(figsize = (10, 5))
    #     fig, axs = plt.subplots(n_row, n_col, figsize=(7, 7))
    #     axs = axs.flatten()
    #     for img, ax in zip(L[ k_means_meals.labels_ == i][:36], axs):
    #         ax.imshow(mpimg.imread(os.path.join(c_data_path,img+'.jpeg')))
    #     plt.tight_layout()
    #     st.pyplot(fig)
    #     st.write('----------------------------------------------------------------------')

    return labels

def Weight_Loss_Plan():
    TDEE,total_calo,total_protein,total_carb,total_fat = calc_TDEE()

    BreakfastFoodItemIDData, LunchFoodItemIDdata, DinnerFoodItemIDdata = meal_food_data()

    BreakfastNutrition = BreakfastFoodItemIDData
    LunchNutrition = LunchFoodItemIDdata
    DinnerNutrition = DinnerFoodItemIDdata

    BreakfastFoodItemIDData=BreakfastFoodItemIDData.to_numpy()
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.to_numpy()
    LunchFoodItemIDdata=LunchFoodItemIDdata.to_numpy()
  
    brklbl = breakfast_cluster_food(BreakfastFoodItemIDData, BreakfastNutrition)

    lnchlbl = lunch_cluster_food(LunchFoodItemIDdata, LunchNutrition)

    dnrlbl = dinner_cluster_food(DinnerFoodItemIDdata, DinnerNutrition)
    
    rows_list = []
    for idx, row in BreakfastNutrition.iterrows():
        if row['KMCluster']==1:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)
    breakfast_df = df

    lenn = len(rows_list)
    
    rows_list = []
    for idx, row in LunchNutrition.iterrows():
        if row['KMCluster']==0 or row['KMCluster']==1:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)
  
    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)
    lunch_df = df

    rows_list = []
    st.subheader('CREATE MEAL PLAN FOR WEIGHT LOSS')
    for idx, row in DinnerNutrition.iterrows():
        if row['KMCluster']==1 or row['KMCluster']==2:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)

    dinner_df = df
    
    template = jinja2.Template(f"""<!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width">
                <title>Demo</title>
                <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
                <link rel="stylesheet" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.min.css">
                <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
                <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
                <script src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>

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
                        overflow-x: scroll;
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
                    .dataTable-table th a {{
                        color: rgb(49, 51, 63);
                        font-weight: 600;
                    }}

                    .table tbody + tbody {{
                    border-top: 2px solid #eceeef;
                    }}
                    
                    .table-striped tbody tr:nth-of-type(odd) {{
                    background-color: rgba(0, 0, 0, 0.05);
                    }}

                        .table_wrapper{{
                        display: block;
                        white-space: nowrap;
                    }}
                    .table {{
                        width: 100%;
                        border: 1px solid black;
                        table-layout: fixed;
                        overflow-x: hidden;
                        height: 600px;
                        display:block;
                    }}
                    td {{
                        border: 1px solid #dddddd;
                        text-align: center;
                        padding: 8px;
                        white-space: nowrap;
                        width: 100px;
                        vertical-align: middle;
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
                    .dataTable-table tbody tr td {{
                        vertical-align: middle;
                    }}
                    .table td {{
                        vertical-align: middle;
                    }}
                    .dataTable-table tbody tr th {{
                        vertical-align: middle;
                        text-align: center;
                    }}

                    # thead, tfoot {{
                    # display: none;
                    # }}
                    # table {{
                    # background: none !important;
                    # border: none !important;
                    # }}
                    # tr {{
                    # display: inline-block;
                    # padding: 1rem 0.5rem 1rem 0.5rem;
                    # margin: 1.5rem;
                    # border: 1px solid grey;
                    # border-radius 10px;
                    # box-shadow: 0 0 10px;
                    # }}
                    # td {{
                    # display: block;
                    # }}
                    td {{border: 1px #DDD solid; padding: 5px; cursor: pointer;}}

                    .selected {{
                        background-color: #0dcaf0 !important; 
                        color: #FFF !important;
                    }}

                    .btn-purple {{
                    color: #fff;
                    background-color: #6f42c1;
                    border-color: #643ab0;
                    }}

                    
                    
                    .card-img-top {{
                        width: 100%;
                        height: 7vh;
                        object-fit: cover;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                    }}
                    .modal-dialog {{
                        width: 27rem;
                        margin: 0 auto;
                        }}
                    .progress {{margin-bottom:0;}}
                    .start {{float:left;}}
                    .end {{float:right; text-align:right;}}
                    div p {{
                        display: inline-block;
                    }}
                    div input {{
                        display: inline-block;
                    }}
                    h5 {{
                        text-align: center;
                    }}
                    .image-parent {{
                        max-width: 40px;
                    }}
                    .modal-content{{
                        position: relative;
                        top: 50%;
                        transform: translateY(-50%);
                    }}

                    .dataTable-input {{
                        display: none;
                    }}

                    table.dataTable{{
                        box-sizing: border-box;
                        overflow: scroll;
                        }}
                    
                    #volumnHelp {{
                        display:none;
                    }}

                    .progress-bar{{

                    }}
                </style>
            </head>
        
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="calories-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="calories-left" style="width:100%">
                        Remaining: {total_calo}
                    </div>
                </div>
                <h6 class="start mt-1">Calories Daily Intake</h6>
            </div>
            <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="fats-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="fats-left" style="width:100%">
                        Remaining: {total_fat}g
                    </div>
                </div>
                <h6 class="start mt-1">Fat Daily Intake</div>
            </div>
             <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="protein-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="protein-left" style="width:100%">
                        Remaining: {total_protein}g
                    </div>
                </div>
                <h6 class="start mt-1">Protein Daily Intake</div>
            </div>
             <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="carb-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="carb-left" style="width:100%">
                        Remaining: {total_carb}g
                    </div>
                </div>
                <h6 class="start mt-1">Carbohydrate Daily Intake</div>
            </div>

            
            
            <body>
                <h3>BREAKFAST</h3>
                {{{{ breakfast_dataframe }}}}
                <br>
                <h3>LUNCH</h3>
                {{{{ lunch_dataframe }}}}
                <br>
                 <h3>DINNER</h3>
                {{{{ dinner_dataframe }}}}
    
            </body>

            <div class="modal fade" id="myModal" role="dialog" aria-hidden="true">
                <div class="modal-dialog">
                <div class="modal-content">
                    <div class="card">
                        <img class="food-image card-img-top" src="" alt="food image">
                        <div class="card-body">
                            <h5 class="Food-modal card-title"><span></span></h5>
                            <br>
                            <div class="row">
                                <div class="Volumn-modal col-sm-3"><p>Volumn(g): </p></div>
                                <div class="col-sm-4">
                                    <input type="number" class="form-control form-control-sm" id="volumn-input">
                                </div>
                                <div class="col-sm-5">
                                    <small id="volumnHelp" class="text-danger">
                                    Must be a number.
                                    </small>      
                                </div>
                            </div>
                            <div class="row">
                                <div class="Calories-modal"><p>Calories:  </p><span></span></div>
                            </div>
                            <div class="row">
                                <div class="col-sm-6">
                                    
                                    <div class="Protein-modal"><p>Protein(g):  </p><span></span>g</div>
                                    <div class="Fibre-modal"><p>Fibre(g):  </p><span></span>g</div>
                                </div>
                                <div class="col-sm-6">
                                    <div class="Fat-modal"><p>Fat(g):</p><span></span>g</div>
                                    <div class="Carbohydrate-modal"><p>Carbohydrate(g):  </p><span></span>g</div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col text-center">
                                    <button class="btn btn-primary" id="btn-modal-save">Select</button>
                                    <button class="btn btn-secondary" id="btn-modal-close"> Close</button>
                                </div>
                            </div>
                            
                        </div>
                    </div>
                    </div>
                </div>
            </div>

            <div class="text-right mb-1">
                <button class="btn btn-secondary mt-3" id="export">Export CSV</button>
            </div>

            <div id="accordion">
                <div class="card">
                    <div class="card-header text-white bg-secondary" id="headingOne">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
                        Breakfast Meal
                        </button>
                    </h5>
                    </div>

                    <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
                        <div class="panel-body" id="breakfast" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                               
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header bg-danger" id="headingTwo">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
                        Lunch Meal
                        </button>
                    </h5>
                    </div>
                    <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
                        <div class="panel-body" id="lunch" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                            
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header bg-dark" id="headingThree">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseThree" aria-expanded="false" aria-controls="collapseThree">
                        Dinner Meal
                        </button>
                    </h5>
                    </div>
                    <div id="collapseThree" class="collapse" aria-labelledby="headingThree" data-parent="#accordion">
                        <div class="panel-body" id="dinner" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                            
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

            <script defer type="text/javascript">
                let myTable = new simpleDatatables.DataTable("#myTable", {{paging:false}});
            </script>

            <script defer type="text/javascript">
                let myTable1 = new simpleDatatables.DataTable("#myTable1", {{paging:false}});
            </script>

            <script defer type="text/javascript">
                let myTable2 = new simpleDatatables.DataTable("#myTable2", {{paging:false}});
            </script>
            
            <script defer type="text/javascript">
                var tableIDs = 'abc';
            
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

                var tr1;

                $("#volumn-input").on("focus", function() {{
                    if($.isNumeric($('#volumn-input').val())) {{
                        var values = new Array();
                        var data = $(event.target);
                        
                        
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});    

                        ratio_old = parseFloat(values[0]['Volumn']);
                    }}                            
                }});

                $(document).on("blur", "#volumn-input", function() {{
                    var values = new Array();

                    var data = $(event.target);

                    if($.isNumeric($('#volumn-input').val())) {{
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);

                        $(".Calories-modal span").text(calo_fixed.toFixed(1));
                        $('.Fat-modal span').text(fats_fixed.toFixed(1));
                        $('.Protein-modal span').text(proteins_fixed.toFixed(1));
                        $('.Carbohydrate-modal span').text(carbohydrates_fixed.toFixed(1));
                        

                        var food_name = $('.Food-modal span').text();
                        food_name = food_name.replace(/^\s+|\s+$/gm,'')

                        var a = $('#myTable tr td:contains("' + food_name + '")').filter(function(){{
                            console.log($.trim($(this).text()));
                            if($.trim($(this).text()) == food_name)
                            return true;
                            else
                            return false;
                        }});
                        var tr = $(a).parents('tr:eq(0)');
                        tr1 = tr; 
                    }} 
                }});

                $('#btn-modal-save').on('click',{{tableIDs: tableIDs}}, myfunction) 

                function myfunction(e) {{
                    var values = new Array();
                    
                    if($.isNumeric($('#volumn-input').val())) {{
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});    

                        ratio_old = parseFloat(values[0]['Volumn']);

                        var values = new Array();
                        
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);

                        $(".Calories-modal span").text(calo_fixed.toFixed(1));
                        $('.Fat-modal span').text(fats_fixed.toFixed(1));
                        $('.Protein-modal span').text(proteins_fixed.toFixed(1));
                        $('.Carbohydrate-modal span').text(carbohydrates_fixed.toFixed(1));
                        
                        var food_name = $('.Food-modal span').text();
                        food_name = food_name.replace(/^\s+|\s+$/gm,'')

                        var a;
                        var tr;
                        if (tableIDs == 'myTable') {{
                            a = $('#myTable tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                        }}
                        else if (tableIDs == 'myTable1') {{
                            a = $('#myTable1 tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                            console.log('123')
                        }}
                        else if (tableIDs == 'myTable2') {{
                            a = $('#myTable2 tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                            console.log('456')
                        }}
                        
                        $(tr).find('td:eq(0)').text($('#volumn-input').val());
                        $(tr).find('td:eq(3)').text(calo_fixed.toFixed(1));
                        $(tr).find('td:eq(4)').text(fats_fixed.toFixed(1));
                        $(tr).find('td:eq(5)').text(proteins_fixed.toFixed(1));
                        $(tr).find('td:eq(6)').text(carbohydrates_fixed.toFixed(1));

                        $(tr).addClass("selected");
                        calc_new1();
                        show_meal();
                        $("#volumnHelp").css("display", "none");
                    }}

                    else {{
                        $("#volumnHelp").css("display", "inline-block");
                    }}
                }}
            </script>
            <script type="text/javascript" src="https://cdn.datatables.net/1.10.8/js/jquery.dataTables.min.js"></script>

            <script defer type="text/javascript">
                function calc_new1() {{
                    
                    var valuesss = new Array();
                    var rows_selection = new Array();
                    var selected_rowss = document.getElementsByClassName("selected");
                    $('table.table-striped').DataTable().rows('.selected').invalidate();
                    var selection_rows = $('table.table-striped').DataTable().rows('.selected').data()

                    var numberOfChecked = $('table.table-striped').DataTable().rows('.selected').count();
                    if (numberOfChecked == 0) {{
                            $("#calories-intake").css("width", 0 + "%").text(0);
                            $("#calories-left").css("width", 100 + "%").text(({total_calo}).toFixed(1) + " remaining");

                            $("#fats-intake").css("width", 0 + "%").text(0 + "g");
                            $("#fats-left").css("width", 100 + "%").text(({total_fat}).toFixed(1) + "g remaining");

                            $("#protein-intake").css("width", 0 + "%").text(0 + "g");
                            $("#protein-left").css("width", 100 + "%").text(({total_protein}).toFixed(1) + "g remaining");

                            $("#carb-intake").css("width", 0 + "%").text(0 + "g");
                            $("#carb-left").css("width", 100 + "%").text(({total_carb}).toFixed(1) + "g remaining");
                        }}

            
                    $.each(selection_rows, function(){{
                        console.log(this)
                        var Row=this;
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});   

                                      
                        var total_calories = 0;
                        var total_fats = 0;
                        var total_proteins = 0;
                        var total_carbs = 0;
                
                        for(var i = 0; i < valuesss.length; i++) {{
                            total_calories = total_calories + parseFloat(valuesss[i]['Calories']);
                            total_fats = total_fats + parseFloat(valuesss[i]['Fats']);
                            total_proteins = total_proteins + parseFloat(valuesss[i]['Proteins']);
                            total_carbs = total_carbs + parseFloat(valuesss[i]['Carbohydrates']);
                        }}

                       
                        var calories_ratio_percentage = (total_calories/{total_calo}).toFixed(1)*100;
                        var fat_ratio_percentage = (total_fats/{total_fat}).toFixed(1)*100;
                        var protein_ratio_percentage = (total_proteins/{total_protein}).toFixed(1)*100;
                        var carb_ratio_percentage = (total_carbs/{total_carb}).toFixed(1)*100;

                        if (total_calories > {total_calo})
                        {{
                            $('#calories-intake').addClass('bg-danger');
                            $('#calories-intake').css("width", 100 + "%").text("Excess calories: " + (total_calories - {total_calo}).toFixed(1));
                            $("#calories-left").css("width", 0 + "%").text(({total_calo} - total_calories).toFixed(1) + " remaining");
                        }}
                       
                        else {{
                            $('#calories-intake').removeClass('bg-danger');
                            $("#calories-intake").css("width", calories_ratio_percentage + "%").text(total_calories.toFixed(1));
                            $("#calories-left").css("width", 100-calories_ratio_percentage + "%").text(({total_calo} - total_calories).toFixed(1) + " remaining");
                        }}

                        if (total_fats > {total_fat})
                        {{
                            $('#fats-intake').addClass('bg-danger');
                            $('#fats-intake').css("width", 100 + "%").text("Excess fat: " + (total_fats - {total_fat}).toFixed(1) +"g");
                            $("#fats-left").css("width", 0 + "%").text(({total_fat} - total_fats).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#fats-intake').removeClass('bg-danger');
                            $("#fats-intake").css("width", fat_ratio_percentage + "%").text(total_fats.toFixed(1) + "g");
                            $("#fats-left").css("width", 100-fat_ratio_percentage + "%").text(({total_fat} - total_fats).toFixed(1) + "g remaining");
                        }}

                        if (total_proteins > {total_protein})
                        {{
                            $('#protein-intake').addClass('bg-danger');
                            $('#protein-intake').css("width", 100 + "%").text("Excess protein: " + (total_proteins - {total_protein}).toFixed(1) + "g");
                            $("#protein-left").css("width", 0 + "%").text(({total_protein} - total_proteins).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#protein-intake').removeClass('bg-danger');
                            $("#protein-intake").css("width", protein_ratio_percentage + "%").text(total_proteins.toFixed(1) + "g");
                            $("#protein-left").css("width", 100-protein_ratio_percentage + "%").text(({total_protein} - total_proteins).toFixed(1) + "g remaining");
                        }}

                        if (total_carbs > {total_carb})
                        {{
                            $('#carb-intake').addClass('bg-danger');
                            $('#carb-intake').css("width", 100 + "%").text("Excess carbohydrate: " + (total_carbs - {total_carb}).toFixed(1) + "g");
                            $("#carb-left").css("width", 0 + "%").text(({total_carb} - total_carbs).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#carb-intake').removeClass('bg-danger');
                            $("#carb-intake").css("width", carb_ratio_percentage + "%").text(total_carbs.toFixed(1) + "g");
                            $("#carb-left").css("width", 100-carb_ratio_percentage + "%").text(({total_carb} - total_carbs).toFixed(1) + "g remaining");
                        }}


                        if (numberOfChecked == 0) {{
                            document.getElementById("calories").innerHTML = '0';
                            document.getElementById("fats").innerHTML = '0';
                            document.getElementById("proteins").innerHTML = '0';
                            document.getElementById("carbohydrates").innerHTML = '0';
                        }}

                        $("#myModal").modal("hide");
                    }});
                }}
            </script>

            <script defer type="text/javascript">
                $('#btn-modal-close').on('click', function() {{
                    $("#myModal").modal("hide");
                }})
            </script>

            <!-- jQuery library -->
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>

            <!-- Latest compiled JavaScript -->
            <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
            <script src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>

            <script defer type="text/javascript">
                $("#myTable").on('click','tr:gt(0)',function(){{
                    
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());

                    $(".modal-dialog").css("height", "39vh");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");
                    
                    var tableID = $(this).closest('table').attr('id');
                    var tableIDD = tableID;
                    
                   tableIDs = tableID;
                   console.log(tableIDs)
                }});

                    
                    
            </script>

            <script defer type="text/javascript">
                $("#myTable1").on('click','tr:gt(0)',function() {{
                    console.log('2222222')
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());
                    
                    var tableID1 = $(this).closest('table').attr('id');
                    var tableIDD1 = tableID1;
                    tableIDs = tableID1;
                    
                    $(".modal-dialog").css("height", "79vh");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");


                }});

                    
                       
            </script>

            <script defer type="text/javascript">
                $("#myTable2").on('click','tr:gt(0)',function() {{
                    meal = 'dinner';
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());
                    
                    var tableID2 = $(this).closest('table').attr('id');
                    var tableIDD2 = tableID2;
                    tableIDs = tableID2;
                    
                    $(".modal-dialog").css("height", "119vh");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");

                 
                }});

                
            </script>

            <script defer type="text/javascript">
                function show_meal() {{
                    var table = document.getElementById("myTable");
                    var selected_rowss = table.getElementsByClassName("selected");
                    $("#breakfast").empty();

                    $('#myTable').DataTable().rows('.selected').every(function(element, index){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});         
                       
                        $("#breakfast").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
 
                   }})   

                    var table1 = document.getElementById("myTable1");
                    var selected_rowss1 = table1.getElementsByClassName("selected");
                    $("#lunch").empty();
                    $('#myTable1').DataTable().rows('.selected').every(function(){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});                
                    $("#lunch").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
                        
                        }})   

                    var table2 = document.getElementById("myTable2");
                    var selected_rowss2 = table2.getElementsByClassName("selected");
                    $("#dinner").empty();
                    $('#myTable2').DataTable().rows('.selected').every(function(){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});                   
                      
                        
                        $("#dinner").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
                        
                        }})   
                }}
            </script>

            <script defer type="text/javascript">
                $('#export').on('click', function() {{
                    var titles = [];
                    var data = [];

                    $('#myTable thead th').each(function() {{
                        titles.push($(this).text());
                    }});

                    titles.push('Meal');
                    console.log(titles)

                    var table = $('#myTable');
                    var table1 = $('#myTable1');
                    var table2 = $('#myTable2');

                    table.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Breakfast');
                        data.push(row); 
                    }});

                    table1.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Lunch');
                        data.push(row); 
                    }});

                    table2.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Dinner');
                        data.push(row); 
                    }});

                    console.log(data)
                    
                    csvFileData = data;
                    var csv = 'Volume (g), Food_items, Calories, Fats, Proteins, Carbohydrates, Fibre, Meal\\n'; 

                    csvFileData.forEach(function(row) {{
                        csv += row.join(',');  
                        csv += "\\n";  
                    }});  

                    var hiddenElement = document.createElement('a');  
                    hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);  
                    hiddenElement.target = '_blank';  
                    
                    hiddenElement.download = 'Diet Plan Meal';
                    hiddenElement.click();  
                }});
            </script>

            <script defer type="text/javascript">
                $(document).ready(function() {{
                $('table.table-striped').DataTable( {{
                    stateSave: true,
                    responsive: true,
                    "bPaginate": false,
                    "bInfo": false,
                }});
            }});
            </script>

        </html>"""
                                )

    output_html = template.render(lunch_dataframe=lunch_df.to_html(classes='table table-striped', header="true", table_id="myTable1", escape=False ,formatters=dict(Image=path_to_image_html)),
                breakfast_dataframe=breakfast_df.to_html(classes='table table-striped', header="true", table_id="myTable", escape=False ,formatters=dict(Image=path_to_image_html)),
                dinner_dataframe=dinner_df.to_html(classes='table table-striped', header="true", table_id="myTable2", escape=False ,formatters=dict(Image=path_to_image_html)))
    
    components.html(output_html,height=3700)

def Weight_Gain_Plan():

    TDEE,total_calo,total_protein,total_carb,total_fat = calc_TDEE()

    BreakfastFoodItemIDData, LunchFoodItemIDdata, DinnerFoodItemIDdata = meal_food_data()

    BreakfastNutrition = BreakfastFoodItemIDData
    LunchNutrition = LunchFoodItemIDdata
    DinnerNutrition = DinnerFoodItemIDdata

    BreakfastFoodItemIDData=BreakfastFoodItemIDData.to_numpy()
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.to_numpy()
    LunchFoodItemIDdata=LunchFoodItemIDdata.to_numpy()

    brklbl = breakfast_cluster_food(BreakfastFoodItemIDData, BreakfastNutrition)

    lnchlbl = lunch_cluster_food(LunchFoodItemIDdata, LunchNutrition)

    dnrlbl = dinner_cluster_food(DinnerFoodItemIDdata, DinnerNutrition)

    rows_list = []
    for idx, row in BreakfastNutrition.iterrows():
        if row['KMCluster']==0 or row['KMCluster']==2 or row['KMCluster']==1:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)
    breakfast_df = df

    rows_list = []
    for idx, row in LunchNutrition.iterrows():
        if row['KMCluster']==2 or row['KMCluster']==0 or row['KMCluster']==1:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)
    lunch_df = df


    rows_list = []
    st.subheader('CREATE MEAL PLAN FOR WEIGHT GAIN')
    for idx, row in DinnerNutrition.iterrows():
        if row['KMCluster']==0 or row['KMCluster']==1 or row['KMCluster']==2:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)
    dinner_df = df

    template = jinja2.Template(f"""<!DOCTYPE html>
        <html>

            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width">
                <title>Demo</title>
                <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
                <link rel="stylesheet" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.min.css">
                <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
                <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
                <script src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>

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
                        overflow-x: scroll;
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

                    .dataTable-table th a {{
                        color: rgb(49, 51, 63);
                        font-weight: 600;
                    }}

                    .table tbody + tbody {{
                    border-top: 2px solid #eceeef;
                    }}
                    
                    .table-striped tbody tr:nth-of-type(odd) {{
                    background-color: rgba(0, 0, 0, 0.05);
                    }}

                        .table_wrapper{{
                        display: block;
                        white-space: nowrap;
                    }}
                    .table {{
                        width: 100%;
                        border: 1px solid black;
                        table-layout: fixed;
                        overflow-x: hidden;
                        height: 600px;
                        display:block;
                    }}
                    td {{
                        border: 1px solid #dddddd;
                        text-align: center;
                        padding: 8px;
                        white-space: nowrap;
                        width: 100px;
                        vertical-align: middle;
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
                    .dataTable-table tbody tr td {{
                        vertical-align: middle;
                    }}
                    .table td {{
                        vertical-align: middle;
                    }}
                    .dataTable-table tbody tr th {{
                        vertical-align: middle;
                        text-align: center;
                    }}

                    # thead, tfoot {{
                    # display: none;
                    # }}
                    # table {{
                    # background: none !important;
                    # border: none !important;
                    # }}
                    # tr {{
                    # display: inline-block;
                    # padding: 1rem 0.5rem 1rem 0.5rem;
                    # margin: 1.5rem;
                    # border: 1px solid grey;
                    # border-radius 10px;
                    # box-shadow: 0 0 10px;
                    # }}
                    # td {{
                    # display: block;
                    # }}
                    td {{border: 1px #DDD solid; padding: 5px; cursor: pointer;}}

                    .selected {{
                        background-color: #0dcaf0 !important; 
                        color: #FFF !important;
                    }}

                    .btn-purple {{
                    color: #fff;
                    background-color: #6f42c1;
                    border-color: #643ab0;
                    }}

                    
                    
                    .card-img-top {{
                        width: 100%;
                        height: 7vh;
                        object-fit: cover;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                    }}
                    .modal-dialog {{
                        width: 27rem;
                        margin: 0 auto;
                        }}
                    .progress {{margin-bottom:0;}}
                    .start {{float:left;}}
                    .end {{float:right; text-align:right;}}
                    div p {{
                        display: inline-block;
                    }}
                    div input {{
                        display: inline-block;
                    }}
                    h5 {{
                        text-align: center;
                    }}
                    .image-parent {{
                        max-width: 40px;
                    }}
                    .modal-content{{
                        position: relative;
                        top: 50%;
                        transform: translateY(-50%);
                    }}

                    .dataTable-input {{
                        display: none;
                    }}

                    table.dataTable{{
                        box-sizing: border-box;
                        overflow: scroll;
                        }}

                    #volumnHelp {{
                        display:none;
                    }}
                </style>
            </head>
        
            

            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="calories-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="calories-left" style="width:100%">
                        Remaining: {total_calo}
                    </div>
                </div>
                <h6 class="start mt-1">Calories Daily Intake</h6>
            </div>
            <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="fats-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="fats-left" style="width:100%">
                        Remaining: {total_fat}g
                    </div>
                </div>
                <h6 class="start mt-1">Fat Daily Intake</div>
            </div>
             <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="protein-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="protein-left" style="width:100%">
                        Remaining: {total_protein}g
                    </div>
                </div>
                <h6 class="start mt-1">Protein Daily Intake</div>
            </div>
             <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="carb-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="carb-left" style="width:100%">
                        Remaining: {total_carb}g
                    </div>
                </div>
                <h6 class="start mt-1">Carbohydrate Daily Intake</div>
            </div>

            
            
            <body>
                <h3>BREAKFAST</h3>
                {{{{ breakfast_dataframe }}}}
                <br>
                <h3>LUNCH</h3>
                {{{{ lunch_dataframe }}}}
                <br>
                <h3>DINNER</h3>
                {{{{ dinner_dataframe }}}}
    
            </body>

            <div class="modal fade" id="myModal" role="dialog" aria-hidden="true">
                <div class="modal-dialog">
                <div class="modal-content">
                    <div class="card">
                        <img class="food-image card-img-top" src="" alt="food image">
                        <div class="card-body">
                            <h5 class="Food-modal card-title"><span></span></h5>
                            <br>
                            <div class="row">
                                <div class="Volumn-modal col-sm-3"><p>Volumn(g): </p></div>
                                <div class="col-sm-4">
                                    <input type="number" class="form-control form-control-sm" id="volumn-input">
                                </div>
                                <div class="col-sm-5">
                                    <small id="volumnHelp" class="text-danger">
                                    Must be a number.
                                    </small>      
                                </div>
                            </div>
                            <div class="row">
                                <div class="Calories-modal"><p>Calories:  </p><span></span></div>
                            </div>
                            <div class="row">
                                <div class="col-sm-6">
                                    
                                    <div class="Protein-modal"><p>Protein(g):  </p><span></span>g</div>
                                    <div class="Fibre-modal"><p>Fibre(g):  </p><span></span>g</div>
                                </div>
                                <div class="col-sm-6">
                                    <div class="Fat-modal"><p>Fat(g):</p><span></span>g</div>
                                    <div class="Carbohydrate-modal"><p>Carbohydrate(g):  </p><span></span>g</div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col text-center">
                                    <button class="btn btn-primary" id="btn-modal-save">Select</button>
                                    <button class="btn btn-secondary" id="btn-modal-close"> Close</button>
                                </div>
                            </div>
                            
                        </div>
                    </div>
                    </div>
                </div>
            </div>

            <div class="text-right mb-1">
                <button class="btn btn-secondary mt-3" id="export">Export CSV</button>
            </div>

            <div id="accordion">
                <div class="card">
                    <div class="card-header text-white bg-secondary" id="headingOne">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
                        Breakfast Meal
                        </button>
                    </h5>
                    </div>

                    <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
                        <div class="panel-body" id="breakfast" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                               
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header bg-danger" id="headingTwo">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
                        Lunch Meal
                        </button>
                    </h5>
                    </div>
                    <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
                        <div class="panel-body" id="lunch" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                            
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header bg-dark" id="headingThree">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseThree" aria-expanded="false" aria-controls="collapseThree">
                        Dinner Meal
                        </button>
                    </h5>
                    </div>
                    <div id="collapseThree" class="collapse" aria-labelledby="headingThree" data-parent="#accordion">
                        <div class="panel-body" id="dinner" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                            
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

            <script defer type="text/javascript">
                let myTable = new simpleDatatables.DataTable("#myTable", {{paging:false}});
            </script>

            <script defer type="text/javascript">
                let myTable1 = new simpleDatatables.DataTable("#myTable1", {{paging:false}});
            </script>

            <script defer type="text/javascript">
                let myTable2 = new simpleDatatables.DataTable("#myTable2", {{paging:false}});
            </script>
            
            <script defer type="text/javascript">
                var tableIDs = 'abc';
            
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

                var tr1;

                $("#volumn-input").on("focus", function() {{
                    if($.isNumeric($('#volumn-input').val())) {{
                        var values = new Array();
                        var data = $(event.target);
                        
                        
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});    

                        ratio_old = parseFloat(values[0]['Volumn']);
                    }}                            
                }});

                $(document).on("blur", "#volumn-input", function() {{
                    var values = new Array();

                    var data = $(event.target);

                    if($.isNumeric($('#volumn-input').val())) {{
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);

                        $(".Calories-modal span").text(calo_fixed.toFixed(1));
                        $('.Fat-modal span').text(fats_fixed.toFixed(1));
                        $('.Protein-modal span').text(proteins_fixed.toFixed(1));
                        $('.Carbohydrate-modal span').text(carbohydrates_fixed.toFixed(1));
                        

                        var food_name = $('.Food-modal span').text();
                        food_name = food_name.replace(/^\s+|\s+$/gm,'')

                        var a = $('#myTable tr td:contains("' + food_name + '")').filter(function(){{
                            console.log($.trim($(this).text()));
                            if($.trim($(this).text()) == food_name)
                            return true;
                            else
                            return false;
                        }});
                        var tr = $(a).parents('tr:eq(0)');
                        tr1 = tr; 
                    }} 
                }});

                $('#btn-modal-save').on('click',{{tableIDs: tableIDs}}, myfunction) 

                function myfunction(e) {{
                    var values = new Array();
                    
                    if($.isNumeric($('#volumn-input').val())) {{
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});    

                        ratio_old = parseFloat(values[0]['Volumn']);

                        var values = new Array();
                        
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);

                        $(".Calories-modal span").text(calo_fixed.toFixed(1));
                        $('.Fat-modal span').text(fats_fixed.toFixed(1));
                        $('.Protein-modal span').text(proteins_fixed.toFixed(1));
                        $('.Carbohydrate-modal span').text(carbohydrates_fixed.toFixed(1));
                        
                        var food_name = $('.Food-modal span').text();
                        food_name = food_name.replace(/^\s+|\s+$/gm,'')

                        var a;
                        var tr;
                        if (tableIDs == 'myTable') {{
                            a = $('#myTable tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                        }}
                        else if (tableIDs == 'myTable1') {{
                            a = $('#myTable1 tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                            console.log('123')
                        }}
                        else if (tableIDs == 'myTable2') {{
                            a = $('#myTable2 tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                            console.log('456')
                        }}
                        
                        $(tr).find('td:eq(0)').text($('#volumn-input').val());
                        $(tr).find('td:eq(3)').text(calo_fixed.toFixed(1));
                        $(tr).find('td:eq(4)').text(fats_fixed.toFixed(1));
                        $(tr).find('td:eq(5)').text(proteins_fixed.toFixed(1));
                        $(tr).find('td:eq(6)').text(carbohydrates_fixed.toFixed(1));

                        $(tr).addClass("selected");
                        calc_new1();
                        show_meal();
                        $("#volumnHelp").css("display", "none");
                    }}

                    else {{
                        $("#volumnHelp").css("display", "inline-block");
                    }}
                }}
            </script>
            <script type="text/javascript" src="https://cdn.datatables.net/1.10.8/js/jquery.dataTables.min.js"></script>


            <script defer type="text/javascript">
                function calc_new1() {{
                    
                    var valuesss = new Array();
                    var rows_selection = new Array();
                    var selected_rowss = document.getElementsByClassName("selected");
                    $('table.table-striped').DataTable().rows('.selected').invalidate();
                    var selection_rows = $('table.table-striped').DataTable().rows('.selected').data()

                    var numberOfChecked = $('table.table-striped').DataTable().rows('.selected').count();
                    if (numberOfChecked == 0) {{
                            $("#calories-intake").css("width", 0 + "%").text(0);
                            $("#calories-left").css("width", 100 + "%").text(({total_calo}).toFixed(1) + " remaining");

                            $("#fats-intake").css("width", 0 + "%").text(0 + "g");
                            $("#fats-left").css("width", 100 + "%").text(({total_fat}).toFixed(1) + "g remaining");

                            $("#protein-intake").css("width", 0 + "%").text(0 + "g");
                            $("#protein-left").css("width", 100 + "%").text(({total_protein}).toFixed(1) + "g remaining");

                            $("#carb-intake").css("width", 0 + "%").text(0 + "g");
                            $("#carb-left").css("width", 100 + "%").text(({total_carb}).toFixed(1) + "g remaining");
                        }}

            
                    $.each(selection_rows, function(){{
                        console.log(this)
                        var Row=this;
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});   

                                      
                        var total_calories = 0;
                        var total_fats = 0;
                        var total_proteins = 0;
                        var total_carbs = 0;
                
                        for(var i = 0; i < valuesss.length; i++) {{
                            total_calories = total_calories + parseFloat(valuesss[i]['Calories']);
                            total_fats = total_fats + parseFloat(valuesss[i]['Fats']);
                            total_proteins = total_proteins + parseFloat(valuesss[i]['Proteins']);
                            total_carbs = total_carbs + parseFloat(valuesss[i]['Carbohydrates']);
                        }}

                       
                        var calories_ratio_percentage = (total_calories/{total_calo}).toFixed(1)*100;
                        var fat_ratio_percentage = (total_fats/{total_fat}).toFixed(1)*100;
                        var protein_ratio_percentage = (total_proteins/{total_protein}).toFixed(1)*100;
                        var carb_ratio_percentage = (total_carbs/{total_carb}).toFixed(1)*100;

                        if (total_calories > {total_calo})
                        {{
                            $('#calories-intake').addClass('bg-danger');
                            $('#calories-intake').css("width", 100 + "%").text("Excess calories: " + (total_calories - {total_calo}).toFixed(1));
                            $("#calories-left").css("width", 0 + "%").text(({total_calo} - total_calories).toFixed(1) + " remaining");
                        }}
                       
                        else {{
                            $('#calories-intake').removeClass('bg-danger');
                            $("#calories-intake").css("width", calories_ratio_percentage + "%").text(total_calories.toFixed(1));
                            $("#calories-left").css("width", 100-calories_ratio_percentage + "%").text(({total_calo} - total_calories).toFixed(1) + " remaining");
                        }}

                        if (total_fats > {total_fat})
                        {{
                            $('#fats-intake').addClass('bg-danger');
                            $('#fats-intake').css("width", 100 + "%").text("Excess fat: " + (total_fats - {total_fat}).toFixed(1) +"g");
                            $("#fats-left").css("width", 0 + "%").text(({total_fat} - total_fats).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#fats-intake').removeClass('bg-danger');
                            $("#fats-intake").css("width", fat_ratio_percentage + "%").text(total_fats.toFixed(1) + "g");
                            $("#fats-left").css("width", 100-fat_ratio_percentage + "%").text(({total_fat} - total_fats).toFixed(1) + "g remaining");
                        }}

                        if (total_proteins > {total_protein})
                        {{
                            $('#protein-intake').addClass('bg-danger');
                            $('#protein-intake').css("width", 100 + "%").text("Excess protein: " + (total_proteins - {total_protein}).toFixed(1) + "g");
                            $("#protein-left").css("width", 0 + "%").text(({total_protein} - total_proteins).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#protein-intake').removeClass('bg-danger');
                            $("#protein-intake").css("width", protein_ratio_percentage + "%").text(total_proteins.toFixed(1) + "g");
                            $("#protein-left").css("width", 100-protein_ratio_percentage + "%").text(({total_protein} - total_proteins).toFixed(1) + "g remaining");
                        }}

                        if (total_carbs > {total_carb})
                        {{
                            $('#carb-intake').addClass('bg-danger');
                            $('#carb-intake').css("width", 100 + "%").text("Excess carbohydrate: " + (total_carbs - {total_carb}).toFixed(1) + "g");
                            $("#carb-left").css("width", 0 + "%").text(({total_carb} - total_carbs).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#carb-intake').removeClass('bg-danger');
                            $("#carb-intake").css("width", carb_ratio_percentage + "%").text(total_carbs.toFixed(1) + "g");
                            $("#carb-left").css("width", 100-carb_ratio_percentage + "%").text(({total_carb} - total_carbs).toFixed(1) + "g remaining");
                        }}


                        if (numberOfChecked == 0) {{
                            document.getElementById("calories").innerHTML = '0';
                            document.getElementById("fats").innerHTML = '0';
                            document.getElementById("proteins").innerHTML = '0';
                            document.getElementById("carbohydrates").innerHTML = '0';
                        }}

                        $("#myModal").modal("hide");
                    }});
                }}
            </script>

            <script defer type="text/javascript">
                $('#btn-modal-close').on('click', function() {{
                    $("#myModal").modal("hide");
                }})
            </script>

            <!-- jQuery library -->
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>

            <!-- Latest compiled JavaScript -->
            <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
            <script src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>

            <script defer type="text/javascript">
                $("#myTable").on('click','tr:gt(0)',function(){{
                    
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());
                    
                    $(".modal-dialog").css("height", "42%");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");
                    
                    var tableID = $(this).closest('table').attr('id');
                    var tableIDD = tableID;
                    
                   tableIDs = tableID;
                   console.log(tableIDs)
                }});

                    
                    
            </script>

            <script defer type="text/javascript">
                $("#myTable1").on('click','tr:gt(0)',function() {{
                    console.log('2222222')
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());
                    
                    var tableID1 = $(this).closest('table').attr('id');
                    var tableIDD1 = tableID1;
                    tableIDs = tableID1;
                    
                    $(".modal-dialog").css("height", "82%");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");


                }});

                    
                       
            </script>

            <script defer type="text/javascript">
                $("#myTable2").on('click','tr:gt(0)',function() {{
                    meal = 'dinner';
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());
                    
                    var tableID2 = $(this).closest('table').attr('id');
                    var tableIDD2 = tableID2;
                    tableIDs = tableID2;
                    
                    $(".modal-dialog").css("height", "126%");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");

                 
                }});

                
            </script>

            <script defer type="text/javascript">
                function show_meal() {{
                    var table = document.getElementById("myTable");
                    var selected_rowss = table.getElementsByClassName("selected");
                    $("#breakfast").empty();

                    $('#myTable').DataTable().rows('.selected').every(function(element, index){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});         
                       
                        $("#breakfast").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
 
                   }})   

                    var table1 = document.getElementById("myTable1");
                    var selected_rowss1 = table1.getElementsByClassName("selected");
                    $("#lunch").empty();
                    $('#myTable1').DataTable().rows('.selected').every(function(){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});                
                    $("#lunch").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
                        
                        }})   

                    var table2 = document.getElementById("myTable2");
                    var selected_rowss2 = table2.getElementsByClassName("selected");
                    $("#dinner").empty();
                    $('#myTable2').DataTable().rows('.selected').every(function(){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});                   
                                       
                        $("#dinner").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
                        
                        }})   
        
                }}
            </script>

            <script defer type="text/javascript">
                $('#export').on('click', function() {{
                    var titles = [];
                    var data = [];

                    $('#myTable thead th').each(function() {{
                        titles.push($(this).text());
                    }});

                    titles.push('Meal');
                    console.log(titles)

                    var table = $('#myTable');
                    var table1 = $('#myTable1');
                    var table2 = $('#myTable2');

                    table.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Breakfast');
                        data.push(row); 
                    }});

                    table1.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Lunch');
                        data.push(row); 
                    }});

                    table2.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Dinner');
                        data.push(row); 
                    }});

                    console.log(data)
                    
                    csvFileData = data;
                    var csv = 'Volume (g), Food_items, Calories, Fats, Proteins, Carbohydrates, Fibre, Meal\\n'; 

                    csvFileData.forEach(function(row) {{
                        csv += row.join(',');  
                        csv += "\\n";  
                    }});  

                    var hiddenElement = document.createElement('a');  
                    hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);  
                    hiddenElement.target = '_blank';  
                    
                    hiddenElement.download = 'Diet Plan Meal';  
                    hiddenElement.click();  
                }});
            </script>

            <script defer type="text/javascript">
                $(document).ready(function() {{
                $('table.table-striped').DataTable( {{
                    stateSave: true,
                    responsive: true,
                    "bPaginate": false,
                    "bInfo": false,
                }});
            }});
            </script>
        </html>"""
                                )

    output_html = template.render(lunch_dataframe=lunch_df.to_html(classes='table table-striped', header="true", table_id="myTable1", escape=False ,formatters=dict(Image=path_to_image_html)),
                breakfast_dataframe=breakfast_df.to_html(classes='table table-striped', header="true", table_id="myTable", escape=False ,formatters=dict(Image=path_to_image_html)),
                dinner_dataframe=dinner_df.to_html(classes='table table-striped', header="true", table_id="myTable2", escape=False ,formatters=dict(Image=path_to_image_html)))

    components.html(output_html,height=3700)  

def Maintenance_Plan():

    TDEE,total_calo,total_protein,total_carb,total_fat = calc_TDEE()

    BreakfastFoodItemIDData, LunchFoodItemIDdata, DinnerFoodItemIDdata = meal_food_data()

    BreakfastNutrition = BreakfastFoodItemIDData
    LunchNutrition = LunchFoodItemIDdata
    DinnerNutrition = DinnerFoodItemIDdata

    BreakfastFoodItemIDData=BreakfastFoodItemIDData.to_numpy()
    DinnerFoodItemIDdata=DinnerFoodItemIDdata.to_numpy()
    LunchFoodItemIDdata=LunchFoodItemIDdata.to_numpy()

    brklbl = breakfast_cluster_food(BreakfastFoodItemIDData, BreakfastNutrition)

    lnchlbl = lunch_cluster_food(LunchFoodItemIDdata, LunchNutrition)

    dnrlbl = dinner_cluster_food(DinnerFoodItemIDdata, DinnerNutrition)

    rows_list = []
    for idx, row in BreakfastNutrition.iterrows():
        if row['KMCluster']==0 or row['KMCluster']==1:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)
    breakfast_df = df

    rows_list = []
    for idx, row in LunchNutrition.iterrows():
        if row['KMCluster']==0 or row['KMCluster']==1:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)
    lunch_df = df

    rows_list = []
    st.subheader('CREATE MEAL PLAN FOR MAINTENANCE')
    for idx, row in DinnerNutrition.iterrows():
        if row['KMCluster']==1 or row['KMCluster']==2:
            row = row[['Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']]
            rows_list.append(row)

    df = pd.DataFrame(rows_list)
    df.insert(loc = 0,column = 'Volume (g)',value = '100')
    df.columns = ['Volume (g)', 'Image','Food_items', 'Calories', 'Fats', 'Proteins', 'Carbohydrates', 'Fibre']
    mapping = {df.columns[2]:'Food', df.columns[3]: 'Calories', df.columns[4]: 'Fats (g)', df.columns[5]: 'Proteins (g)', df.columns[6]: 'Carbohydrates (g)', df.columns[7]: 'Fibre (g)'}
    df = df.rename(columns=mapping)

    df.append(df, ignore_index = True, sort = False)

    df = df.reset_index(drop=True)

    dinner_df = df

    template = jinja2.Template(f"""<!DOCTYPE html>
        <html>

            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width">
                <title>Demo</title>
                <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
                <link rel="stylesheet" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.min.css">
                <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
                <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
                <script src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>

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
                        overflow-x: scroll;
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

                    .dataTable-table th a {{
                        color: rgb(49, 51, 63);
                        font-weight: 600;
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
                        width: 100%;
                        border: 1px solid black;
                        table-layout: fixed;
                        overflow-x: hidden;
                        height: 600px;
                        display:block;
                    }}
                    td {{
                        border: 1px solid #dddddd;
                        text-align: center;
                        padding: 8px;
                        white-space: nowrap;
                        width: 100px;
                        vertical-align: middle;
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
                    .dataTable-table tbody tr td {{
                        vertical-align: middle;
                    }}
                    .table td {{
                        vertical-align: middle;
                    }}
                    .dataTable-table tbody tr th {{
                        vertical-align: middle;
                        text-align: center;
                    }}

                    # thead, tfoot {{
                    # display: none;
                    # }}
                    # table {{
                    # background: none !important;
                    # border: none !important;
                    # }}
                    # tr {{
                    # display: inline-block;
                    # padding: 1rem 0.5rem 1rem 0.5rem;
                    # margin: 1.5rem;
                    # border: 1px solid grey;
                    # border-radius 10px;
                    # box-shadow: 0 0 10px;
                    # }}
                    # td {{
                    # display: block;
                    # }}
                    td {{border: 1px #DDD solid; padding: 5px; cursor: pointer;}}

                    .selected {{
                        background-color: #0dcaf0 !important; 
                        color: #FFF !important;
                    }}

                    .btn-purple {{
                    color: #fff;
                    background-color: #6f42c1;
                    border-color: #643ab0;
                    }}

                    
                    
                    .card-img-top {{
                        width: 100%;
                        height: 7vh;
                        object-fit: cover;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                    }}
                    .modal-dialog {{
                        width: 27rem;
                        margin: 0 auto;
                        }}
                    .progress {{margin-bottom:0;}}
                    .start {{float:left;}}
                    .end {{float:right; text-align:right;}}
                    div p {{
                        display: inline-block;
                    }}
                    div input {{
                        display: inline-block;
                    }}
                    h5 {{
                        text-align: center;
                    }}
                    .image-parent {{
                        max-width: 40px;
                    }}
                    .modal-content{{
                        position: relative;
                        top: 50%;
                        transform: translateY(-50%);
                    }}

                    .dataTable-input {{
                        display: none;
                    }}

                    table.dataTable{{
                        box-sizing: border-box;
                        overflow: scroll;
                    }}

                    #volumnHelp {{
                        display:none;
                    }}
                </style>
            </head>

            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="calories-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="calories-left" style="width:100%">
                        Remaining: {total_calo}
                    </div>
                </div>
                <h6 class="start mt-1">Calories Daily Intake</h6>
            </div>
            <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="fats-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="fats-left" style="width:100%">
                        Remaining: {total_fat}g
                    </div>
                </div>
                <h6 class="start mt-1">Fat Daily Intake</div>
            </div>
             <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="protein-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="protein-left" style="width:100%">
                        Remaining: {total_protein}g
                    </div>
                </div>
                <h6 class="start mt-1">Protein Daily Intake</div>
            </div>
             <br>
            <div>
                <div class="progress" style="height: 30px;">
                    <div class="progress-bar bg-info" id="carb-intake" style="width: 0%;"></div>
                    <div class="progress-bar bg-secondary" id="carb-left" style="width:100%">
                        Remaining: {total_carb}g
                    </div>
                </div>
                <h6 class="start mt-1">Carbohydrate Daily Intake</div>
            </div>
            
            <body>
                <h3>BREAKFAST</h3>
                {{{{ breakfast_dataframe }}}}
                <br>
                <h3>LUNCH</h3>
                {{{{ lunch_dataframe }}}}
                <br>
                 <h3>DINNER</h3>
                {{{{ dinner_dataframe }}}}
    
            </body>

            <div class="modal fade" id="myModal" role="dialog" aria-hidden="true">
                <div class="modal-dialog">
                <div class="modal-content">
                    <div class="card">
                        <img class="food-image card-img-top" src="" alt="food image">
                        <div class="card-body">
                            <h5 class="Food-modal card-title"><span></span></h5>
                            <br>
                            <div class="row">
                                <div class="Volumn-modal col-sm-3"><p>Volumn(g): </p></div>
                                <div class="col-sm-4">
                                    <input type="number" class="form-control form-control-sm" id="volumn-input">
                                </div>
                                <div class="col-sm-5">
                                    <small id="volumnHelp" class="text-danger">
                                    Must be a number.
                                    </small>      
                                </div>
                            </div>
                            <div class="row">
                                <div class="Calories-modal"><p>Calories:  </p><span></span></div>
                            </div>
                            <div class="row">
                                <div class="col-sm-6">
                                    
                                    <div class="Protein-modal"><p>Protein(g):  </p><span></span>g</div>
                                    <div class="Fibre-modal"><p>Fibre(g):  </p><span></span>g</div>
                                </div>
                                <div class="col-sm-6">
                                    <div class="Fat-modal"><p>Fat(g):</p><span></span>g</div>
                                    <div class="Carbohydrate-modal"><p>Carbohydrate(g):  </p><span></span>g</div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col text-center">
                                    <button class="btn btn-primary" id="btn-modal-save">Select</button>
                                    <button class="btn btn-secondary" id="btn-modal-close"> Close</button>
                                </div>
                            </div>
                            
                        </div>
                    </div>
                    </div>
                </div>
            </div>

            <div class="text-right mb-1">
                <button class="btn btn-secondary mt-3" id="export">Export CSV</button>
            </div>

            <div id="accordion">
                <div class="card">
                    <div class="card-header text-white bg-secondary" id="headingOne">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
                        Breakfast Meal
                        </button>
                    </h5>
                    </div>

                    <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
                        <div class="panel-body" id="breakfast" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                               
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header bg-danger" id="headingTwo">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
                        Lunch Meal
                        </button>
                    </h5>
                    </div>
                    <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
                        <div class="panel-body" id="lunch" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                            
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header bg-dark" id="headingThree">
                    <h5 class="mb-0">
                        <button class="btn text-white collapsed" data-toggle="collapse" data-target="#collapseThree" aria-expanded="false" aria-controls="collapseThree">
                        Dinner Meal
                        </button>
                    </h5>
                    </div>
                    <div id="collapseThree" class="collapse" aria-labelledby="headingThree" data-parent="#accordion">
                        <div class="panel-body" id="dinner" style="padding:0px">
                            <ul class="list-group" style="margin-bottom: 0px;">
                            
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

            <script defer type="text/javascript">
                let myTable = new simpleDatatables.DataTable("#myTable", {{paging:false}});
            </script>

            <script defer type="text/javascript">
                let myTable1 = new simpleDatatables.DataTable("#myTable1", {{paging:false}});
            </script>

            <script defer type="text/javascript">
                let myTable2 = new simpleDatatables.DataTable("#myTable2", {{paging:false}});
            </script>
            
            <script defer type="text/javascript">
                var tableIDs = 'abc';
            
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

                var tr1;

                $("#volumn-input").on("focus", function() {{
                    if($.isNumeric($('#volumn-input').val())) {{
                        var values = new Array();
                        var data = $(event.target);
                        
                        
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});    

                        ratio_old = parseFloat(values[0]['Volumn']);
                    }}                            
                }});

                $(document).on("blur", "#volumn-input", function() {{
                    var values = new Array();

                    var data = $(event.target);

                    if($.isNumeric($('#volumn-input').val())) {{
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);

                        $(".Calories-modal span").text(calo_fixed.toFixed(1));
                        $('.Fat-modal span').text(fats_fixed.toFixed(1));
                        $('.Protein-modal span').text(proteins_fixed.toFixed(1));
                        $('.Carbohydrate-modal span').text(carbohydrates_fixed.toFixed(1));
                        

                        var food_name = $('.Food-modal span').text();
                        food_name = food_name.replace(/^\s+|\s+$/gm,'')

                        var a = $('#myTable tr td:contains("' + food_name + '")').filter(function(){{
                            console.log($.trim($(this).text()));
                            if($.trim($(this).text()) == food_name)
                            return true;
                            else
                            return false;
                        }});
                        var tr = $(a).parents('tr:eq(0)');
                        tr1 = tr; 
                    }} 
                }});

                $('#btn-modal-save').on('click',{{tableIDs: tableIDs}}, myfunction) 

                function myfunction(e) {{
                    var values = new Array();
                    
                    if($.isNumeric($('#volumn-input').val())) {{
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});    

                        ratio_old = parseFloat(values[0]['Volumn']);

                        var values = new Array();
                        
                        values.push({{ 'Volumn':$('#volumn-input').val(), 'Food_items':$('.Food-modal span').text() , 'Calories':$('.Calories-modal span').text(),
                                            'Fats':$('.Fat-modal span').text(), 'Proteins':$('.Protein-modal span').text(),
                                            'Carbohydrates':$('.Carbohydrate-modal span').text(),
                                            }});     

                        ratio = parseFloat(values[0]['Volumn']) / ratio_old;
                        calo_fixed = (parseFloat(values[0]['Calories']) * ratio);
                        fats_fixed = (parseFloat(values[0]['Fats']) * ratio);
                        proteins_fixed = (parseFloat(values[0]['Proteins']) * ratio);
                        carbohydrates_fixed = (parseFloat(values[0]['Carbohydrates']) * ratio);

                        $(".Calories-modal span").text(calo_fixed.toFixed(1));
                        $('.Fat-modal span').text(fats_fixed.toFixed(1));
                        $('.Protein-modal span').text(proteins_fixed.toFixed(1));
                        $('.Carbohydrate-modal span').text(carbohydrates_fixed.toFixed(1));
                        
                        var food_name = $('.Food-modal span').text();
                        food_name = food_name.replace(/^\s+|\s+$/gm,'')

                        var a;
                        var tr;
                        if (tableIDs == 'myTable') {{
                            a = $('#myTable tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                        }}
                        else if (tableIDs == 'myTable1') {{
                            a = $('#myTable1 tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                            console.log('123')
                        }}
                        else if (tableIDs == 'myTable2') {{
                            a = $('#myTable2 tr td:contains("' + food_name + '")').filter(function(){{
                                console.log($.trim($(this).text()));
                                if($.trim($(this).text()) == food_name)
                                return true;
                                else
                                return false;
                            }});
                            tr = $(a).parents('tr:eq(0)');
                            console.log('456')
                        }}
                        
                        $(tr).find('td:eq(0)').text($('#volumn-input').val());
                        $(tr).find('td:eq(3)').text(calo_fixed.toFixed(1));
                        $(tr).find('td:eq(4)').text(fats_fixed.toFixed(1));
                        $(tr).find('td:eq(5)').text(proteins_fixed.toFixed(1));
                        $(tr).find('td:eq(6)').text(carbohydrates_fixed.toFixed(1));

                        $(tr).addClass("selected");
                        calc_new1();
                        show_meal();
                        $("#volumnHelp").css("display", "none");
                    }}

                    else {{
                        $("#volumnHelp").css("display", "inline-block");
                    }}
                }}
            </script>
            <script type="text/javascript" src="https://cdn.datatables.net/1.10.8/js/jquery.dataTables.min.js"></script>

            <script defer type="text/javascript">
                function calc_new1() {{
                    
                    var valuesss = new Array();
                    var rows_selection = new Array();
                    var selected_rowss = document.getElementsByClassName("selected");
                    $('table.table-striped').DataTable().rows('.selected').invalidate();
                    var selection_rows = $('table.table-striped').DataTable().rows('.selected').data()

                    var numberOfChecked = $('table.table-striped').DataTable().rows('.selected').count();
                    if (numberOfChecked == 0) {{
                            $("#calories-intake").css("width", 0 + "%").text(0);
                            $("#calories-left").css("width", 100 + "%").text(({total_calo}).toFixed(1) + " remaining");

                            $("#fats-intake").css("width", 0 + "%").text(0 + "g");
                            $("#fats-left").css("width", 100 + "%").text(({total_fat}).toFixed(1) + "g remaining");

                            $("#protein-intake").css("width", 0 + "%").text(0 + "g");
                            $("#protein-left").css("width", 100 + "%").text(({total_protein}).toFixed(1) + "g remaining");

                            $("#carb-intake").css("width", 0 + "%").text(0 + "g");
                            $("#carb-left").css("width", 100 + "%").text(({total_carb}).toFixed(1) + "g remaining");
                        }}

            
                    $.each(selection_rows, function(){{
                        console.log(this)
                        var Row=this;
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});   

                                      
                        var total_calories = 0;
                        var total_fats = 0;
                        var total_proteins = 0;
                        var total_carbs = 0;
                
                        for(var i = 0; i < valuesss.length; i++) {{
                            total_calories = total_calories + parseFloat(valuesss[i]['Calories']);
                            total_fats = total_fats + parseFloat(valuesss[i]['Fats']);
                            total_proteins = total_proteins + parseFloat(valuesss[i]['Proteins']);
                            total_carbs = total_carbs + parseFloat(valuesss[i]['Carbohydrates']);
                        }}

                       
                        var calories_ratio_percentage = (total_calories/{total_calo}).toFixed(1)*100;
                        var fat_ratio_percentage = (total_fats/{total_fat}).toFixed(1)*100;
                        var protein_ratio_percentage = (total_proteins/{total_protein}).toFixed(1)*100;
                        var carb_ratio_percentage = (total_carbs/{total_carb}).toFixed(1)*100;

                        if (total_calories > {total_calo})
                        {{
                            $('#calories-intake').addClass('bg-danger');
                            $('#calories-intake').css("width", 100 + "%").text("Excess calories: " + (total_calories - {total_calo}).toFixed(1));
                            $("#calories-left").css("width", 0 + "%").text(({total_calo} - total_calories).toFixed(1) + " remaining");
                        }}
                       
                        else {{
                            $('#calories-intake').removeClass('bg-danger');
                            $("#calories-intake").css("width", calories_ratio_percentage + "%").text(total_calories.toFixed(1));
                            $("#calories-left").css("width", 100-calories_ratio_percentage + "%").text(({total_calo} - total_calories).toFixed(1) + " remaining");
                        }}

                        if (total_fats > {total_fat})
                        {{
                            $('#fats-intake').addClass('bg-danger');
                            $('#fats-intake').css("width", 100 + "%").text("Excess fat: " + (total_fats - {total_fat}).toFixed(1) +"g");
                            $("#fats-left").css("width", 0 + "%").text(({total_fat} - total_fats).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#fats-intake').removeClass('bg-danger');
                            $("#fats-intake").css("width", fat_ratio_percentage + "%").text(total_fats.toFixed(1) + "g");
                            $("#fats-left").css("width", 100-fat_ratio_percentage + "%").text(({total_fat} - total_fats).toFixed(1) + "g remaining");
                        }}

                        if (total_proteins > {total_protein})
                        {{
                            $('#protein-intake').addClass('bg-danger');
                            $('#protein-intake').css("width", 100 + "%").text("Excess protein: " + (total_proteins - {total_protein}).toFixed(1) + "g");
                            $("#protein-left").css("width", 0 + "%").text(({total_protein} - total_proteins).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#protein-intake').removeClass('bg-danger');
                            $("#protein-intake").css("width", protein_ratio_percentage + "%").text(total_proteins.toFixed(1) + "g");
                            $("#protein-left").css("width", 100-protein_ratio_percentage + "%").text(({total_protein} - total_proteins).toFixed(1) + "g remaining");
                        }}

                        if (total_carbs > {total_carb})
                        {{
                            $('#carb-intake').addClass('bg-danger');
                            $('#carb-intake').css("width", 100 + "%").text("Excess carbohydrate: " + (total_carbs - {total_carb}).toFixed(1) + "g");
                            $("#carb-left").css("width", 0 + "%").text(({total_carb} - total_carbs).toFixed(1) + "g remaining");
                        }}
                        else {{
                            $('#carb-intake').removeClass('bg-danger');
                            $("#carb-intake").css("width", carb_ratio_percentage + "%").text(total_carbs.toFixed(1) + "g");
                            $("#carb-left").css("width", 100-carb_ratio_percentage + "%").text(({total_carb} - total_carbs).toFixed(1) + "g remaining");
                        }}


                        if (numberOfChecked == 0) {{
                            document.getElementById("calories").innerHTML = '0';
                            document.getElementById("fats").innerHTML = '0';
                            document.getElementById("proteins").innerHTML = '0';
                            document.getElementById("carbohydrates").innerHTML = '0';
                        }}

                        $("#myModal").modal("hide");
                    }});
                }}
            </script>

            <script defer type="text/javascript">
                $('#btn-modal-close').on('click', function() {{
                    $("#myModal").modal("hide");
                }})
            </script>

            <!-- jQuery library -->
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>

            <!-- Latest compiled JavaScript -->
            <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
            <script src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>

            <script defer type="text/javascript">
                $("#myTable").on('click','tr:gt(0)',function(){{
                    
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());
                    
                    $(".modal-dialog").css("height", "42%");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");
                    
                    var tableID = $(this).closest('table').attr('id');
                    var tableIDD = tableID;
                    
                   tableIDs = tableID;
                   console.log(tableIDs)
                }});

                    
                    
            </script>

            <script defer type="text/javascript">
                $("#myTable1").on('click','tr:gt(0)',function() {{
                    console.log('2222222')
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());
                    
                    var tableID1 = $(this).closest('table').attr('id');
                    var tableIDD1 = tableID1;
                    tableIDs = tableID1;
                    
                    $(".modal-dialog").css("height", "82%");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");


                }});

                    
                       
            </script>

            <script defer type="text/javascript">
                $("#myTable2").on('click','tr:gt(0)',function() {{
                    meal = 'dinner';
                    if($(this).hasClass('selected')) {{
                        $(this).removeClass('selected');
                        calc_new1();
                        show_meal();
                        return;
                    }}

                    $(".food-image").attr("src", $(this).find('img').attr('src'));
                    $(".card-body div span").text("");
                    $(".col-sm-4 input").val($(this).find('td:eq(0)').text());
                    $(".Food-modal span").text(" " + $(this).find('td:eq(2)').text());
                    $(".Calories-modal span").text(" " + $(this).find('td:eq(3)').text());
                    $(".Fat-modal span").text(" " + $(this).find('td:eq(4)').text());
                    $(".Protein-modal span").text(" " + $(this).find('td:eq(5)').text());
                    $(".Carbohydrate-modal span").text(" " + $(this).find('td:eq(6)').text());
                    $(".Fibre-modal span").text(" " + $(this).find('td:eq(7)').text());
                    
                    var tableID2 = $(this).closest('table').attr('id');
                    var tableIDD2 = tableID2;
                    tableIDs = tableID2;
                    
                    $(".modal-dialog").css("height", "126%");
                    $("#volumnHelp").css("display", "none");
                    $("#myModal").modal("show");

                 
                }});

                
            </script>

            <script defer type="text/javascript">
                function show_meal() {{
                    var table = document.getElementById("myTable");
                    var selected_rowss = table.getElementsByClassName("selected");
                    $("#breakfast").empty();

                    $('#myTable').DataTable().rows('.selected').every(function(element, index){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});         
                       
                        $("#breakfast").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
 
                   }})   

                    var table1 = document.getElementById("myTable1");
                    var selected_rowss1 = table1.getElementsByClassName("selected");
                    $("#lunch").empty();
                    $('#myTable1').DataTable().rows('.selected').every(function(){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});                
                    $("#lunch").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
                        
                        }})   

                    var table2 = document.getElementById("myTable2");
                    var selected_rowss2 = table2.getElementsByClassName("selected");
                    $("#dinner").empty();
                    $('#myTable2').DataTable().rows('.selected').every(function(){{
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        valuesss.push({{ 'Volumn':Row[1], 'Food_items':Row[3] , 'Calories':Row[4],
                                        'Fats':Row[5], 'Proteins':Row[6],
                                        'Carbohydrates':Row[7], 'Fibre':Row[8],
                                        }});                   
                      
                        
                        $("#dinner").append('<div class="border-bottom"><div class="d-flex w-100 justify-content-between"><h6 class="mb-1 ml-3 mt-2">' + valuesss[0]['Food_items'] + '</h6><small class="mr-2 mt-2">Volumn: ' + valuesss[0]['Volumn'] + 'g</small></div><p class="mb-2 mr-2 "><span class="ml-3">Calories: ' + valuesss[0]['Calories'] + ' </span><span class="ml-3">Fats: ' + valuesss[0]['Fats'] +
                        'g </span><span class="ml-3">Proteins: ' + valuesss[0]['Proteins'] + 'g </span><span class="ml-3">Carbohydrates: ' + valuesss[0]['Carbohydrates'] + 'g </span><span class="ml-3">Fibre: ' + valuesss[0]['Fibre'] + 'g <span></p></div>')
                        
                        }})   
                }}
            </script>

            <script defer type="text/javascript">
                $('#export').on('click', function() {{
                    var titles = [];
                    var data = [];

                    $('#myTable thead th').each(function() {{
                        titles.push($(this).text());
                    }});

                    titles.push('Meal');
                    console.log(titles)

                    var table = $('#myTable');
                    var table1 = $('#myTable1');
                    var table2 = $('#myTable2');

                    table.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Breakfast');
                        data.push(row); 
                    }});

                    table1.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Lunch');
                        data.push(row); 
                    }});

                    table2.DataTable().rows('.selected').every(function (i, el){{
                        var row = [];
                        var valuesss = new Array();
                        this.invalidate();
                        var Row=this.data();
                        row.push(Row[1], Row[3], Row[4], Row[5], Row[6], Row[7], Row[8]);         
                       
                        console.log(row);
                
                        row.push('Dinner');
                        data.push(row); 
                    }});

                    console.log(data)
                    
                    csvFileData = data;
                    var csv = 'Volume (g), Food_items, Calories, Fats, Proteins, Carbohydrates, Fibre, Meal\\n'; 

                    csvFileData.forEach(function(row) {{
                        csv += row.join(',');  
                        csv += "\\n";  
                    }});  

                    var hiddenElement = document.createElement('a');  
                    hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);  
                    hiddenElement.target = '_blank';  
                    
                    hiddenElement.download = 'Diet Plan Meal';  
                    hiddenElement.click();  
                }});
            </script>

            <script defer type="text/javascript">
                $(document).ready(function() {{
                $('table.table-striped').dataTable( {{
                    stateSave: true,
                    responsive: true,
                    "bPaginate": false,
                    "bInfo": false,
                }});
            }});
            </script>

        </html>"""
                                )

    output_html = template.render(lunch_dataframe=lunch_df.to_html(classes='table table-striped', header="true", table_id="myTable1", escape=False ,formatters=dict(Image=path_to_image_html)),
                breakfast_dataframe=breakfast_df.to_html(classes='table table-striped', header="true", table_id="myTable", escape=False ,formatters=dict(Image=path_to_image_html)),
                dinner_dataframe=dinner_df.to_html(classes='table table-striped', header="true", table_id="myTable2", escape=False ,formatters=dict(Image=path_to_image_html)))

    components.html(output_html,height=3700)  

def Predict():
    FoodItemIDData = food_data()

    FoodNutrion = FoodItemIDData

    FoodItemIDData=FoodItemIDData.to_numpy()
  
    foodlbs = cluster_food(FoodItemIDData, FoodNutrion)
    FoodNutrion = FoodNutrion.drop(['Food_items'], axis = 1)
    FoodNutrion = FoodNutrion.astype('float32')
    labels = np.array(FoodNutrion['KMCluster'])
    features= FoodNutrion.drop(['KMCluster','VegNovVeg','Iron', 'Calcium', 'Sodium', 'Potassium','VitaminD','Sugars'], axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)
    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state=42)
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    my_scaler = StandardScaler()
    my_imputer = SimpleImputer(strategy="median")

    clf_RF = RandomForestClassifier(random_state=42)

    pipe = Pipeline([('imputer', my_imputer), ('rf_model',clf_RF)])

    param_grid = {
        'rf_model__n_estimators' : [50,100,200],
        'rf_model__max_features' : [0.8,"auto"],
        'rf_model__max_depth' : [4,5]
    }

    grid = GridSearchCV(pipe, cv=5, param_grid=param_grid)

    grid.fit(train_features, train_labels)

    # st.write(grid.best_params_)

    model = pipe.set_params(**grid.best_params_).fit(train_features, train_labels)

    # y_pred = model.predict(test_features)
    # st.write('Test Score:', model.score(test_features, test_labels))

    # st.write(confusion_matrix(test_labels,y_pred))
    # st.text('Model Report:\n ' + classification_report(test_labels,y_pred))

    # important_feature_list = pd.DataFrame(model.steps[1][1].feature_importances_, index=feature_list).sort_values(by=0, ascending=False)
    # st.dataframe(important_feature_list)

    ## RANDOM SEARCH 
    # from sklearn.model_selection import RandomizedSearchCV
    # n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    # max_features = ['auto', 'sqrt']
    # max_depth = [int(x) for x in np.linspace(4, 110, num = 11)]
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
    # bootstrap = [True, False]
    # random_grid = {'n_estimators': n_estimators,
    #             'max_features': max_features,
    #             'max_depth': max_depth,
    #             'min_samples_split': min_samples_split,
    #             'min_samples_leaf': min_samples_leaf,
    #             'bootstrap': bootstrap
    #             }
    # rf = RandomForestClassifier()
    # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    # rf_random.fit(train_features, train_labels)
    # st.write(rf_random.best_params_)

    # def evaluate(model, test_features, test_labels):
    #     y_pred=model.predict(test_features)
    #     st.write("Model Accuracy:", model.score(test_features, test_labels))
        
    #     return model.score(test_features, test_labels)

    # st.write("Base Model Evaluate")
    # base_accuracy = evaluate(model, test_features, test_labels)
    # st.write("Best Random Model Evaluate")
    # best_random = rf_random.best_estimator_
    # random_accuracy = evaluate(best_random, test_features, test_labels)
    # st.write('Improvement random grid of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))



    ## GRID SEARCH
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {
    #     'max_depth': [15,20,25],
    #     'max_features': ['auto'],
    #     'min_samples_leaf': [2,4,8],
    #     'min_samples_split': [5,6,7],
    #     'n_estimators': [80, 100, 150, 200],
    #     'bootstrap': [True]
    # }
    # rf1 = RandomForestClassifier()
    # grid_search = GridSearchCV(estimator = rf1, param_grid = param_grid, 
    #                         cv = 3, n_jobs = -1, verbose = 2)
    # grid_search.fit(train_features, train_labels)
    # st.write(grid_search.best_params_)
    # best_grid = grid_search.best_estimator_
    # grid_accuracy = evaluate(best_grid, test_features, test_labels)
    # st.write('Improvement Best grid of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
    # print (f'Train Accuracy - : {rf_Grid.score(train_features, train_labels):.3f}')
    # print (f'Test Accuracy - : {rf_Grid.score(test_features,test_labels):.3f}')


    y_pred=model.predict([[float(food_calories),float(food_fat), float(food_protein), float(food_carb), float(food_fibre)]])
    # st.write(y_pred)


    # tree = model.steps[1][1].estimators_[5]
    # export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, class_names=['0','1','2'], rounded = True, proportion = False, 
    #             precision = 2, filled = True)

    # (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # graph.write_png('tree.png')

    if y_pred==1:
        st.subheader(food_name.upper())
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Volumn", "100 g")
        col2.metric("Calories", food_calories)
        col3.metric("Fat", str(food_fat) + ' g')
        col4.metric("Protein", str(food_protein) + ' g')
        col5.metric("Carbohydrate", str(food_carb) + ' g')
        col6.metric("Fibre", str(food_fibre) + ' g')
        st.info('LOW CALORIES, MOST SUITABLE FOR **WEIGHT LOSS** AND **MAINTENANCE**')
    if y_pred==0:
        st.subheader(food_name.upper())
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Volumn", "100 g")
        col2.metric("Calories", food_calories)
        col3.metric("Fat", str(food_fat) + ' g')
        col4.metric("Protein", str(food_protein) + ' g')
        col5.metric("Carbohydrate", str(food_carb) + ' g')
        col6.metric("Fibre", str(food_fibre) + ' g')
        st.info('HIGH PROTEIN, SUITABLE FOR **WEIGHT LOSS**, **WEIGHT GAIN** AND **MAINTENANCE**')
    if y_pred==2:
        st.subheader(food_name.upper())
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Volumn", "100 g")
        col2.metric("Calories", food_calories)
        col3.metric("Fat", str(food_fat) + ' g')
        col4.metric("Protein", str(food_protein) + ' g')
        col5.metric("Carbohydrate", str(food_carb) + ' g')
        col6.metric("Fibre", str(food_fibre) + ' g')
        st.info('HIGH CALORIES - HIGH CARBOHYDRATE & FAT, ONLY SUITABLE FOR **WEIGHT GAIN**')

    st.balloons()

def Meal_Plan_UI():
    global age
    global gender
    global weight
    global height
    global activity_level
    global diet_plan
    st.sidebar.subheader("How old are you?")
    age = st.sidebar.number_input('Enter your age!', min_value=0, step=1, key="age")

    st.sidebar.subheader("What is your gender?")
    gender = st.sidebar.selectbox('Choose your gender!', options=['Male','Female'], key="gender")

    st.sidebar.subheader("How much do you weight?")
    weight = st.sidebar.number_input('Enter your weight in kg!', min_value=0, step=1, key="weight")

    st.sidebar.subheader("How tall are you?")
    height = st.sidebar.number_input('Enter your height in cm!', min_value=0, step=1, key="height")

    st.sidebar.subheader("What is your activity level?")

    st.sidebar.markdown('* **Sedentary** (little or no exercise)')
    st.sidebar.markdown('* **Lightly active** (exercise 1-3 times/week	)')
    st.sidebar.markdown('* **Moderately active** (moderate exercise 6-7 times/week)')
    st.sidebar.markdown('* **Very active** (intense exercise 6-7 times/week)')
    st.sidebar.markdown('* **Extra active** (hard exercise 2 or more times per day)')

    activity_level = st.sidebar.select_slider('Choose your activity level!', options=[
        'Sedentary',
        'Lightly active',
        'Moderately active',
        'Very active',
        'Extra active'], key='activity_level')

    st.sidebar.subheader("What is your diet plan?")
    diet_plan = st.sidebar.radio(
    "Choose your diet plan!",
    ('Weight Loss', 'Weight Gain', 'Maintenance'), key='diet_plan')

    st.sidebar.subheader("Are you ready?")
    submitForm = st.sidebar.checkbox('Do it now!')
  
    if age and gender and weight and height and activity_level and diet_plan and submitForm:
        if diet_plan == 'Weight Loss':
            if 'plan_state' not in st.session_state:
                Weight_Loss_Plan()
        elif diet_plan == 'Weight Gain':
            if 'plan_state' not in st.session_state:
                Weight_Gain_Plan()
        elif diet_plan == 'Maintenance':
            if 'plan_state' not in st.session_state:
                Maintenance_Plan()
    
def Predict_UI():
    global food_calories
    global food_fat
    global food_protein
    global food_carb
    global food_fibre
    global food_name

    st.sidebar.subheader("Enter the food name!")
    food_name = st.sidebar.text_input("Enter the food name!", key="food_name")

    st.sidebar.subheader("Enter calories in 100g of food!")
    food_calories = st.sidebar.number_input("Enter the amount of calories in the food!", min_value=0.0, step=0.1, key="food_calories")

    st.sidebar.subheader("Enter the amount of fat in 100g of food!")
    food_fat = st.sidebar.number_input("Enter the grams of fat!", min_value=0.0, step=0.1, key="food_fat")

    st.sidebar.subheader("Enter the amount of protein in 100g of food!")
    food_protein = st.sidebar.number_input("Enter the grams of protein!", min_value=0.0, step=0.1, key="food_protein")

    st.sidebar.subheader("Enter the amount of carbohydrate in 100g of food!")
    food_carb = st.sidebar.number_input("Enter the grams of carbohydrate!", min_value=0.0, step=0.1, key="food_carb")

    st.sidebar.subheader("Enter the amount of fibre in 100g of food!")
    food_fibre = st.sidebar.number_input("Enter the grams of fibre!", min_value=0.0, step=0.1, key="food_fibre")

    st.sidebar.subheader("Are you ready?")
    submitForm = st.sidebar.checkbox('Do it now!')

    if submitForm:
        Predict()
    
def main():
    st.set_page_config(page_title='Eat Better Daily', page_icon="https://res.cloudinary.com/hoaibao232/image/upload/v1644550372/eatbetterdaily_alkw1o.png", layout="centered")
    state = _get_state()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
            
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .css-1d391kg {
            padding-left: 20px;
            padding-right: 20px;
            # padding-top: 43px;
        }
        .streamlit-expanderHeader {
            font-weight: 600;
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

    pages = {
        "Create meal plan": Meal_Plan_UI,
        "Check food for diet plan": Predict_UI,
    }

    if "page" in st.session_state:
        st.session_state.update(st.session_state)

    else:
        st.session_state.update({
            # Default page
            "page": "Create meal plan",
        })


    with st.sidebar:
        st.sidebar.title('Eat Better Daily')
        page = st.radio("What you want", tuple(pages.keys()))

    pages[page]()

    state.sync()

if __name__ == "__main__":
    main()