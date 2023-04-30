#print(1)
#pip show catboost
#pip install Pillow
#name 'Image' is not defined
#from streamlit import Image
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pickle
import base64
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

st.title("Best Hack 2023")
st.text("Produced by: Irisky")
st.text("Please, waiting for loading :)")

#image = Image.open('label.jpeg')




#displaying the image on streamlit app

#st.image(image, caption='Enter any caption here')



train_data = pd.read_parquet('train.parquet')
test_data = pd.read_parquet('test.parquet')
train_target = pd.read_csv('train_target.csv')
#test_submit_example = pd.read_csv('test_submit_example.csv')
# принимает весь массив
# def get_day_type(day):
# #     print(day.lower())
#     weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
#     if day in weekdays:
#     #np.array(weekdays).isin(day.lower()):
#         return 0
#     elif day in ['Saturday','Sunday']:
#         return 1
#     else:
#         return None
def prev_dest_nan_filler(train_data):
    # 1. Найдем поезда с пропущенным значением prev_distance
    one_tmp = train_data[train_data['prev_distance'].isna()][['prev_distance', 'rsv_st_id']]
    # 2. найдем все поезда, у которых та же rsv_st_id и заполнено prev_distance
    all_tmp = train_data[(train_data['rsv_st_id'].isin(one_tmp['rsv_st_id']) == True)&(train_data['prev_distance'].isna() == False )]
    # 3. сгруппируем по rsv_st_id и найдем среднее значение prev_distance
    prev_distance_mean = all_tmp.groupby('rsv_st_id')['prev_distance'].mean().reset_index()
    # 4
    replace_dict = dict(zip(prev_distance_mean['rsv_st_id'], prev_distance_mean['prev_distance']))
    #train_data.fillna({'prev_distance': replace_dict.set_index('rsv_st_id')['prev_distance']})
    train_data['prev_distance'] = train_data['prev_distance'].fillna(train_data.rsv_st_id.map(replace_dict))
    return train_data

def dist_nan_filler(dta):
    # 1. Найдем поезда с пропущенным значением distance
    one_tmp = dta[dta['distance'].isna()][['distance', 'snd_st_id']]
    # 2. найдем все поезда, у которых та же snd_st_id  и заполнено distance
    all_tmp = dta[(dta['snd_st_id'].isin(one_tmp['snd_st_id']) == True)&(dta['distance'].isna() == False)]
    
    # 3. сгруппируем по snd_st_id и найдем среднее значение distance
    distance_mean = all_tmp.groupby('snd_st_id')['distance'].mean().reset_index()
    # 4
    replace_dict = dict(zip(distance_mean['snd_st_id'], distance_mean['distance']))
    #train_data.fillna({'prev_distance': replace_dict.set_index('rsv_st_id')['prev_distance']})
    dta['distance'] = dta['distance'].fillna(dta.snd_st_id.map(replace_dict))
    knn = KNNImputer(n_neighbors = 5)
    return np.concatenate(knn.fit_transform(dta[['distance']])).tolist()

def hours_groupping(nums):
    result = []
    for num in nums:
        # Определяем значение для текущего числа
        if num < 6:
            val = 0
        elif num < 12:
            val = 1
        elif num < 18:
            val = 2
        else:
            val = 3
        result.append(val)
    return result
def group(x):
    if x in ['АППАР РАЗЛ НАЗН', 'ОБОРУД КИНЕМАТ', 'ИЗД ЭЛБЫТ ПР', 'ОБОР РАЗЛИЧ', 'ОБОР ПИЩПРОМ', 'КОНТ УН РЕГ', 'ЛОШАДИ']:
        x = 'ОБОРУДОВАНИЕ И МАШИНЫ'
    elif x in ['ШКАФЫ ПР', 'ОБОРУД ТОРГОВ', 'НОСИЛКИ САНИТАР', 'МЕБЕЛЬ ПР', 'МЕБЕЛЬ МЯГКАЯ', 'МАТРАЦЫ ПРУЖИН', 'МЕБЕЛЬ ДЕТСКАЯ', 'СТОЛЫ ПР', 'КРЕСЛА ПР', 
                            'ГАРНИТУРЫ КУХОН', 'БАНКЕТКИ', 'БУФЕТЫ', 'КРОВАТИ ДЕР ВС', 'ГАРНИТУРЫ ПР', 'ГАРНИТУРЫ СПАЛ', 'ДИВАНЫ', 'ДИВАНЫ-КРОВАТИ', 'ЗАГОТОВКИ ЩИТ', 'ЯЩИКИ ОБУВ ИГР', 'КРЕСЛА-КРОВАТИ', 'МЕБЕЛЬ КР МЕТАЛ']:
        x = 'МЕБЕЛЬ'
    elif x in ['ОРЕХИ КЕДРОВЫЕ', 'ОРЕХИ ЛЕЩИННЫЕ', 'ОРЕХИ МИНДАЛЬН', 'ОРЕХИ ПР', 'ЯДРА ОРЕХ ФРУКТ']:
        x = 'ОРЕХИ'
    elif x in ['АЛИГНИН', 'КАРТОН КОРОБОЧН', 'КАРТОН КРОВ НЕП', 'КАРТОН ОБЛИЦОВ', 'КАРТОН ОБУВН', 'КАРТОН ПЕРЕПЛЕТ', 'БУМАГА ЦВЕТН', 'БУМАГА ПР', 'БУМАГА ПИСЧ', 'БУМАГА ОБОЙН', 'БУМАГА КОПИРОВ', 'БУМАГА ДЕКОРАТ', 'БУМАГА Д/ПЕЧАТИ', 'БУМАГА ГАЗЕТН', 'БУМАГА АФИШН', 'БУМАГА АЛЬБОМН', 'КАРТОН ПР', 'КАРТОН ПРОКЛАД', 'КАРТОН СТР ПР', 'КАРТОН ТАРН', 'КАРТОН ТЕХНИЧ', 'БУМАГА ИНДИКАТ', 'ПЕРГАМЕНТ', 'ПАПКИ ОБЛОЖКИ','ПРОКЛАДКИ Д/ЯИЦ','ИЗДЕЛИЯ КАРТ ПР','ИЗДЕЛИЯ БУМ ПР','ПАКЕТЫ БУМ', 'ИЗДЕЛИЯ ФИБР ПР','ГИЛЬЗЫ ПАПИРОСН', 'ВАТА ЦЕЛЛЮЛОЗН', 'ЦЕЛЛЮЛОЗА СУЛЬФ', 'БУМАГА КАРТОН', 'ИЗД БУМ КАРТ']:
        x='БУМАГА И КАРТОН'
    elif x in ['ТАБАК СЫРЬЕ ФЕР', 'ОТХОДЫ ТАБАЧНЫЕ', 'ОТХОДЫ ТАБАЧНЫЕ', 'ТАБАК СЫРЬЕ ФЕР', 'СЫРЬЕ ТАБАК МАХ', 'ВОДОРОСЛИ ВС', 'КОРНИ ЛЕКАРСТ', 'ХМЕЛЬ', 'БОДЯГА', 'МОХ']:
        x='ТАБАК И ЛЕКАРСТВА'
    elif x in ['МУКА СЛАНЦЕВ']:
        x='Минерально-строит.'
    elif x in ['БЕНЗОЛ']:
        x='Нефтянные грузы'
    else: 
        x=np.nan
    return x

def prev_group(x):
    if x in ['ШКАФЫ ПР', 'ОБОРУД ТОРГОВ', 'НОСИЛКИ САНИТАР', 'МЕБЕЛЬ ПР', 'МЕБЕЛЬ МЯГКАЯ', 'МАТРАЦЫ ПРУЖИН', 'МЕБЕЛЬ ДЕТСКАЯ', 'СТОЛЫ ПР', 'КРЕСЛА ПР', 
                            'ГАРНИТУРЫ КУХОН', 'БАНКЕТКИ', 'БУФЕТЫ', 'КРОВАТИ ДЕР ВС', 'ГАРНИТУРЫ ПР', 'ГАРНИТУРЫ СПАЛ', 'ДИВАНЫ', 'ДИВАНЫ-КРОВАТИ', 'ЗАГОТОВКИ ЩИТ', 'ЯЩИКИ ОБУВ ИГР', 'КРЕСЛА-КРОВАТИ', 'МЕБЕЛЬ КР МЕТАЛ']:
        x = 'МЕБЕЛЬ'
    elif x in ['ОРЕХИ КЕДРОВЫЕ', 'ОРЕХИ ЛЕЩИННЫЕ', 'ОРЕХИ МИНДАЛЬН', 'ОРЕХИ ПР', 'ЯДРА ОРЕХ ФРУКТ']:
        x = 'ОРЕХИ' 
    elif x in ['КАРТОФЕЛЬ ПОЗДН', 'КАРТОФЕЛЬ РАН', 'АПЕЛЬСИНЫ СВЕЖ', 'ФРУКТЫ СВЕЖ ПР', 'ХУРМА СВЕЖ', 'КОРНЕПЛОДЫ СВЕЖ', 'ЧЕСНОК СВЕЖ', 'ЛУК РЕПЧАТЫЙ', 'КАПУСТА СВЕЖ', 'ОВОЩИ СВЕЖИЕ ПР', 'КАРТОФЕЛЬ СВЕЖ', 'КОНФЕТЫ ПР']:
        x='ЕДА'
    elif x in ['ЦВЕТЫ ИСКУС', 'СТЕБЛИ ДЖУТА', 'БЕЛЬЕ ПОСТЕЛЬН', 'ОДЕЖДА ПР', 'ТКАНИ ИСКУС СИН', 'ШКУРЫ НЕВ ПР', 'ХЛОПОК-СЫРЕЦ', 'ВАТА МЕД']:
        x='ТКАНИ'
    elif x in ['МУКА СЛАНЦЕВ']:
        x='Минерально-строит.'
    elif x in ['БЕНЗОЛ']:
        x='Нефтянные грузы'
    else: 
        x=np.nan
    return x


  
def __transfomate(data):
    #data['weekday_name'] = data['weekday_name'].apply(get_day_type)
    data = prev_dest_nan_filler(data)
    data['distance'] = dist_nan_filler(data)
    data['prev_date_arrival'] = data['prev_date_arrival'].fillna(data['prev_date_depart']+(data['prev_date_arrival'] - data['prev_date_depart']).mean())
    #print("проверка, что нанов нет в prev_distance: {0}".format(len(data[data['prev_distance'].isna()])))
    data['part_of_the_day'] = hours_groupping(data['prev_date_arrival'].dt.hour)  
    data['sum_of_is_load'] =((data['prev_is_load'] == 1)&(data['is_load']==1)).astype(int)
    data['quarter'] = data['prev_date_arrival'].dt.quarter
    data['weekday_name'] = data['prev_date_arrival'].dt.day_name()
    data['group'] = data.query('fr_group == "Остальные грузы"').freight.apply(lambda x: group(x))
    data['group'] = data['group'].fillna(data.fr_group)
    data['prev_group'] = data.query('prev_fr_group == "Остальные грузы"').prev_freight.apply(lambda x: prev_group(x))
    data['prev_group'] = data['prev_group'].fillna(data.prev_fr_group)
    
    return data#.drop(columns_to_drop,axis=1)

# Это делаем только для трейн!
train_data['target'] = train_target
train_data = train_data[train_data['target'].notnull()] # Удаляем строку с target null
train_data = train_data[train_data['target']>=0]
X = __transfomate(train_data).drop(['prev_date_arrival','prev_date_depart','rod','date_depart','prev_freight','freight', 'fr_group', 'prev_fr_group'],axis=1)
y = X['target']
X = X.drop(['target'], axis = 1)
test_data = __transfomate(test_data).drop(['prev_date_arrival','prev_date_depart','rod','prev_freight','freight', 'fr_group', 'prev_fr_group'], axis = 1)

#test_data = __transfomate(test_data).drop(['prev_date_arrival','prev_date_depart','rod','prev_freight','freight'], axis = 1)


for_distance = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors = 5)),
    ('scaler', MinMaxScaler())
])

for_prev_fr_group = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

for_common_ch = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore', dtype=int))])




for_weekday = Pipeline(steps=[
    ('onehotencoder', OrdinalEncoder()),
    ('minmax', MinMaxScaler()) 
])


CT = ColumnTransformer([
        
        ('distance', for_distance, ['prev_distance','distance']),
        ('vidsobst', OrdinalEncoder(), ['vidsobst']),
        ('group', for_prev_fr_group, ['prev_group','group']),
        ('common_ch', for_common_ch, list(map(str, ['common_ch']))),
        ('weekday', for_weekday, ['weekday_name']),
         ('weekday_name', OneHotEncoder(), ['quarter','part_of_the_day'] )
        
        ],  remainder = MinMaxScaler())

X_train,X_test2,y_train,y_test2 = train_test_split(X,y, train_size = 0.001, test_size = 0.001, random_state = 33)


pipeline = Pipeline([
    ('scaler', CT),
    ('regressor', CatBoostRegressor(iterations=2000, learning_rate=0.01, depth=10, loss_function='MAE', random_seed=42))
])
def change_flag():
    st.session_state["flag"] = 1
uploaded_file2 = st.file_uploader("Choose a file to get prediction", on_change=change_flag())

def arr_to_string(arr):
    strstr = ''
    for i in arr:
        strstr=strstr+'\n'+str(i)
    return strstr


X_test = None
y_pred = None
if uploaded_file2 is not None:
    X_test = pd.read_parquet(uploaded_file2)
    if (X_test is not None and st.session_state["flag"] ==1):
        pipeline.fit(X_train,y_train)
        test_data = __transfomate(X_test).drop(['prev_date_arrival','prev_date_depart','rod','prev_freight','freight'], axis = 1)
        #st.write(test_data.isna().sum())
        y_pred = pipeline.predict(test_data)

        # Создаем два столбца: левый и правый
        left_column, right_column = st.columns(2)
        with right_column:
            button = st.download_button(
            label="Download text file",
            data= arr_to_string(y_pred) if y_pred is not None else 'No Data. \nWith love, Irisky.',
            file_name='iriski_y_pred.txt',
            mime='text'
            )  
        with left_column:
            st.write("y predicted:")
            st.write(y_pred)
        st.session_state["flag"] = 0





# Помещаем кнопку в правый столбец

      
        
