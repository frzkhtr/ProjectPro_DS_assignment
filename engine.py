import streamlit as st
import pandas as pd
import numpy as np
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
#from pylab import rcParams
#from imblearn.combine import SMOTEENN, SMOTETomek
#from imblearn.over_sampling import SMOTE, ADASYN
#from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import xgboost

def preprocess(data):
    data.dropna(inplace = True)
    data['mail_subscribed'] = data['mail_subscribed'].apply(lambda x: '0' if x not in ['no', 'yes'] else x)
    data.drop(data[data['mail_subscribed'] == '0'].index, inplace = True)
    data.drop(data[data['maximum_days_inactive'] == 'NO'].index[0], inplace = True)
    data['maximum_days_inactive'] = data['maximum_days_inactive'].apply(lambda x : int(x))
    data['churn'] = data['churn'].apply(lambda x : int(x))
    encode_col = ['multi_screen', 'mail_subscribed', 'gender']
    encode = LabelEncoder()
    for ec in encode_col:
        data[ec] = encode.fit_transform(data[ec])
    data.reset_index(drop = True, inplace = True)
    return data
def data_split(data, splt, remove = False):
    col = ['gender', 'age', 'no_of_days_subscribed', 'multi_screen', 'mail_subscribed', 'weekly_mins_watched', 'minimum_daily_mins', 'maximum_daily_mins',
       'weekly_max_night_mins', 'videos_watched', 'maximum_days_inactive', 'customer_support_calls', 'churn']
    data = data[col].copy()
    if remove:
        data.drop(['gender', 'age'], axis = 1, inplace = true)
    X = data.drop(columns = ['churn'])
    y = data['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(1 - splt), random_state=42)
    return X_train, X_test, y_train, y_test



def count_value(column):
    c = df[column].value_counts().to_frame()
    c.columns = ['Count']
    st.bar_chart(c)

def show_des(data):
    st.write(data.describe())

df = pd.read_csv('Data.csv')
col_name = None

st.title('Project Pro assignment: Predicting Churn')
st.write('Raw Data EDA:')
if st.button('Check basic Stats', key = '1'):
    show_des(df)
col = df.columns
col_name = st.selectbox('Select a column to see its distribution: ', col, key = '3')
if col_name != 'year':
    count_value(col_name)


st.write('processed data EDA:')
df_processed = preprocess(df)
if st.button('Check basic Stats', key = '2'):
    show_des(df_processed)
col = df_processed.columns
col_name = st.selectbox('Select a column to see its distribution: ', col, key = '4')
if col_name != 'year':
    count_value(col_name)

st.write('Modelling :')
splt = st.slider('Select train test split(Default - 0.20):', 0.05, 0.30, 0.20)
X_train, X_test, y_train, y_test = data_split(df_processed, splt)
st.write('model used: XGBoost classifier')
global clf_d
if st.button('Run XGB classifier without hyper parameter Tuning', key = 'WH'):
    clf_d = xgboost.XGBClassifier()
    clf_d.fit(X_train, y_train)
    st.write(f'Accuracy on Train Data : {clf_d.score(X_train, y_train)}')
    st.write(f'Accuracy on Test Data: {clf_d.score(X_test, y_test)}')
    y_pred = clf_d.predict(X_test)
    st.write('Classification report: ')
    st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))
    st.write('Confusion Matrix: ')
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))

if st.button('Run XGB classifier with hyper parameter tuning', key = 'h'):
    #XGBoost
    #hyper-parameter Tuning

    #Providing a list of values for some xgboost model parameter
    params = {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [5, 7, 9, 11 , 13],
        'min_child_weight': [1, 2, 3, 5,7, 9],                            
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
        'n_estimators': [200,400,600,800],
        'scale_pos_weight': [1, 5, 10, 20, 25, 50, 75, 100]
        }


    model = xgboost.XGBClassifier()                #Defining the model

    #Defining the Randomized search Cv function with its own parameter and evaluation metrix
    random_search = RandomizedSearchCV(model, param_distributions = params, n_iter = 30, 
                                   scoring = 'f1', n_jobs = -1, cv = 5, verbose = 3)

    random_search.fit(X_train,y_train)             #fitting the random search model

    st.write(random_search.best_estimator_)                 #getting the best parameter value for xg boost
    clf_d = random_search.best_estimator_
    clf_d.fit(X_train, y_train)
    st.write(f'Accuracy on Train Data : {clf_d.score(X_train, y_train)}')
    st.write(f'Accuracy on Test Data: {clf_d.score(X_test, y_test)}')
    y_pred = clf_d.predict(X_test)
    st.write('Classificaytion report: ')
    st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))
    st.write('Confusion Matrix: ')
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))

if st.button('See Explainable AI Dashboard', key = 'eai'):
    #XGBoost
    #hyper-parameter Tuning

    #Providing a list of values for some xgboost model parameter
    params = {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [5, 7, 9, 11 , 13],
        'min_child_weight': [1, 2, 3, 5,7, 9],                            
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
        'n_estimators': [200,400,600,800],
        'scale_pos_weight': [1, 5, 10, 20, 25, 50, 75, 100]
        }


    model = xgboost.XGBClassifier()                #Defining the model

    #Defining the Randomized search Cv function with its own parameter and evaluation metrix
    random_search = RandomizedSearchCV(model, param_distributions = params, n_iter = 30, 
                                   scoring = 'f1', n_jobs = -1, cv = 5, verbose = 3)

    random_search.fit(X_train,y_train)             #fitting the random search model

    st.write(random_search.best_estimator_)                 #getting the best parameter value for xg boost
    clf_d = random_search.best_estimator_
    clf_d.fit(X_train, y_train)
    #st.button(key = 'h')
    explainer = ClassifierExplainer(clf_d, X_test, y_test)
    ExplainerDashboard(explainer).run()

