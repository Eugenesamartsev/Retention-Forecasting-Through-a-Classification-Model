#!/usr/bin/env python
# coding: utf-8

# # Retention Forecasting Through a Classification Model

# The purpose of this project is to predict the users who are most likely to fail an online training course based on data about their performance. To do this, I will use 3 different classification models and compare their prediction accuracy and ROC-AUC.

# ### Importing libraries

# In[184]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report,  accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc 


# In[185]:


sns.set(rc ={'figure.figsize' : (9, 6)})


# ### Reading the data sets

# In[186]:


submissions_data = pd.read_csv(r"C:\Users\SilkRIT\Desktop\разное\ML_stepik\Проект ML\submissions_data_train.csv")
events_data = pd.read_csv(r"C:\Users\SilkRIT\Desktop\разное\ML_stepik\Проект ML\event_data_train.csv")


# In[187]:


submissions_data.head()


# In[188]:


events_data.head()


# ### Data preparation

# converting data format

# In[189]:


events_data['date'] = pd.to_datetime(events_data['timestamp'], unit ='s')
submissions_data['date'] = pd.to_datetime(submissions_data['timestamp'], unit ='s')


# checking when course started and the end date of data set

# In[190]:


events_data.date.min()


# In[191]:


events_data.date.max()


# In[192]:


events_data['day'] = events_data.date.dt.date

submissions_data['day'] = submissions_data.date.dt.date


# let's see the course attendance

# In[193]:


events_data.groupby('day') \
.user_id.nunique().plot()


# let's see the distribution of users by open steps of the course

# In[194]:


events_data.pivot_table(index ='user_id',
                        columns = 'action',
                        values = 'step_id',
                        aggfunc = 'count',
                        fill_value = 0).reset_index().discovered.hist()


# now I create the table with distribution of users by submission status of the course

# In[195]:


users_scores = submissions_data.pivot_table(index ='user_id',
                        columns = 'submission_status',
                        values = 'step_id',
                        aggfunc = 'count',
                        fill_value = 0).reset_index()


# Now I want to evaluate the period of user's absence on course, after which we consider that he left it. To do this I will create the distribution of gaps between visits.    

# In[196]:


events_data[['day','user_id','timestamp']].drop_duplicates(subset = ['user_id', 'day']).head()


# In[197]:


gap_data = events_data[['day','user_id','timestamp']].drop_duplicates(subset = ['user_id', 'day']) \
.groupby('user_id')['timestamp'].apply(list)\
.apply(np.diff).values


# In[198]:


gap_data = pd.Series(np.concatenate(gap_data,axis = 0))


# In[199]:


gap_data = gap_data /(24*60*60)


# In[200]:


gap_data


# In[201]:


gap_data[gap_data < 200].hist()


# Now we see 90% of users leave the course after 18 days

# In[202]:


# 90% покатдают курс после простоя в 18 дней

gap_data.quantile(0.90)


# 95% leave after 60 days

# In[203]:


# 95% покатдают курс после простоя в 60 дней

gap_data.quantile(0.95)


# For simplicity, let's take a period of 30 days

# In[204]:


drop_out_threshold = 30*24*60*60 


# In[205]:


events_data.tail()


# In[206]:


# last timestamp
now = 1526772811


# next I add a column with information whether the user has left or not

# In[207]:


users_data = events_data.groupby('user_id', as_index = False) \
.agg({'timestamp' : 'max'}).rename(columns = {'timestamp' : 'last_timestamp'})


# In[208]:


users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold


# In[209]:


users_data.head()


# In[210]:


users_scores.head()


# now wee need to join tables with users data and their scores 

# In[211]:


users_data = users_data.merge(users_scores, on ='user_id', how = 'outer')


# In[212]:


users_data.head()


# In[213]:


users_data = users_data.fillna(0)
users_data.head(10)


# In[214]:


users_events_data = events_data.pivot_table(index ='user_id',
                        columns = 'action',
                        values = 'step_id',
                        aggfunc = 'count',
                        fill_value = 0).reset_index()


# adding the count of actions

# In[215]:


users_data = users_data.merge(users_events_data, on ='user_id', how = 'outer')


# adding the count of unique days on the course 

# In[216]:


users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index()


# In[217]:


users_days.head()


# In[218]:


users_data = users_data.merge(users_days, on ='user_id', how = 'outer')


# In[219]:


users_data.head(10)


# Now i check whether user passed the course or not. In our case the course is considered completed if the user has completed 170 steps.

# In[220]:


count_of_courses = 170
users_data['passed_course'] = users_data.passed > count_of_courses


# In[221]:


users_data.head(10)


# checking the completness of data

# In[222]:


users_data.user_id.nunique() == events_data.user_id.nunique() 


# In[223]:


users_data.groupby('passed_course').count()


# Percentage of course completion

# In[224]:


users_data['passed_course'].value_counts(normalize=True)


# In next following steps I will form the data for modeling. According to the purpose of the project I will collect the information of the users behaviour for first 3 days after they started the course with indication whether they passed the course or not.

# In[225]:


user_min_time = events_data.groupby('user_id',as_index = False) \
.agg({'timestamp' : 'min'}) \
.rename({'timestamp' : 'min_timestamp'},axis =1)


# In[226]:


users_data = users_data.merge(user_min_time, how = 'outer')


# In[227]:


users_data.head()


# In[228]:


events_data.shape


# In[229]:


events_data['user_time'] = events_data.user_id.map(str) + "_" + events_data.timestamp.map(str) 


# In[230]:


events_data.shape


# 3 days for performance analyzing

# In[231]:


learning_time_treshold = 3*24*60*60


# In[232]:


user_learning_time_treshold = user_min_time.user_id.map(str) + "_" + (user_min_time.min_timestamp + learning_time_treshold).map(str)


# In[233]:


user_min_time['user_learning_time_treshold'] = user_learning_time_treshold
user_min_time.head()


# In[234]:


events_data = events_data.merge(user_min_time[['user_id', 'user_learning_time_treshold']], how = 'outer')


# In[235]:


events_data.head()


# In[236]:


events_data_train = events_data[events_data.user_time <= events_data.user_learning_time_treshold]


# In[237]:


events_data_train.head()


# In[238]:


events_data_train.groupby('user_id').day.nunique().max()


# In[239]:


submissions_data.head()


# In[240]:


#Добавляем в submissions_data время первого степа
submissions_data = submissions_data.merge(user_min_time, on='user_id', how='left')
# Время от первого степа до последнего
submissions_data['users_time'] = submissions_data['timestamp'] - submissions_data['min_timestamp']
#Выбираем степы первых трёх дней
submissions_data_train = submissions_data[submissions_data.users_time <= 3*24*60*60]
submissions_data_train.groupby('user_id').day.nunique().max()


# In[241]:


X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index() \
.rename(columns = {'day' : 'days'})


# In[242]:


X.head()


# In[243]:


steps_tried = submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index() \
.rename(columns = {'step_id' : 'steps_tried'})


# In[244]:


steps_tried.head()


# In[245]:


X = X.merge(steps_tried, on = 'user_id', how = 'outer')


# In[246]:


X.head()


# In[247]:


X.shape


# In[248]:


X = X.merge(submissions_data_train.pivot_table(index ='user_id',
                        columns = 'submission_status',
                        values = 'step_id',
                        aggfunc = 'count',
                        fill_value = 0).reset_index(), on = 'user_id', how ='outer')


# In[249]:


X['correct ratio'] = X.correct / (X.correct + X.wrong)


# In[250]:


X = X.merge(events_data_train.pivot_table(index ='user_id',
                        columns = 'action',
                        values = 'step_id',
                        aggfunc = 'count',
                        fill_value = 0).reset_index()[['user_id','viewed']], how ='outer')


# In[251]:


X= X.fillna(0)


# In[252]:


users_data.head()


# In[253]:


X = X.merge(users_data[['user_id','passed_course', 'is_gone_user']], how = 'outer')


# In[254]:


X.head()


# In[255]:


X = X[~((X.is_gone_user == False) & (X.passed_course == False))]


# In[256]:


X.head()


# In[257]:


X.groupby(['passed_course','is_gone_user']).user_id.count()


# **Target column**

# In[258]:


y = X['passed_course']


# In[259]:


y = y.map(int)


# In[260]:


y.head()


# In[261]:


X = X.drop(['passed_course','is_gone_user'], axis = 1)


# In[262]:


X = X.set_index(X.user_id)
X = X.drop('user_id', axis =1 )


# **Variables**

# In[263]:


X.head()


# In[264]:


X


# ### Modeling

# In our case I will use 3 models:
# - Random Forest
# - k Nearest Neighbor
# - XGBoost
# 
# Also I will split our data set to train and test data. For model tuning I will use GridSearchCV technique and compare models by accuracy score and ROC-AUC.

# In[265]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[266]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[267]:


print(classification_report(y_test, pred_rfc))


# In[268]:


print(confusion_matrix(y_test, pred_rfc))


# In[269]:


rfc.score(X_test,y_test)


# In[270]:


parametrs = {'n_estimators' : range(10, 50, 10), 'max_depth' : range(1, 12, 2), 'min_samples_leaf' : range(1,5),
            'min_samples_split' : range(1,10)}
grid_rfc = GridSearchCV(rfc, param_grid=parametrs, scoring='accuracy', cv=10)


# In[271]:


grid_rfc.fit(X_train, y_train)


# In[272]:


pred_rfc2 = grid_rfc.predict(X_test)
print(classification_report(y_test, pred_rfc2))


# In[273]:


print(confusion_matrix(y_test, pred_rfc2))


# In[274]:


grid_rfc.best_params_


# In[275]:


grid_rfc.score(X_test,y_test)


# In[276]:


knn=KNeighborsClassifier()
k_range = list(range(1, 30))
param_grid = dict(n_neighbors=k_range)
knn_grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)


# In[277]:


knn_grid.fit(X_train,y_train)


# In[278]:


knn_grid.best_params_


# In[279]:


pred_knn_grid = knn_grid.predict(X_test)
print(classification_report(y_test, pred_knn_grid))


# In[280]:


print(confusion_matrix(y_test, pred_knn_grid))


# In[281]:


knn_grid.score(X_test,y_test)


# In[282]:


mean_score = cross_val_score(knn_grid, X_test, y_test, scoring="roc_auc", cv = 7).mean()


# In[283]:


predproba_knn_grid = knn_grid.predict_proba(X_test)


# In[284]:


predproba_knn_grid = predproba_knn_grid[:,1]


# In[285]:


knn_grid_auc = roc_auc_score(y_test, predproba_knn_grid)
print('KNN: ROC AUC=%.3f' % (knn_grid_auc))


# In[286]:


fpr, tpr, treshold = roc_curve(y_test, predproba_knn_grid)


# In[287]:


roc_auc = auc(fpr, tpr)


# In[288]:


plt.plot(fpr, tpr, color='darkorange',
         label='ROC-curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-curve KNN')
plt.legend(loc="lower right")
plt.show()


# In[289]:


predproba_grid_rfc = grid_rfc.predict_proba(X_test)
predproba_grid_rfc = predproba_grid_rfc[:,1]
grid_rfc_auc = roc_auc_score(y_test, predproba_grid_rfc)
print('KNN: ROC AUC=%.3f' % (grid_rfc_auc))


# In[290]:


fpr, tpr, treshold = roc_curve(y_test, predproba_grid_rfc)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
         label='ROC-curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-curve RFC')
plt.legend(loc="lower right")
plt.show()


# In[291]:


xgbc = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}


# In[292]:


grid_search_xgbc = GridSearchCV(
    estimator=xgbc,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)


# In[293]:


grid_search_xgbc.fit(X_train,y_train)


# In[294]:


grid_search_xgbc.best_estimator_


# In[295]:


predproba_grid_search_xgbc = grid_search_xgbc.predict_proba(X_test)
predproba_grid_search_xgbc = predproba_grid_search_xgbc[:,1]
grid_search_xgbc_auc = roc_auc_score(y_test, predproba_grid_search_xgbc)
print('grid_search_xgbc: ROC AUC=%.3f' % (grid_search_xgbc_auc))


# In[296]:


fpr, tpr, treshold = roc_curve(y_test, predproba_grid_search_xgbc)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
         label='ROC-curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-curve XGBC')
plt.legend(loc="lower right")
plt.show()


# In[297]:


grid_rfc.score(X_test,y_test)


# In[298]:


xgbc_pred = grid_search_xgbc.predict(X_test)
xgbc_acc = accuracy_score(y_test, xgbc_pred)


# In[299]:


models = pd.DataFrame({'model': ['Random forest', 'k Nearest Neighbor', 'XGBoost'],
                       'Score': [grid_rfc.score(X_test,y_test),knn_grid.score(X_test,y_test),xgbc_acc],
                      'ROC_AUC' : [grid_rfc_auc,knn_grid_auc,grid_search_xgbc_auc]
                      })


# Here you can see the accuracy score and ROC-AUC of 3 models. According to evaluation the most accurate model is **XGBoost model**. 

# In[300]:


models


# ### Conclusion

# In this project, you can see the complete process of building a classification model, including:
# - Data Obtaining
# - Data Cleaning
# - Data Visualizing
# - Data Modeling
# - Data Interpreting
# 
# The resulting model helped the company increase retention by identifying most likely exiting users and targeting them.

# In[ ]:




