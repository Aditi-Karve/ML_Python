# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:58:44 2021

@author: Aditi
"""
import os
os.chdir("C:\\Users\\Tanay\\Desktop\\Iqbal Project")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('aug_train.csv')
test = pd.read_csv('aug_test.csv')
train.describe()
train.describe(include = 'all')
train.info()
test.shape
#Out[185]: (2129, 13)
type(test)
#Out[187]: pandas.core.frame.DataFrame
train.shape
#Out[186]: (19158, 14)
type(train)
#Out[187]: pandas.core.frame.DataFrame
train.dtypes

df1 = train

"""
EDA STARTS HERE

"""

#Out[184]: (19158, 14)
#Variabel 1 - Response Variable target
df1.target.describe()
"""
count    19158.000000
mean         0.249348
std          0.432647
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max          1.000000
Name: target, dtype: float64
"""
plt.hist(df1.target)
sns.boxplot(df1.target)
df1.target.value_counts()
"""
0.0    14381
1.0     4777
Name: target, dtype: int64
14381+4777 = 19158 -- No Outliers and only 2 categories
"""
df1.dtypes


#Variable 2 - city
plt.hist(df1.city)
c = df1.city.value_counts()
sum(df1.city.value_counts())
#19158 - no missing values
#This column seem to have diverse values for various cities so it can be dropped for now
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
ct1 = pd.crosstab(df1['city'], df1['target']) 
ct1
c, p, dof, expected = chi2_contingency(ct1)
p # > 0.05 which shows not significant variable

df1.dtypes
#Variable 3 - Continuous variable city_development_index
sum(df1.city_development_index.value_counts())
#Out[226]: 19158 - no NA values
plt.hist(df1.city_development_index)
#Right Skewed histogram
sns.boxplot(df1.city_development_index)
#Outliers exists
df1.city_development_index.describe()
"""
count    19158.000000
mean         0.828848
std          0.123362
min          0.448000
25%          0.740000
50%          0.903000
75%          0.920000
max          0.949000
Name: city_development_in
"""
#Lower limit = Q1 - 1.5*IQR
lower_limit = 0.448000 - 1.5*(0.92-0.74)
lower_limit #Out[241]: 0.17799999999999994
#Out[236]: 0.17799999999999994
len(df1.city_development_index[df1.city_development_index < 0.45])
#17 outliers but far away from lower limit , so from a common sense , 
#we can replace these 17 outliers by 0.48
df1.city_development_index[df1.city_development_index < 0.45] = 0.48
sns.boxplot(df1.city_development_index)
#No outliers now
plt.hist(df1.city_development_index)
##Do independent t test on this to check significance of this variable in
##predicting target
CDI_0 = df1[df1.target == 0]
CDI_1 = df1[df1.target == 1]
len(df1.target[df1.target == 0]) #14381
len(df1.target[df1.target == 1]) #4777
import scipy
scipy.stats.ttest_ind(CDI_0.target, CDI_1.target)
#Out[273]: Ttest_indResult(statistic=-inf, pvalue=0.0) - pvalue 0 - so significant

"""
This does not need anova but when I tried ANOVA it also gives same p value as t test :)
This was interesting
##Anova for city_development_index
from scipy import stats
cdi_anova = df1[['target','city_development_index']]
cdi_anova
grouped_anova2 = cdi_anova.groupby(['target'])
F_score, p = stats.f_oneway(grouped_anova2.get_group(0)["city_development_index"], 
                            grouped_anova2.get_group(1)["city_development_index"])
F_score
p #Out[274]: 0.0 - so significant
"""
df1.dtypes
#Variable 4 - gender
df1.gender.value_counts()
"""
Male      13221
Female     1238
Other       191
Name: gender, dtype: int64
(13221+1238+191) = 14650
19158 - 14650 = 4508 are NA values means 
4508/14650 = 0.3077133105802048 30% are NA values so treat those
"""
df1.gender[df1.gender == 'Other'] = 'Male_1'
df1.gender[df1.gender == 'Female'] = 'Male_1'
df1.gender.value_counts()
#4508 are NA values , impute with Mode
"""
This code did not work so I will try it later
from numpy import isnan 
from sklearn.impute import SimpleImputer
value = df1.gender
imputer = SimpleImputer(missing_values=nan,  
                        strategy='most_frequent') 
transformed_values = imputer.fit_transform(value)
numpy.isnan(df1[gender]) = 'Male'
"""
from scipy.stats import chi2_contingency
df1['gender'].fillna(df1['gender'].mode()[0],inplace=True)
plt.hist(df1.gender)
#Do Chi quare test to check its significance
gen = pd.crosstab(df1['gender'], df1['target']) 
gen
c, p, dof, expected = chi2_contingency(gen)
p #Out[96]: 0.2227444852962903 > 0.05 , not significant

df1.dtypes
#Variable 5
df1.relevent_experience.value_counts()
df1.relevent_experience.value_counts().sum()
#19158 - so no NA values
plt.hist(df1.relevent_experience)
#Now Chisquare to check its significance
rel = pd.crosstab(df1['relevent_experience'], df1['target']) 
rel
c, p, dof, expected = chi2_contingency(rel)
p #Out[106]: 1.5006628411178982e-70 significant

df1.dtypes
#Variable 6 - enrolled_university
df1.enrolled_university.value_counts()
df1.enrolled_university.value_counts().sum()
#18772 means there are NA values
19158 - 18772 #386 NA values , replace by mode
df1['enrolled_university'].fillna(df1['enrolled_university'].mode()[0],inplace=True)
df1.enrolled_university.value_counts()
plt.hist(df1.enrolled_university)
#Use chisquare test to check if the variable is significant
eu = pd.crosstab(df1['enrolled_university'], df1['target']) 
eu
c, p, dof, expected = chi2_contingency(eu)
p  #Out[123]: 2.267945402973493e-96 < 0.05 , very much significant variable

df1.dtypes
#Variable 7 - education_level
df1.education_level.value_counts()
df1.education_level.value_counts().sum()
#18698 means there are NA values
19158 - 18698 #460 NA values , replace by mode
df1['education_level'].fillna(df1['education_level'].mode()[0],inplace=True)
df1.education_level.value_counts()
plt.hist(df1.education_level)
#Use chisquare test to check if the variable is significant
el = pd.crosstab(df1['education_level'], df1['target']) 
el
c, p, dof, expected = chi2_contingency(el)
p #Out[137]: 1.168254108960399e-33 < 0.05 , very much significant

df1.dtypes
#Variable 8 - major_discipline
df1.major_discipline.value_counts()
df1.major_discipline.value_counts().sum()
#18698 means there are NA values
19158 - 16345 #2813 NA values , replace by mode
df1['major_discipline'].fillna(df1['major_discipline'].mode()[0],inplace=True)
df1.major_discipline.value_counts()
plt.hist(df1.major_discipline)
#Use chisquare test to check if the variable is significant
md = pd.crosstab(df1['major_discipline'], df1['target']) 
md
c, p, dof, expected = chi2_contingency(md)
p #Out[349]: 0.12236179432062115 not significant

df1.dtypes
#Variable 9 - experience
df1.experience.value_counts()
df1.experience.value_counts().sum()
#19093 means there are NA values
19158 - 19093 #65 NA values , replace by mean
#convert experience in numeric
df1['experience']=df1.get('experience').replace('>20','21')
df1['experience']=df1.get('experience').replace('<1','0')
df1['experience'] = df1['experience'].astype("float")
df1['experience'].replace(np.nan,21,inplace=True)
df1.experience.value_counts()
plt.hist(df1.experience)
sns.boxplot(df1.experience)
#Use independent t test to check if the variable is significant
exp_0 = df1[df1.target == 0]
exp_1 = df1[df1.target == 1]
len(exp_0)
len(exp_1)
import scipy
scipy.stats.ttest_ind(exp_0.experience, exp_1.experience)
#Out[372]: Ttest_indResult(statistic=24.49269196805569, pvalue=1.7868071617356925e-130)
# Significant variable

df1.dtypes
#Variable 10 - company_size
df1.company_size.value_counts()
df1.company_size.value_counts().sum()
#13220 means there are NA values
19158 - 13220 #5938 NA values , replace by mode
df1['company_size'].fillna(df1['company_size'].mode()[0],inplace=True)
df1.company_size.value_counts()
plt.hist(df1.company_size)
#Use chisquare test to check if the variable is significant
cs = pd.crosstab(df1['company_size'], df1['target']) 
cs
c, p, dof, expected = chi2_contingency(cs)
p #Out[388]: 7.971605523261e-124 - Significant variable

df1.dtypes
#Variable 11 - company_type
df1.company_type.value_counts()
df1.company_type.value_counts().sum()
#13018 means there are NA values
19158 - 13018 #6140 NA values , replace by mode
df1['company_type'].fillna(df1['company_type'].mode()[0],inplace=True)
df1.company_type.value_counts()
plt.hist(df1.company_type)
#Use chisquare test to check if the variable is significant
ct = pd.crosstab(df1['company_type'], df1['target']) 
ct
c, p, dof, expected = chi2_contingency(ct)
p #Out[402]: 3.782006631159789e-18 - Significant variable


df1.dtypes
#Variable 12 - last_new_job
df1.last_new_job.value_counts()
df1.last_new_job.value_counts().sum()
#18735 means there are NA values
19158 - 18735 #423 NA values , replace by mode
df1['last_new_job'].fillna(df1['last_new_job'].mode()[0],inplace=True)
df1.last_new_job.value_counts()
plt.hist(df1.last_new_job)
#Use chisquare test to check if the variable is significant
lj = pd.crosstab(df1['last_new_job'], df1['target']) 
lj
c, p, dof, expected = chi2_contingency(lj)
p #Out[413]: 1.3204936717164466e-28

df1.dtypes
#Variable 13 - training_hours
df1.training_hours.value_counts()
df1.training_hours.value_counts().sum()
19158 #No NA values
plt.hist(df1.training_hours)
#Use independent t test to check if the variable is significant
th_0 = df1[df1.target == 0]
th_1 = df1[df1.target == 1]
import scipy
scipy.stats.ttest_ind(th_0.training_hours, th_1.training_hours)
#Out[422]: Ttest_indResult(statistic=2.9870990541592386, pvalue=0.002819949452636266)
#p < 0.05 , so significant variable

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
#df1["city"] = ord_enc.fit_transform(df1[["city"]])
df1["gender"] = ord_enc.fit_transform(df1[["gender"]])
df1["relevent_experience"] = ord_enc.fit_transform(df1[["relevent_experience"]])
df1["enrolled_university"] = ord_enc.fit_transform(df1[["enrolled_university"]])
df1["education_level"] = ord_enc.fit_transform(df1[["education_level"]])
df1["major_discipline"] = ord_enc.fit_transform(df1[["major_discipline"]])
df1["experience"] = ord_enc.fit_transform(df1[["experience"]])
df1["company_size"] = ord_enc.fit_transform(df1[["company_size"]])
df1["company_type"] = ord_enc.fit_transform(df1[["company_type"]])
df1["last_new_job"] = ord_enc.fit_transform(df1[["last_new_job"]])
df1.dtypes

df1 = df1.astype({"gender":'int',
                    "relevent_experience":'int',
                    "enrolled_university":'int',
                    "education_level":'int',
                    "major_discipline":'int',
                    "company_size":'int',
                    "company_type":'int',
                    "last_new_job":'int',
                    "experience":'int'})

df1.dtypes
df1.target = df1.target.astype(int)
df1.dtypes

df1.target.value_counts()
"""
Out[656]: 
0    14381
1     4777
Name: target, dtype: int64
The data is quite imbalanced and need balancing
"""

#!pip install imbalanced-learn
import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X = df1[['last_new_job','major_discipline','company_size',
          'education_level','enrolled_university','relevent_experience',
          'city_development_index','experience','company_type','training_hours']]
y = df1['target']
X_sm, y_sm = sm.fit_resample(X, y)
len(y_sm)
len(X_sm)
X_sm.shape
print(f'''Shape of X before SMOTE: {X.shape}
Shape of X after SMOTE: {X_sm.shape}''')
print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100

X = X_sm
y = y_sm
len(X)
len(y)

## Split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0) 

#RANDOM FOREST #########################################################################
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    from scipy.stats import uniform as sp_randFloat
    from scipy.stats import randint as sp_randInt

param_grid = {"n_estimators": [int(x) for x in np.linspace(start = 100, stop = 300, num = 50)],
              "max_depth" : [int(x) for x in np.linspace(1, 50, num = 11)],
              "max_features" : ['auto', 'sqrt'],
              "min_samples_split" : [2, 5, 10],
              "min_samples_leaf" : [1, 2, 4],
              "bootstrap" : [True, False],
              "criterion":["gini","entropy"]}

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_jobs=-1)    

randm_src = RandomizedSearchCV(estimator=rf_classifier, param_distributions = param_grid,
                               cv = 5, n_iter = 10, n_jobs=-1)
randm_src.fit(X_train, y_train)

    print(" Results from Random Search " )
    print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)
    print("\n The best score across ALL searched params:\n", randm_src.best_score_)
    print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)

"""
print(" Results from Random Search " )
 Results from Random Search 

print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)

 The best estimator across ALL searched params:
 RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=214,
                       n_jobs=-1)

print("\n The best score across ALL searched params:\n", randm_src.best_score_)

 The best score across ALL searched params:
 0.7941676228010445

print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)

 The best parameters across ALL searched params:
 {'n_estimators': 214, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 20, 'criterion': 'gini', 'bootstrap': True}
 """
 
rf_final = RandomForestClassifier(n_estimators = 214,
                                 max_depth = 20,
                                 max_features = 'auto',
                                 min_samples_split = 5,
                                 min_samples_leaf = 1,
                                 criterion = 'gini',
                                 bootstrap = True)

trained_model = rf_final.fit(X_train,y_train)
print( "Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train)))
#Train Accuracy ::  0.9413707679603633
y_pred = trained_model.predict(X_test)
print( "Test Accuracy  :: ", accuracy_score(y_test, y_pred))
#Test Accuracy  ::  0.8044498522509995
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.8044498522509995
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

print('Log loss = {:.5f}'.format(log_loss(y_test, y_pred_proba)))
#Log loss = 0.64413
print('AUC = {:.5f}'.format(roc_auc_score(y_test, y_pred_proba)))
#AUC = 0.73052
print('Average Precision = {:.5f}'.format(average_precision_score(y_test, y_pred_proba)))
#Average Precision = 0.71298
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, y_pred)))
#Accuracy = 0.80445
print('Precision = {:.5f}'.format(precision_score(y_test, y_pred)))
#Precision = 0.80054
print('Recall = {:.5f}'.format(recall_score(y_test, y_pred)))
#Recall = 0.81517
print('F1 score = {:.5f}'.format(f1_score(y_test, y_pred)))
#F1 score = 0.80779

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.show()
