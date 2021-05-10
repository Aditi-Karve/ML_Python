# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:49:21 2021

@author: Aditi Karve
"""
import os
os.chdir('C:\\Users\\Tanay\\Desktop\\Students_grades')
import pandas as pd 
import numpy as np 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv('student-mat.csv')
df.info()
"""
RangeIndex: 395 entries, 0 to 394
Data columns (total 33 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   school      395 non-null    object
 1   sex         395 non-null    object
 2   age         395 non-null    int64 
 3   address     395 non-null    object
 4   famsize     395 non-null    object
 5   Pstatus     395 non-null    object
 6   Medu        395 non-null    int64 
 7   Fedu        395 non-null    int64 
 8   Mjob        395 non-null    object
 9   Fjob        395 non-null    object
 10  reason      395 non-null    object
 11  guardian    395 non-null    object
 12  traveltime  395 non-null    int64 
 13  studytime   395 non-null    int64 
 14  failures    395 non-null    int64 
 15  schoolsup   395 non-null    object
 16  famsup      395 non-null    object
 17  paid        395 non-null    object
 18  activities  395 non-null    object
 19  nursery     395 non-null    object
 20  higher      395 non-null    object
 21  internet    395 non-null    object
 22  romantic    395 non-null    object
 23  famrel      395 non-null    int64 
 24  freetime    395 non-null    int64 
 25  goout       395 non-null    int64 
 26  Dalc        395 non-null    int64 
 27  Walc        395 non-null    int64 
 28  health      395 non-null    int64 
 29  absences    395 non-null    int64 
 30  G1          395 non-null    int64 
 31  G2          395 non-null    int64 
 32  G3          395 non-null    int64 
dtypes: int64(16), object(17)
"""
df.G3.describe()
"""
count    395.000000
mean      10.415190
std        4.581443
min        0.000000
25%        8.000000
50%       11.000000
75%       14.000000
max       20.000000
Name: G3, dtype: float64
"""
df.isnull().sum()
#no NA values

#Target variale G3
#Histogram
plt.hist(df.G3, bins = 'auto', facecolor = 'light blue')
plt.xlabel('G3')
plt.ylabel('counts')
plt.title('Histogram of G3')

#Boxplot
props2 = dict(boxes = 'grey', whiskers = 'green', medians = 'black', caps = 'blue')
df['G3'].plot.box(color=props2, patch_artist = True, vert = False)

#________outliers
#Shows no outliers
Q1 = np.percentile(df.G3, 25, interpolation = 'midpoint')  
Q2 = np.percentile(df.G3, 50, interpolation = 'midpoint')  
Q3 = np.percentile(df.G3, 75, interpolation = 'midpoint')   
print('Q1 25 percentile of the given data is, ', Q1) 
print('Q1 50 percentile of the given data is, ', Q2) 
print('Q1 75 percentile of the given data is, ', Q3)   


df.dtypes

#Variable 1 - School - categorical
df.school.value_counts()
"""
Out[30]: 
GP    349
MS     46
Name: school, dtype: int64
"""
#90% observations are GP school dominant , so there is no differentiation in schools 
#for that matter , but still this variable can be checked for it's significance with G3

#Convert GP and MS using label encoding
school_GP = df[df.school == 'GP']
school_MS = df[df.school == 'MS']
import scipy
scipy.stats.ttest_ind(school_GP.G3, school_MS.G3)
#This variable is not significant with G3 , so skip this variable

#Variable 2 - sex
df.sex.value_counts()
"""
F    208
M    187
Name: sex, dtype: int64
"""
#This is a good ratio and not biased
plt.hist(df.sex)
sex_M = df[df.sex == 'M']
sex_F = df[df.sex == 'F']
import scipy
scipy.stats.ttest_ind(sex_M.G3, sex_F.G3)
#Sex is significant to G3
#pvalue=0.039865332341527636

#Variable 3 - Age
plt.hist(df.age)
age = dict(boxes = 'green', whiskers = 'blue', medians = 'black', caps = 'blue')
df['age'].plot.box(color=age, patch_artist = True, vert = False)
#There is an outlier at age 22 , which can be ignored as it is quite closer to the range 21
df.age.describe()
# calculate Pearson's correlation
from scipy.stats import pearsonr
corr, _ = pearsonr(df.age, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: -0.162
#not a good significator for G3
#so Skip this variable , as it is not a good significator

#Variable 4 - address
df.address.value_counts()
"""
U    307 - Urban
R     88 - Rural
Name: address, dtype: int64
"""
add_U = df[df.address == 'U']
add_R = df[df.address == 'R']
import scipy
scipy.stats.ttest_ind(add_U.G3, add_R.G3)
#This is a significant variable for G3
#pvalue=0.03563267975655872

#Variable 5 - famsize
df.famsize.value_counts()
"""
GT3    281
LE3    114
Name: famsize, dtype: int64
"""
fam_GT3 = df[df.famsize == 'GT3']
fam_LE3 = df[df.famsize == 'LE3']
import scipy
scipy.stats.ttest_ind(fam_GT3.G3, fam_LE3.G3)
#This is not a significant variable to G3 pvalue=0.1062048278385956

#Variable 6 - Pstatus
df.Pstatus.value_counts()
"""
T    354
A     41
Name: Pstatus, dtype: int64
"""
#Cohabitating parents dominate this variable so there is no significant differentiation as such
#but still check the P values and decide
Pstatus_T = df[df.Pstatus == 'T']
Pstatus_A = df[df.Pstatus == 'A']
import scipy
scipy.stats.ttest_ind(Pstatus_T.G3, Pstatus_A.G3)
#not a significant variable for G3

#Variable 7 - Medu
df.Medu.value_counts()
"""
4    131
2    103
3     99
1     59
0      3
Name: Medu, dtype: int64
"""
plt.hist(df.Medu)
Medu = dict(boxes = 'green', whiskers = 'blue', medians = 'black', caps = 'blue')
df['Medu'].plot.box(color=Medu, patch_artist = True, vert = False)
#No outliers
#Use correlation to check its significance
from scipy.stats import pearsonr
corr, _ = pearsonr(df.Medu, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.217
# This can be taken as a significant variable in the model as of now

#Variable 8 - Fedu
plt.hist(df.Fedu)
Fedu = dict(boxes = 'green', whiskers = 'blue', medians = 'black', caps = 'blue')
df['Fedu'].plot.box(color=Fedu, patch_artist = True, vert = False)
#There is 1 outlier at 0 but can be ignored as this is education level and it is possible
#to have no education
#Use correlation to check its significance
from scipy.stats import pearsonr
corr, _ = pearsonr(df.Fedu, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.152
#somewhat significant

#Variable 9 - Mjob
df.Mjob.value_counts()
"""
other       141
services    103
at_home      59
teacher      58
health       34
Name: Mjob, dtype: int64
"""
#This is a categorical variable and can be done anova to check significance
#or can be clubbed as mom  at home or working mom
df['Mjob']=df.get('Mjob').replace('other','in job')
df['Mjob']=df.get('Mjob').replace('services','in job')
df['Mjob']=df.get('Mjob').replace('teacher','in job')
df['Mjob']=df.get('Mjob').replace('health','in job')
df['Mjob']=df.get('Mjob').replace('at_home','not in job')
df.Mjob.value_counts()
"""
in job        336
not in job     59
Name: Mjob, dtype: int64
"""
#Major oprtion of the count is of working moms so this column is dominant and can be skipped

#Variable 10 - Fjob
df.Fjob.value_counts()
"""
other       217
services    111
teacher      29
at_home      20
health       18
Name: Fjob, dtype: int64
"""
#This is a categorical variable and can be done anova to check significance
#and can be clubbed as mom  at home or working mom
df['Fjob']=df.get('Fjob').replace('other','in job')
df['Fjob']=df.get('Fjob').replace('services','in job')
df['Fjob']=df.get('Fjob').replace('teacher','in job')
df['Fjob']=df.get('Fjob').replace('health','in job')
df['Fjob']=df.get('Fjob').replace('at_home','not in job')
df.Fjob.value_counts()
"""
in job        375
not in job     20
Name: Fjob, dtype: int64
"""
#since this is dominant for fathers in job , this variable can be skipped

#variable 11 - reason
df.reason.value_counts()
"""
course        145
home          109
reputation    105
other          36
Name: reason, dtype: int64
"""
#Use anova to check for its significance
reason_anova = df[['reason','G3']]
reason_anova
grouped_anova2 = reason_anova.groupby(['reason'])
F_score, p = stats.f_oneway(grouped_anova2.get_group('course')["G3"], 
                            grouped_anova2.get_group('home')["G3"],
                            grouped_anova2.get_group('reputation')["G3"],
                            grouped_anova2.get_group('other')["G3"])
F_score
p #Out[82]: 0.10233745609730385 - not significant to G3 either
#This variable is not significant to G3

#Variable 12 - guardian
df.guardian.value_counts()
"""
mother    273
father     90
other      32
Name: guardian, dtype: int64
"""
guardian_anova = df[['guardian','G3']]
guardian_anova
grouped_anova2 = guardian_anova.groupby(['guardian'])
F_score, p = stats.f_oneway(grouped_anova2.get_group('mother')["G3"], 
                            grouped_anova2.get_group('father')["G3"],
                            grouped_anova2.get_group('other')["G3"])
F_score
p #Out[100]: 0.2051326420058259 - not significant to G3 also
#so this variable need to be skipped

#Variable 13 - traveltime
df.traveltime.value_counts()
"""
1    257
2    107
3     23
4      8
Name: traveltime, dtype: int64
"""
#use correlation to see the significance
from scipy.stats import pearsonr
corr, _ = pearsonr(df.traveltime, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: -0.117
#not a very good significator for G3 but can be considered as this makes sense

#Variable 14 - studytime
df.studytime.value_counts()
"""
2    198
1    105
3     65
4     27
Name: studytime, dtype: int64
"""
#use correlation to see the significance
corr, _ = pearsonr(df.studytime, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.098
#not a very good significator for G3 but can be considered as it makes sense

#Variable 15 - failures
df.failures.value_counts()
"""
0    312
1     50
2     17
3     16
Name: failures, dtype: int64
"""
#Use correlation to check the significance
from scipy.stats import pearsonr
corr, _ = pearsonr(df.failures, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: -0.360
#This variable can be considered as a good significator for G3

#Variable 16 - schoolsup
df.schoolsup.value_counts()
"""
no     344
yes     51
Name: schoolsup, dtype: int64
"""
#Use independent t-test to see the significance
school_yes = df[df.schoolsup == 'yes']
school_no = df[df.schoolsup == 'no']
import scipy
scipy.stats.ttest_ind(school_yes.G3, school_no.G3)  #not significant
#pvalue=0.10038496363910417

#Variable 17 - famsup
df.famsup.value_counts()
"""
yes    242
no     153
Name: famsup, dtype: int64
"""
#Use independent t-test to see the significance
famsup_yes = df[df.famsup == 'yes']
famsup_no = df[df.famsup == 'no']
import scipy
scipy.stats.ttest_ind(famsup_yes.G3, famsup_no.G3)  #not significant 
#pvalue=0.43771108589489893

#Variable 18 - paid
df.paid.value_counts()
"""
no     214
yes    181
Name: paid, dtype: int64
"""
#Use independent t-test to see the significance
paid_yes = df[df.paid == 'yes']
paid_no = df[df.paid == 'no']
import scipy
scipy.stats.ttest_ind(paid_yes.G3, paid_no.G3)  #significant 
#pvalue=0.04276506403357553

#Variable 19 - activities
df.activities.value_counts()
"""
yes    201
no     194
Name: activities, dtype: int64
"""
#Use independent t-test to see the significance
activities_yes = df[df.activities == 'yes']
activities_no = df[df.activities == 'no']
import scipy
scipy.stats.ttest_ind(activities_yes.G3, activities_no.G3)  #not significant

#Variable 20 - nursery
df.nursery.value_counts()
"""
yes    314
no      81
Name: nursery, dtype: int64
"""
# Nursery YES values are dominating
#Use independent t-test to see the significance
nursery_yes = df[df.nursery == 'yes']
nursery_no = df[df.nursery == 'no']
import scipy
scipy.stats.ttest_ind(nursery_yes.G3, nursery_no.G3)  #not significant

#Variable 21 - higher
df.higher.value_counts()
"""
yes    375
no      20
Name: higher, dtype: int64
"""
# Nursery YES values are dominating
#Use independent t-test to see the significance
higher_yes = df[df.higher == 'yes']
higher_no = df[df.higher == 'no']
import scipy
scipy.stats.ttest_ind(higher_yes.G3, higher_no.G3)  #significant
#pvalue=0.0002668001587281805

#Variable 22 - internet
df.internet.value_counts()
"""
yes    329
no      66
Name: internet, dtype: int64
"""
#Use independent t-test to see the significance
internet_yes = df[df.internet == 'yes']
internet_no = df[df.internet == 'no']
import scipy
scipy.stats.ttest_ind(internet_yes.G3, internet_no.G3)  #pvalue=0.05048021213717338
#can be considered as of now

#Variable 23 - romantic
df.romantic.value_counts()
"""
no     263
yes    132
Name: romantic, dtype: int64
"""
#Use independent t-test to see the significance
romantic_yes = df[df.romantic == 'yes']
romantic_no = df[df.romantic == 'no']
import scipy
scipy.stats.ttest_ind(romantic_yes.G3, romantic_no.G3)  #significant
#pvalue=0.009712726394119265

#Variable 24 - famrel
df.famrel.value_counts()
"""
4    195
5    106
3     68
2     18
1      8
Name: famrel, dtype: int64
"""
#Use independent ttest to see the significance 
from scipy.stats import pearsonr
corr, _ = pearsonr(df.famrel, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.051
#not a good significator for G3

#Variable 25 - freetime
df.freetime.value_counts()
"""
3    157
4    115
2     64
5     40
1     19
Name: freetime, dtype: int64
"""
from scipy.stats import pearsonr
corr, _ = pearsonr(df.freetime, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.011
#NOt a good significator for either of the 3

#Variable 26 - goout
df.goout.value_counts()
"""
3    130
2    103
4     86
5     53
1     23
Name: goout, dtype: int64
"""
from scipy.stats import pearsonr
corr, _ = pearsonr(df.goout, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: -0.133
#NOt a good significator for either of the 3

#Variable 27 - Dalc
df.Dalc.value_counts()
"""
1    276
2     75
3     26
5      9
4      9
Name: Dalc, dtype: int64
"""
from scipy.stats import pearsonr
corr, _ = pearsonr(df.Dalc, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: -0.055
#NOt a good significator for either of the 3

#Variable 28 - Walc
df.Walc.value_counts()
"""
1    151
2     85
3     80
4     51
5     28
Name: Walc, dtype: int64
"""
from scipy.stats import pearsonr
corr, _ = pearsonr(df.Walc, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: -0.052
#NOt a good significator for G3

#Variable 29 - health
df.health.value_counts()
"""
5    146
3     91
4     66
1     47
2     45
Name: health, dtype: int64
"""
from scipy.stats import pearsonr
corr, _ = pearsonr(df.health, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: -0.061
#NOt a good significator for G3

#Variable 30 - absences
plt.hist(df.absences)
plt.boxplot(df.absences)
#Many outliers are there , can be untouched as of now
#use correlation to check the significance
from scipy.stats import pearsonr
corr, _ = pearsonr(df.absences, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.034
#not a good significator for G3
#so Skip this variable , as it is not a good significator

#variable 31 - G1
#use correlation to check the significance
from scipy.stats import pearsonr
corr, _ = pearsonr(df.G1, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.801
#very good significator for G3

#variable 32 - G2
#use correlation to check the significance
from scipy.stats import pearsonr
corr, _ = pearsonr(df.G2, df.G3)
print('Pearsons correlation: %.3f' % corr)
#Pearsons correlation: 0.905
#very good significator for G3

###### Also correlation heatmap ############

corr = df.corr()
sns.heatmap(corr)

################# Start label encoding using dummy ############################
dp = df
#df = dp
df.head()
from sklearn.preprocessing import LabelEncoder

from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split

df_dummies_add = pd.get_dummies(df['address'],drop_first = True)
df_dummies_gen = pd.get_dummies(df['sex'],drop_first = True)
df_dummies_paid = pd.get_dummies(df['paid'],prefix = 'paid',drop_first = True)
df_dummies_higher = pd.get_dummies(df['higher'],prefix = 'higher',drop_first = True)
df_dummies_internet = pd.get_dummies(df['internet'],prefix = 'internet',drop_first = True)

df_new = pd.concat([df, df_dummies_gen, df_dummies_add, df_dummies_paid, 
                    df_dummies_higher, df_dummies_internet], axis=1)

df_new.shape #Out[323]: (395, 38)

fit = ols('''G3~G1+G2+M+U+Medu+Fedu+traveltime+studytime+failures+
          paid_yes+higher_yes+internet_yes+goout''', data=df_new).fit()

fit.summary() 
'''
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -1.8597      0.807     -2.305      0.022      -3.446      -0.273
G1               0.1510      0.058      2.609      0.009       0.037       0.265
G2               0.9764      0.051     18.984      0.000       0.875       1.078
M                0.1046      0.213      0.491      0.624      -0.314       0.523
U               -0.0473      0.256     -0.184      0.854      -0.552       0.457
Medu             0.1429      0.119      1.201      0.230      -0.091       0.377
Fedu            -0.1401      0.118     -1.190      0.235      -0.372       0.091
traveltime       0.0984      0.152      0.646      0.518      -0.201       0.398
studytime       -0.1980      0.128     -1.543      0.124      -0.450       0.054
failures        -0.2529      0.151     -1.675      0.095      -0.550       0.044
paid_yes         0.0934      0.208      0.448      0.654      -0.316       0.503
higher_yes       0.3228      0.483      0.669      0.504      -0.626       1.272
internet_yes    -0.1122      0.277     -0.405      0.685      -0.656       0.432
goout            0.0651      0.091      0.717      0.474      -0.113       0.244
==============================================================================
Omnibus:                      230.903   Durbin-Watson:                   1.873
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1537.482
Skew:                          -2.507   Prob(JB):                         0.00
Kurtosis:                      11.263   Cond. No.                         150.
==============================================================================
'''

#remove all those columns which have p value > 0.05 , so oly intercept and G1,G2 are left
#but that makes an overly simple model , also since the dataset is very small , It would be wise
#to keep a few features based on common sense and possibilities

X = df_new[['G1', 'G2','M','Medu','studytime','goout','higher_yes']] #Independent variable 

#X = df_new[['G1', 'G2']]
y = df_new['G3'] #dependent variable 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LinearRegression 
dp1 = df_new

from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 
lm.fit(X_train,y_train)
print(lm.intercept_) 
#-2.549303503235773 with all the columns
predictions = lm.predict(X_test) 
plt.scatter(y_test,predictions)
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(lm.score(X_test, y_test)))
#Variance score: 0.7706770639171207
### plotting residual errors in training data
plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
## plotting residual errors in test data
plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
## plotting legend
plt.legend(loc = 'upper right')
## plot title
plt.title("Residual errors")
## method call for showing the plot
plt.show()

sns.distplot((y_test-predictions))
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
#MAE: 1.2600232386554402
print('MSE:', metrics.mean_squared_error(y_test, predictions))
#MSE: 4.434980242445719
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#RMSE: 2.105939277957871
from sklearn.metrics import r2_score
print(r2_score(y_test, predictions))
#0.8212400009278952

########## Using Cross Validation  ##############
from sklearn.model_selection import cross_val_score
mse=cross_val_score(lm,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse) #-4.005091827052748
r2=cross_val_score(lm,X,y,scoring='r2',cv=5)
mean_r2=np.mean(r2)
mean_r2 #Out[54]: 0.8063528650869429

#########    Using K-Fold   ###############
from sklearn.model_selection import KFold
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=folds)
scores #Out[58]: array([0.81578371, 0.8281371 , 0.86477575, 0.81592492, 0.85079035])
np.mean(scores) #Out[59]: 0.835082366565549

########## Using Random Forest ##########
from sklearn.ensemble import RandomForestRegressor
random_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
random_regressor.fit(X_train, y_train)
y_pred = random_regressor.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred)) #0.6459312596204794

############ Grid Search #######################
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
#specify range of hyperparameters to tune
from sklearn.feature_selection import RFE
rfe = RFE(lm, n_features_to_select=11)             
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
hyper_params = [{'n_features_to_select': list(range(1, 12))}]
hyper_params
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm)
# Use GridSearchCV()
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)
model_cv.fit(X_train, y_train)
"""
GridSearchCV(cv=KFold(n_splits=5, random_state=100, shuffle=True),
             estimator=RFE(estimator=LinearRegression()),
             param_grid=[{'n_features_to_select': [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                   10, 11]}],
             return_train_score=True, scoring='r2', verbose=1)
"""
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
# plotting cv results
plt.figure(figsize=(16,6))
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

#Now we can choose the optimal value of number of features and build a final model.

# final model
n_features_optimal = 5
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(X_train, y_train)
# predict prices of X_test
y_pred = lm.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
#0.7706770639171207

