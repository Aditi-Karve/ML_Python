# Jesus is my Saviour!
import os
os.chdir('C:\\Users\\Dr Vinod\\Desktop\\WD_python')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

hd = pd.read_csv('HeartDisease.csv')
hd.info()
hd.shape #303 observations & 14 variables

#Target variable - target
hd.target.describe()
hd.target.value_counts()
'''
Patient has heart disease
1    165 - yes 
0    138 - no '''
hd.target.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='target', data=hd)
plt.xlabel('Patient has heart disease')
plt.ylabel('counts')
plt.title('Histogram of Patient has heart disease') 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['target'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

#age
hd.age.describe()
hd.age.value_counts() #across different ages
hd.age.value_counts().sum() #303
hd.age.isnull().sum() #No missing values

#Histogram
plt.hist(hd.age, bins='auto')
plt.xlabel("Patient's Age")
plt.ylabel('counts')
plt.title("Histogram of Patient's age") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['age'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

#gender
hd.gender.describe()
hd.gender.value_counts()
'''
1    207 - Female
0     96 - male'''
hd.gender.value_counts().sum() #303
hd.gender.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='gender', data=hd)
plt.xlabel("Patient's gender")
plt.ylabel('counts')
plt.title("Histogram of Patient's gender") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['gender'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

#chest_pain
hd.chest_pain.describe()
hd.chest_pain.value_counts()
'''
It refers to the chest pain experienced by the patient -(0,1,2,3)
0    143
2     87
1     50
3     23'''
hd.chest_pain.value_counts().sum() #303
hd.chest_pain.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='chest_pain', data=hd)
plt.xlabel("Patient's chest_pain experience")
plt.ylabel('counts')
plt.title("Histogram of Patient's chest_pain experience") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['chest_pain'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

#rest_bps - Blood pressure of the patient while resting (in mm/Hg)
hd.rest_bps.describe()
hd.rest_bps.value_counts() #across various 
hd.rest_bps.value_counts().sum() #303
hd.rest_bps.isnull().sum() #No missing values

#Histogram
plt.hist(hd.rest_bps, bins='auto')
plt.xlabel("Blood pressure of the patient while resting")
plt.ylabel('counts')
plt.title("Histogram of Blood pressure of the patient while resting") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['rest_bps'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

iqr = hd.rest_bps.describe()['75%'] - hd.rest_bps.describe()['25%']
up_lim = hd.rest_bps.describe()['75%']+1.5*iqr
len(hd.rest_bps[hd.rest_bps > up_lim]) #9 Outliers - Ignoring outliers

#cholestrol - Patient's cholesterol level (in mg/dl)
hd.cholestrol.describe()
hd.cholestrol.value_counts() #across various 
hd.cholestrol.value_counts().sum() #303
hd.cholestrol.isnull().sum() #No missing values

#Histogram
plt.hist(hd.cholestrol, bins='auto')
plt.xlabel("Patient's cholesterol level")
plt.ylabel('counts')
plt.title("Histogram of Patient's cholesterol level") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['cholestrol'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

iqr = hd.cholestrol.describe()['75%'] - hd.cholestrol.describe()['25%']
up_lim = hd.cholestrol.describe()['75%']+1.5*iqr
len(hd.cholestrol[hd.cholestrol > up_lim]) #5 Outliers - Keeping outliers

#Removing extreme outlier ie 564
hd = hd[hd.cholestrol < 500]
hd.cholestrol.describe()

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['cholestrol'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

#fasting_blood_sugar 
hd.fasting_blood_sugar.describe()
hd.fasting_blood_sugar.value_counts() 
'''
0    257
1     45'''
hd.fasting_blood_sugar.value_counts().sum() #302
hd.fasting_blood_sugar.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='fasting_blood_sugar', data=hd)
plt.xlabel(" Patient's fasting_blood_sugar ")
plt.ylabel('counts')
plt.title("Countplot of Patient's fasting_blood_sugar") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['fasting_blood_sugar'].plot.box(color=props2, patch_artist = True, vert = False) #Ignore outliers
'''Details can be in more detail'''

#rest_ecg - Potassium level (0,1,2)
hd.rest_ecg.describe()
hd.rest_ecg.value_counts() 
'''
1    152
0    146
2      4'''
hd.rest_ecg.value_counts().sum() #302
hd.rest_ecg.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='rest_ecg', data=hd)
plt.xlabel(" Patient's Potassium level ")
plt.ylabel('counts')
plt.title("Countplot of Patient's Potassium level") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['rest_ecg'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

#thalach - The patient’s maximum heart rate
hd.thalach.describe()
hd.thalach.value_counts()  #Across various
hd.thalach.value_counts().sum() #302
hd.thalach.isnull().sum() #No missing values

#Histogram
plt.hist(hd.thalach, bins='auto')
plt.xlabel(" Patient’s maximum heart rate")
plt.ylabel('counts')
plt.title("Histogram of Patient’s maximum heart rate") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['thalach'].plot.box(color=props2, patch_artist = True, vert = False) # outliers

iqr = hd.thalach.describe()['75%'] - hd.thalach.describe()['25%']
low_lim = hd.thalach.describe()['25%']-1.5*iqr
len(hd.thalach[hd.thalach < low_lim]) #1 Outlier ie 71 - Keeping outliers

#exer_angina - It refers to exercise-induced angina - (1=Yes, 0=No)
#Chest discomfort or shortness of breath caused when heart muscles receive insufficient oxygen-rich blood
hd.exer_angina.describe()
hd.exer_angina.value_counts()  
'''
0    203
1     99'''
hd.exer_angina.value_counts().sum() #302
hd.exer_angina.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='exer_angina', data=hd)
plt.xlabel(" Patient exercise-induced angina")
plt.ylabel('counts')
plt.title("Countplot of Patient exercise-induced angina") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['exer_angina'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

#old_peak
'''It is the ST depression induced by exercise relative to rest(ST
relates to the position on ECG plots)'''
hd.old_peak.describe()
hd.old_peak.value_counts()  #Across various 
hd.old_peak.value_counts().sum() #302
hd.old_peak.isnull().sum() #No missing values

#Histogram
plt.hist(hd.cholestrol, bins='auto')
plt.xlabel(" Patient's old_peak")
plt.ylabel('counts')
plt.title("Histogram of Patient's old_peak") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['old_peak'].plot.box(color=props2, patch_artist = True, vert = False) #Outliers
len(hd.old_peak[hd.old_peak > 5]) #2 outliers more than 5
#Removed 2 extreme outliers
hd= hd[hd['old_peak']<5]
hd['old_peak'].plot.box(color=props2, patch_artist = True, vert = False) #Outliers
hd.info()

#slope 
'''It refers to the slope of the peak of the exercise ST-Segment-(0,1,2)'''
hd.slope.describe()
hd.slope.value_counts()  
'''
2    142
1    139
0     19'''
hd.slope.value_counts().sum() #300
hd.slope.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='slope', data=hd)
plt.xlabel(" Patient's slope")
plt.ylabel('counts')
plt.title("Countplot of Patient's slope") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['slope'].plot.box(color=props2, patch_artist = True, vert = False) #No Outliers

#ca - Number of major vessels - (0,1,2,3,4)
hd.ca.describe()
hd.ca.value_counts()  
'''
0    173
1     65
2     38
3     19
4      5'''
hd.ca.value_counts().sum() #300
hd.ca.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='ca', data=hd)
plt.xlabel(" Patient's ca")
plt.ylabel('counts')
plt.title("Countplot of Patient's ca") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['ca'].plot.box(color=props2, patch_artist = True, vert = False) #24 Outliers

#thalassemia - It refers to thalassemia which is a blood disorder - (0,1,2,3)
hd.thalassemia.describe()
hd.thalassemia.value_counts()  
'''
2    166
3    114
1     18
0      2'''
hd.thalassemia.value_counts().sum() #300
hd.thalassemia.isnull().sum() #No missing values

#Barplot/ Countplot
sns.countplot(x='thalassemia', data=hd)
plt.xlabel(" Patient's thalassemia")
plt.ylabel('counts')
plt.title("Countplot of Patient's thalassemia") 

#Boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
hd['thalassemia'].plot.box(color=props2, patch_artist = True, vert = False) #2 outliers

#Finding the correlation
hd.corr().target.sort_values()
sns.heatmap(hd.corr())

hd.to_csv('heartdisease-1.csv', index=False)

import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
hd1 = pd.read_csv('heartdisease-1.csv')
hd1.info()
hd1.shape #300, 14

#Predictors
x = hd1.iloc[:,:13]

#Respond / target variable
y = hd1.iloc[:,13]

#Partitioning the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=123)

len(x_train) #210
len(x_test) #90
len(y_train) #210
len(y_test) #90

#Building Tree
clf = tree.DecisionTreeClassifier()
hd_clf = clf.fit(x_train, y_train)

#Plotting Tree
fig, ax = plt.subplots(figsize=(20, 20))
tree.plot_tree(hd_clf, ax=ax, fontsize=8,filled=True)
plt.show()

#Prediction on test data
y_pred = hd_clf.predict(x_test)
len(y_pred)
print(y_pred)

#Confusion Matrix & Report
pd.crosstab(y_test,y_pred, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        28  15   43
1        14  33   47
All      42  48   90'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) #0.68

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred)

#AUC -Area Under Curve
roc_auc = auc(fpr,tpr)
print(roc_auc) #0.68

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       0.67      0.65      0.66        43
           1       0.69      0.70      0.69        47

    accuracy                           0.68        90
   macro avg       0.68      0.68      0.68        90
weighted avg       0.68      0.68      0.68        90'''

#Bagging 300 Trees
from sklearn.ensemble import BaggingClassifier
#Base estimator is clf
#Build Bagging Classifier bc
hd_bc = BaggingClassifier(base_estimator=clf, n_estimators=300, oob_score=True, n_jobs=-1)

#Bagging Classifier fitting with training data set
hd_bc.fit(x_train, y_train)    

#Predictions
y_predbc = hd_bc.predict(x_test)
print(y_predbc)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predbc, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        32  11   43
1        10  37   47
All      42  48   90'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predbc) #0.77

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_predbc)

#AUC -Area Under Curve
bc_roc_auc = auc(fpr,tpr)
print(bc_roc_auc) #0.77

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(bc_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predbc))
'''
              precision    recall  f1-score   support

           0       0.76      0.74      0.75        43
           1       0.77      0.79      0.78        47

    accuracy                           0.77        90
   macro avg       0.77      0.77      0.77        90
weighted avg       0.77      0.77      0.77        90'''

#Random Forest
from sklearn.ensemble import RandomForestClassifier
#Create Model with 500 trees
rf = RandomForestClassifier(n_estimators=500,bootstrap=True,max_features='sqrt')

#Fitting the model
hd_rf = rf.fit(x_train, y_train)

#Prediction
y_predrf = hd_rf.predict(x_test)
len(y_predrf)
print(y_predrf)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predrf, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        34   9   43
1        10  37   47
All      44  46   90'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predrf) #0.79

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_predrf)

#AUC -Area Under Curve
rf_roc_auc = auc(fpr,tpr)
print(rf_roc_auc) #0.79

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(rf_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predrf))
'''
              precision    recall  f1-score   support

           0       0.77      0.79      0.78        43
           1       0.80      0.79      0.80        47

    accuracy                           0.79        90
   macro avg       0.79      0.79      0.79        90
weighted avg       0.79      0.79      0.79        90'''

#Probabilities
yp_predrf = hd_rf.predict_proba(x_test)[:,1]
print(yp_predrf)
len(yp_predrf)


#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, yp_predrf)

#AUC
rfp_roc_auc = auc(fpr,tpr)
print(rfp_roc_auc) #.89

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(rfp_roc_auc))
plt.legend(loc=4)
plt.show()

#Importance of variables
#Extract Feature importance
fi = pd.DataFrame({'feature': list(x_train.columns),
                   'importance': hd_rf.feature_importances_}).\
                    sort_values('importance', ascending=False)

#Display
fi.head()
'''
        feature  importance
12  thalassemia    0.141911
2    chest_pain    0.121503
11           ca    0.114998
7       thalach    0.104726
0           age    0.087574'''

#Adaptive Boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
adafit = ada.fit(x_train, y_train)
print(adafit)

#Prediction
y_predada = adafit.predict(x_test)
len(y_predada)
print(y_predada)

#Confusion Matrix & Report
pd.crosstab(y_test,y_predada, margins=True,rownames=['Actual'], colnames=['Predict'])
'''
Predict   0   1  All
Actual              
0        33  10   43
1        14  33   47
All      47  43   90'''

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predada) #0.73

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_predada)

#AUC -Area Under Curve
ada_roc_auc = auc(fpr,tpr)
print(ada_roc_auc) #0.73

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(ada_roc_auc))
plt.legend(loc=4)
plt.show()

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predada))
'''
              precision    recall  f1-score   support

           0       0.70      0.77      0.73        43
           1       0.77      0.70      0.73        47

    accuracy                           0.73        90
   macro avg       0.73      0.73      0.73        90
weighted avg       0.74      0.73      0.73        90'''

#Probabilities
yp_predada = adafit.predict_proba(x_test)[:,1]
print(yp_predada)
len(yp_predada)

#finding fpr, tpr & thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, yp_predada)

#AUC -Area Under Curve
adap_roc_auc = auc(fpr,tpr)
print(adap_roc_auc) #0.84

#ROC Curve
plt.title('ROC Curve for Heart Disease')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(adap_roc_auc))
plt.legend(loc=4)
plt.show()

