# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:40:34 2021

@author: Aditi
"""
import os
os.chdir("C:\\Users\\Tanay\\Desktop\\Python\\Logistic Reg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

df = pd.read_csv('h1n1_vaccine_prediction.csv')
df.info()
df.describe()

#Variable 1 - h1n1_worry
df.h1n1_worry.describe()
"""
Out[12]: 
count    26615.000000
mean         1.618486
std          0.910311
min          0.000000
25%          1.000000
50%          2.000000
75%          2.000000
max          3.000000
Name: h1n1_worry, dtype: float64
"""
plt.hist(df.h1n1_worry, bins = 'auto', facecolor = 'red')
plt.xlabel('h1n1_worry')
plt.ylabel('counts')
plt.title('Histogram of h1n1_worry')
df.h1n1_worry.value_counts()
"""
Out[14]: 
2.0    10575
1.0     8153
3.0     4591
0.0     3296
Name: h1n1_worry, dtype: int64
"""
df.h1n1_worry.value_counts().sum()
#Out[15]: 26615
26707 - 26615 #92 are NA , replace with mode
df['h1n1_worry'].fillna(df['h1n1_worry'].mode()[0],inplace=True)
df.h1n1_worry.value_counts().sum()
sns.countplot(df.h1n1_worry)
#These are ordinal categorical , so do independent t test
"""
df['h1n1_worry']=df.get('h1n1_worry').replace(1.0,1)
df['h1n1_worry']=df.get('h1n1_worry').replace(2.0,1)
df['h1n1_worry']=df.get('h1n1_worry').replace(3.0,1)
df['h1n1_worry']=df.get('h1n1_worry').replace(0.0,0)
sns.countplot(df.h1n1_worry)
df.h1n1_worry.value_counts()
"""
"""
Out[25]: 
1.0    23411
0.0     3296
Name: h1n1_worry, dtype: int64
23411/26707 #87% observations are dominated by worried about H1N1 category , so skip the variable
"""
h1n1_worry_0 = df[df.h1n1_vaccine == 0]
h1n1_worry_1 = df[df.h1n1_vaccine == 1]
len(df.h1n1_vaccine[df.h1n1_vaccine == 0]) #14381
len(df.h1n1_vaccine[df.h1n1_vaccine == 1]) #4777
import scipy
scipy.stats.ttest_ind(h1n1_worry_0.h1n1_worry, h1n1_worry_1.h1n1_worry)
#Out[45]: Ttest_indResult(statistic=-20.015652900079157, pvalue=1.7943418511969305e-88)
#So this is a significant variable

#Variable 2 - h1n1_awareness
df.h1n1_awareness.describe()
"""
Out[32]: 
count    26591.000000
mean         1.262532
std          0.618149
min          0.000000
25%          1.000000
50%          1.000000
75%          2.000000
max          2.000000
Name: h1n1_awareness, dtype: float64
"""
plt.hist(df.h1n1_awareness, bins = 'auto', facecolor = 'purple')
plt.xlabel('h1n1_awareness')
plt.ylabel('counts')
plt.title('Histogram of h1n1_awareness')

df.h1n1_awareness.value_counts()
"""
Out[34]: 
1.0    14598
2.0     9487
0.0     2506
Name: h1n1_awareness, dtype: int64
"""
df.h1n1_awareness.value_counts().sum()
#Out[15]: 26591
26707 - 26591 #116 are NA , replace with mode
df['h1n1_awareness'].fillna(df['h1n1_awareness'].mode()[0],inplace=True)
df.h1n1_awareness.value_counts().sum()
sns.countplot(df.h1n1_awareness)
"""
#These care 3 categories and can be clubbed to make 2 , 0 - not having knowledge about H1N1 and
#1 for having knowledge about H1N1
df['h1n1_awareness']=df.get('h1n1_awareness').replace(1.0,1)
df['h1n1_awareness']=df.get('h1n1_awareness').replace(2.0,1)
df['h1n1_awareness']=df.get('h1n1_awareness').replace(0.0,0)
sns.countplot(df.h1n1_worry)
df.h1n1_worry.value_counts()
"""
Out[44]: 
1.0    23411
0.0     3296
Name: h1n1_worry, dtype: int64
"""
23411/26707 #87% values are dominated by having knowledge about H1N1 , so skip the variable
"""
h1n1_awareness_0 = df[df.h1n1_vaccine == 0]
h1n1_awareness_1 = df[df.h1n1_vaccine == 1]
len(df.h1n1_vaccine[df.h1n1_vaccine == 0]) #14381
len(df.h1n1_vaccine[df.h1n1_vaccine == 1]) #4777
import scipy
scipy.stats.ttest_ind(h1n1_awareness_0.h1n1_awareness, h1n1_awareness_1.h1n1_awareness)
#Out[58]: Ttest_indResult(statistic=-19.38065499194581, pvalue=4.188858352900413e-83)
#so this is significant variable
df.dtypes

#Variable 3 - antiviral_medication
df.antiviral_medication.describe()
"""
Out[46]: 
count    26636.000000
mean         0.048844
std          0.215545
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max          1.000000
Name: antiviral_medication, dtype: float64
"""
plt.hist(df.antiviral_medication, bins = 'auto', facecolor = 'red')
plt.xlabel('antiviral_medication')
plt.ylabel('counts')
plt.title('Histogram of antiviral_medication')

df.antiviral_medication.value_counts()
"""
0.0    25335
1.0     1301
Name: antiviral_medication, dtype: int64
"""
df.antiviral_medication.value_counts().sum()
#Out[15]: 26636
26707 - 26636 #71 are NA , replace with mode
df['antiviral_medication'].fillna(df['antiviral_medication'].mode()[0],inplace=True)
df.antiviral_medication.value_counts().sum()
sns.countplot(df.antiviral_medication)
sns.countplot(df.antiviral_medication)
df.antiviral_medication.value_counts()
"""
0.0    25406
1.0     1301
Name: antiviral_medication, dtype: int64
"""
25406/26707 #95% values are dominated by people not having taken antiviral medication
#so skip this variable

#Variable 4 - contact_avoidance
df.contact_avoidance.describe()
#Histogram
plt.hist(df.contact_avoidance, bins = 'auto', facecolor = 'red')
plt.xlabel('contact_avoidance')
plt.ylabel('counts')
plt.title('Histogram of contact_avoidance')

df.contact_avoidance.value_counts()
"""
Out[60]: 
1.0    19228
0.0     7271
Name: contact_avoidance, dtype: int64
"""
19228/26707
df.contact_avoidance.value_counts().sum()
#Out[15]: 26499
26707 - 26499 #208 are NA , replace with mode
df['contact_avoidance'].fillna(df['contact_avoidance'].mode()[0],inplace=True)
df.contact_avoidance.value_counts().sum()
sns.countplot(df.contact_avoidance)
df.contact_avoidance.value_counts()
"""
1.0    19436
0.0     7271
Name: contact_avoidance, dtype: int64
"""
#use chisquare test to check significance of this variable
con_avo = pd.crosstab(df['contact_avoidance'], df['h1n1_vaccine']) 
con_avo
"""
h1n1_vaccine           0     1
contact_avoidance             
0.0                 5954  1317
1.0                15079  4357
"""
from scipy.stats import chi2_contingency
c, p, dof, expected = chi2_contingency(con_avo)
p #Out[70]: 2.2161129058289884e-14
#p_value < 0.05 , so variable is significant

#Variable 5 - bought_face_mask
df.bought_face_mask.describe()
#Histogram
plt.hist(df.bought_face_mask, bins = 'auto', facecolor = 'red')
plt.xlabel('bought_face_mask')
plt.ylabel('counts')
plt.title('Histogram of bought_face_mask')

df.bought_face_mask.value_counts()
"""
0.0    24847
1.0     1841
Name: bought_face_mask, dtype: int64
"""
df.bought_face_mask.value_counts().sum()
#Out[15]: 26688
26707 - 26688 #19 are NA , replace with mode
df['bought_face_mask'].fillna(df['bought_face_mask'].mode()[0],inplace=True)
df.bought_face_mask.value_counts().sum()
sns.countplot(df.bought_face_mask)
df.bought_face_mask.value_counts()
"""
0.0    24866
1.0     1841
Name: bought_face_mask, dtype: int64
"""
24866/26707 #93% obs are dominated by people who did not buy face mask so chisquare is also
#not required and skip this variable

#Variable 6 - wash_hands_frequently
df.wash_hands_frequently.describe()
#Histogram
plt.hist(df.wash_hands_frequently, bins = 'auto', facecolor = 'red')
plt.xlabel('wash_hands_frequently')
plt.ylabel('counts')
plt.title('Histogram of wash_hands_frequently')
df.wash_hands_frequently.value_counts()
df.wash_hands_frequently.value_counts().sum()
#26665
26707-26665 #42 are NA , replace with mode
df['wash_hands_frequently'].fillna(df['wash_hands_frequently'].mode()[0],inplace=True)
sns.countplot(df.wash_hands_frequently)
df.wash_hands_frequently.value_counts()
df.wash_hands_frequently.value_counts().sum()
"""
Out[91]: 
1.0    22057
0.0     4650
Name: wash_hands_frequently, dtype: int64
"""
22057/26707
#82% are dominant by category - 1

#Variable 7 - avoid_large_gatherings
df.avoid_large_gatherings.value_counts()
df.avoid_large_gatherings.value_counts().sum()
26707-26620  #87 are NA
df['avoid_large_gatherings'].fillna(df['avoid_large_gatherings'].mode()[0],inplace=True)
df.avoid_large_gatherings.value_counts()
"""
0.0    17160
1.0     9547
Name: avoid_large_gatherings, dtype: int64
"""
plt.hist(df.avoid_large_gatherings, bins = 'auto', facecolor = 'purple')
plt.xlabel('avoid_large_gatherings')
plt.ylabel('counts')
plt.title('Histogram of avoid_large_gatherings')
sns.countplot(df.avoid_large_gatherings)
gath = pd.crosstab(df['avoid_large_gatherings'], df['h1n1_vaccine']) 
gath
"""
h1n1_vaccine                0     1
avoid_large_gatherings             
0.0                     13609  3551
1.0                      7424  2123
"""
c, p, dof, expected = chi2_contingency(gath)
p #Out[107]: 0.003276654049065037
#p_value < 0.05 , so this is a significant variable

#Variable 8 - reduced_outside_home_cont
df.reduced_outside_home_cont.describe()
df.reduced_outside_home_cont.value_counts()
df.reduced_outside_home_cont.value_counts().sum()
26707-26625  #82 are NA
df['reduced_outside_home_cont'].fillna(df['reduced_outside_home_cont'].mode()[0],inplace=True)
plt.hist(df.reduced_outside_home_cont, bins = 'auto', facecolor = 'purple')
plt.xlabel('reduced_outside_home_cont')
plt.ylabel('counts')
plt.title('Histogram of reduced_outside_home_cont')
df.reduced_outside_home_cont.value_counts()
"""
0.0    17726
1.0     8981
Name: reduced_outside_home_cont, dtype: int64
"""
sns.countplot(df.reduced_outside_home_cont)
red_con = pd.crosstab(df['reduced_outside_home_cont'], df['h1n1_vaccine']) 
red_con
"""
h1n1_vaccine                   0     1
reduced_outside_home_cont             
0.0                        14074  3652
1.0                         6959  2022
"""
c, p, dof, expected = chi2_contingency(red_con)
p #Out[109]: 0.00032755872863934096
#p_value < 0.05 , so this is a significant variable

#Variable 9 - avoid_touch_face
df.avoid_touch_face.describe()
df.avoid_touch_face.value_counts()
df.avoid_touch_face.value_counts().sum()
26707-26579  #128 are NA
df['avoid_touch_face'].fillna(df['avoid_touch_face'].mode()[0],inplace=True)
plt.hist(df.avoid_touch_face, bins = 'auto', facecolor = 'purple')
plt.xlabel('avoid_touch_face')
plt.ylabel('counts')
plt.title('Histogram of avoid_touch_face')
sns.countplot(df.avoid_touch_face)
df.avoid_touch_face.value_counts()
"""
1.0    18129
0.0     8578
Name: avoid_touch_face, dtype: int64
"""
face_con = pd.crosstab(df['avoid_touch_face'], df['h1n1_vaccine']) 
face_con
"""
h1n1_vaccine          0     1
avoid_touch_face             
0.0                7117  1461
1.0               13916  4213
"""
c, p, dof, expected = chi2_contingency(face_con)
p #Out[132]: 6.3218424278463514e-31

#Variable 10 - dr_recc_h1n1_vacc
df.dr_recc_h1n1_vacc.describe()
df.dr_recc_h1n1_vacc.value_counts()
df.dr_recc_h1n1_vacc.value_counts().sum()
26707-24547  #2160 are NA
df['dr_recc_h1n1_vacc'].fillna(df['dr_recc_h1n1_vacc'].mode()[0],inplace=True)
plt.hist(df.dr_recc_h1n1_vacc, bins = 'auto', facecolor = 'purple')
plt.xlabel('dr_recc_h1n1_vacc')
plt.ylabel('counts')
plt.title('Histogram of dr_recc_h1n1_vacc')
sns.countplot(df.dr_recc_h1n1_vacc)
df.dr_recc_h1n1_vacc.value_counts()
"""
0.0    21299
1.0     5408
Name: dr_recc_h1n1_vacc, dtype: int6
"""
dr_rec = pd.crosstab(df['dr_recc_h1n1_vacc'], df['h1n1_vaccine']) 
dr_rec
"""
h1n1_vaccine           0     1
dr_recc_h1n1_vacc             
0.0                18504  2795
1.0                 2529  2879
"""
c, p, dof, expected = chi2_contingency(dr_rec)
p #Out[465]: 0.0
#p_value < 0.05 so this is a significant variable

#Variable 11 - dr_recc_seasonal_vacc
df.dr_recc_seasonal_vacc.describe()
df.dr_recc_seasonal_vacc.value_counts()
df.dr_recc_seasonal_vacc.value_counts().sum()
26707-24547  #2160 are NA
df['dr_recc_seasonal_vacc'].fillna(df['dr_recc_seasonal_vacc'].mode()[0],inplace=True)
plt.hist(df.dr_recc_seasonal_vacc, bins = 'auto', facecolor = 'purple')
plt.xlabel('dr_recc_seasonal_vacc')
plt.ylabel('counts')
plt.title('Histogram of dr_recc_seasonal_vacc')
sns.countplot(df.dr_recc_seasonal_vacc)
df.dr_recc_seasonal_vacc.value_counts()
"""
Out[151]: 
0.0    18613
1.0     8094
Name: dr_recc_seasonal_vacc, dtype: int64
"""
dr_rec_se = pd.crosstab(df['dr_recc_seasonal_vacc'], df['h1n1_vaccine']) 
dr_rec_se
"""
h1n1_vaccine               0     1
dr_recc_seasonal_vacc             
0.0                    15758  2855
1.0                     5275  2819
"""
c, p, dof, expected = chi2_contingency(dr_rec_se)
p #Out[155]: 3.3126165190104603e-280
#P_value < 0.05 , so this is a significant variable

#Variable 12 - chronic_medic_condition
df.chronic_medic_condition.describe()
df.chronic_medic_condition.value_counts()
df.chronic_medic_condition.value_counts().sum()
26707-25736  #971 are NA
df['chronic_medic_condition'].fillna(df['chronic_medic_condition'].mode()[0],inplace=True)
plt.hist(df.chronic_medic_condition, bins = 'auto', facecolor = 'purple')
plt.xlabel('chronic_medic_condition')
plt.ylabel('counts')
plt.title('Histogram of chronic_medic_condition')
sns.countplot(df.chronic_medic_condition)
df.chronic_medic_condition.value_counts()
"""
0.0    19417
1.0     7290
Name: chronic_medic_condition, dtype: int64
"""
cmc = pd.crosstab(df['chronic_medic_condition'], df['h1n1_vaccine']) 
cmc
"""
h1n1_vaccine                 0     1
chronic_medic_condition             
0.0                      15751  3666
1.0                       5282  2008
"""
c, p, dof, expected = chi2_contingency(cmc)
p #Out[167]: 1.5428233060113362e-53
#P_value < 0.05 , so this is a significant variable

#Variable 13 - cont_child_undr_6_mnths
df.cont_child_undr_6_mnths.describe()
df.cont_child_undr_6_mnths.value_counts()
df.cont_child_undr_6_mnths.value_counts().sum()
26707-25887  #820 are NA
df['cont_child_undr_6_mnths'].fillna(df['cont_child_undr_6_mnths'].mode()[0],inplace=True)
plt.hist(df.cont_child_undr_6_mnths, bins = 'auto', facecolor = 'purple')
plt.xlabel('cont_child_undr_6_mnths')
plt.ylabel('counts')
plt.title('Histogram of cont_child_undr_6_mnths')
sns.countplot(df.cont_child_undr_6_mnths)
df.cont_child_undr_6_mnths.value_counts()
"""
0.0    24569
1.0     2138
Name: cont_child_undr_6_mnths, dtype: int64
"""
24569/26707 #91% obs are dominated by category 0 - having no regular contact with child
#So skip the vareiable and no need to chisquare

#Variable 14 - is_health_worker
df.is_health_worker.describe()
df.is_health_worker.value_counts()
df.is_health_worker.value_counts().sum()
26707-25903  #804 are NA
df['is_health_worker'].fillna(df['is_health_worker'].mode()[0],inplace=True)
plt.hist(df.is_health_worker, bins = 'auto', facecolor = 'purple')
plt.xlabel('is_health_worker')
plt.ylabel('counts')
plt.title('Histogram of is_health_worker')
sns.countplot(df.is_health_worker)
df.is_health_worker.value_counts()
"""
0.0    23808
1.0     2899
Name: is_health_worker, dtype: int64
"""
23808/26707 #89% obs are dominated by 0 - is not a health worker

#Variable 15 - has_health_insur
df.has_health_insur.describe()
df.has_health_insur.value_counts()
df.has_health_insur.value_counts().sum()
26707-14433  #12274 are NA
#Almost 50% values are missing ,so not adequate information , so skip this variable

#Variable 16 - is_h1n1_vacc_effective
df.is_h1n1_vacc_effective.describe()
df.is_h1n1_vacc_effective.value_counts()
df.is_h1n1_vacc_effective.value_counts().sum()
26707-26316  #391 are NA
df['is_h1n1_vacc_effective'].fillna(df['is_h1n1_vacc_effective'].mode()[0],inplace=True)
plt.hist(df.is_h1n1_vacc_effective, bins = 'auto', facecolor = 'purple')
plt.xlabel('is_h1n1_vacc_effective')
plt.ylabel('counts')
plt.title('Histogram of is_h1n1_vacc_effective')
sns.countplot(df.is_h1n1_vacc_effective)
df.is_h1n1_vacc_effective.value_counts()
"""
Out[198]: 
4.0    12074
5.0     7166
3.0     4723
2.0     1858
1.0      886
"""
df.is_h1n1_vacc_effective.value_counts().sum()
"""
#These categories can be clubbed together into 3
#0 - respondent thinking it is not effective(1) and
#1 - respondent thinking it is effective (2,4,5)
#2 - respondent who is uncertain about the effectiveness
df['is_h1n1_vacc_effective']=df.get('is_h1n1_vacc_effective').replace(1.0,0)
df['is_h1n1_vacc_effective']=df.get('is_h1n1_vacc_effective').replace(2.0,1)
df['is_h1n1_vacc_effective']=df.get('is_h1n1_vacc_effective').replace(3.0,2)
df['is_h1n1_vacc_effective']=df.get('is_h1n1_vacc_effective').replace(4.0,1)
df['is_h1n1_vacc_effective']=df.get('is_h1n1_vacc_effective').replace(5.0,1)
"""
df.is_h1n1_vacc_effective.value_counts()
4.0    12074
5.0     7166
3.0     4723
2.0     1858
1.0      886
Name: is_h1n1_vacc_effective, dtype: int64
"""
21098/26707 #79% are dominated by people thinking it is effective
#so skip this variable , chisquare is not required
"""
is_h1n1_vacc_effective_0 = df[df.h1n1_vaccine == 0]
is_h1n1_vacc_effective_1 = df[df.h1n1_vaccine == 1]
len(df.is_h1n1_vacc_effective[df.h1n1_vaccine == 0]) #14381
len(df.is_h1n1_vacc_effective[df.h1n1_vaccine == 1]) #4777
import scipy
scipy.stats.ttest_ind(is_h1n1_vacc_effective_0.is_h1n1_vacc_effective, is_h1n1_vacc_effective_1.is_h1n1_vacc_effective)
#Out[72]: Ttest_indResult(statistic=-45.34014817064191, pvalue=0.0)
#so this is significant

#Variable 17 - is_h1n1_risky
df.is_h1n1_risky.describe()
df.is_h1n1_risky.value_counts()
df.is_h1n1_risky.value_counts().sum()
26707-26319  #388 are NA
df['is_h1n1_risky'].fillna(df['is_h1n1_risky'].mode()[0],inplace=True)
plt.hist(df.is_h1n1_risky, bins = 'auto', facecolor = 'purple')
plt.xlabel('is_h1n1_risky')
plt.ylabel('counts')
plt.title('Histogram of is_h1n1_risky')
sns.countplot(df.is_h1n1_risky)
df.is_h1n1_risky.value_counts()
"""
2.0    10307
1.0     8139
4.0     5394
5.0     1750
3.0     1117
#Club and convert these 5 into 3 categories , 
#respindents who think it is risky - 1
#respindents who think it is not risky - 0
#respindents who are uncertain about whether it is risky or not risky - 2
df['is_h1n1_risky']=df.get('is_h1n1_risky').replace(1.0,0)
df['is_h1n1_risky']=df.get('is_h1n1_risky').replace(2.0,1)
df['is_h1n1_risky']=df.get('is_h1n1_risky').replace(3.0,2)
df['is_h1n1_risky']=df.get('is_h1n1_risky').replace(4.0,1)
df['is_h1n1_risky']=df.get('is_h1n1_risky').replace(5.0,1)
df.is_h1n1_risky.value_counts()
"""
1.0    17451
0.0     8139
2.0     1117
Name: is_h1n1_risky, dtype: int64
"""
#Use Chisquare test to see if this variable is significant or not
risk = pd.crosstab(df['is_h1n1_risky'], df['h1n1_vaccine']) 
risk
"""
h1n1_vaccine       0     1
is_h1n1_risky             
0.0             7420   719
1.0            12690  4761
2.0              923   194
"""
c, p, dof, expected = chi2_contingency(risk)
p #Out[220]: 3.731485681798496e-248
#P_value < 0.05 , so this is a significant variable
"""
is_h1n1_risky_0 = df[df.h1n1_vaccine == 0]
is_h1n1_risky_1 = df[df.h1n1_vaccine == 1]
len(df.is_h1n1_risky[df.h1n1_vaccine == 0]) #14381
len(df.is_h1n1_risky[df.h1n1_vaccine == 1]) #4777
import scipy
scipy.stats.ttest_ind(is_h1n1_risky_0.is_h1n1_risky, is_h1n1_risky_1.is_h1n1_risky)
#Out[79]: Ttest_indResult(statistic=-55.30715458501025, pvalue=0.0)
#so this is significant

#Variable 18 - sick_from_h1n1_vacc
df.sick_from_h1n1_vacc.describe()
df.sick_from_h1n1_vacc.value_counts()
df.sick_from_h1n1_vacc.value_counts().sum()
26707-26312  #395 are NA
df['sick_from_h1n1_vacc'].fillna(df['sick_from_h1n1_vacc'].mode()[0],inplace=True)
plt.hist(df.sick_from_h1n1_vacc, bins = 'auto', facecolor = 'purple')
plt.xlabel('sick_from_h1n1_vacc')
plt.ylabel('counts')
plt.title('Histogram of sick_from_h1n1_vacc')
sns.countplot(df.sick_from_h1n1_vacc)
df.sick_from_h1n1_vacc.value_counts()
"""
2.0    9524
1.0    8998
4.0    5850
5.0    2187
3.0     148
#Club and convert these 5 into 3 categories , 
#Respondents who are worried about falling sick after vaccine - 1
#Respondents who are not worried about falling sick after vaccine - 0
#Respondents who are uncertain - 2
df['sick_from_h1n1_vacc']=df.get('sick_from_h1n1_vacc').replace(1.0,0)
df['sick_from_h1n1_vacc']=df.get('sick_from_h1n1_vacc').replace(2.0,1)
df['sick_from_h1n1_vacc']=df.get('sick_from_h1n1_vacc').replace(3.0,2)
df['sick_from_h1n1_vacc']=df.get('sick_from_h1n1_vacc').replace(4.0,1)
df['sick_from_h1n1_vacc']=df.get('sick_from_h1n1_vacc').replace(5.0,1)
df.sick_from_h1n1_vacc.value_counts()
"""
1.0    17561
0.0     8998
2.0      148
Name: sick_from_h1n1_vacc, dtype: int64
"""
sick = pd.crosstab(df['sick_from_h1n1_vacc'], df['h1n1_vaccine']) 
sick
"""
h1n1_vaccine             0     1
sick_from_h1n1_vacc             
0.0                   7157  1841
1.0                  13740  3821
2.0                    136    12
"""
c, p, dof, expected = chi2_contingency(sick)
p #Out[238]: 2.3169925261072866e-05
#P_value < 0.05 , so this is a significant variable
"""
sick_from_h1n1_vacc_0 = df[df.h1n1_vaccine == 0]
sick_from_h1n1_vacc_1 = df[df.h1n1_vaccine == 1]
len(df.h1n1_vaccine[df.h1n1_vaccine == 0]) #14381
len(df.h1n1_vaccine[df.h1n1_vaccine == 1]) #4777
import scipy
scipy.stats.ttest_ind(sick_from_h1n1_vacc_0.sick_from_h1n1_vacc, sick_from_h1n1_vacc_1.sick_from_h1n1_vacc)
#Out[86]: Ttest_indResult(statistic=-12.221685787692287, pvalue=2.939896153851835e-34)
#this is significant variable

#Variable 19 - is_seas_vacc_effective
df.is_seas_vacc_effective.describe()
df.is_seas_vacc_effective.value_counts()
df.is_seas_vacc_effective.value_counts().sum()
26707-26245  #462 are NA
df['is_seas_vacc_effective'].fillna(df['is_seas_vacc_effective'].mode()[0],inplace=True)
plt.hist(df.is_seas_vacc_effective, bins = 'auto', facecolor = 'purple')
plt.xlabel('is_seas_vacc_effective')
plt.ylabel('counts')
plt.title('Histogram of is_seas_vacc_effective')
sns.countplot(df.is_seas_vacc_effective)
df.is_seas_vacc_effective.value_counts()
"""
4.0    12091
5.0     9973
2.0     2206
1.0     1221
3.0     1216
#Club and convert these 5 into 3 categories , 
#respondent thinks seasonal vaccine effective - 1 ,
#respondent thinks seasonal vaccine is not effective - 0
#respondent uncertain about effectiveness of seasonal vaccine - 2
df['is_seas_vacc_effective']=df.get('is_seas_vacc_effective').replace(1.0,0)
df['is_seas_vacc_effective']=df.get('is_seas_vacc_effective').replace(2.0,1)
df['is_seas_vacc_effective']=df.get('is_seas_vacc_effective').replace(3.0,2)
df['is_seas_vacc_effective']=df.get('is_seas_vacc_effective').replace(4.0,1)
df['is_seas_vacc_effective']=df.get('is_seas_vacc_effective').replace(5.0,1)
df.is_seas_vacc_effective.value_counts()
"""
1.0    24270
0.0     1221
2.0     1216
Name: is_seas_vacc_effective, dtype: int64
"""
24270/26707 #90% people think seasonal vaccine is effective
#very dominant , so skip this variable
"""
is_seas_vacc_effective_0 = df[df.h1n1_vaccine == 0]
is_seas_vacc_effective_1 = df[df.h1n1_vaccine == 1]
import scipy
scipy.stats.ttest_ind(is_seas_vacc_effective_0.is_seas_vacc_effective, is_seas_vacc_effective_1.is_seas_vacc_effective)
#Out[105]: Ttest_indResult(statistic=-29.52568048418476, pvalue=1.4482499412706185e-188)
#significant

#Variable 20 - is_seas_risky
df.is_seas_risky.describe()
df.is_seas_risky.value_counts()
df.is_seas_risky.value_counts().sum()
26707-26193  #514 are NA
df['is_seas_risky'].fillna(df['is_seas_risky'].mode()[0],inplace=True)
plt.hist(df.is_seas_risky, bins = 'auto', facecolor = 'purple')
plt.xlabel('is_seas_risky')
plt.ylabel('counts')
plt.title('Histogram of is_seas_risky')
sns.countplot(df.is_seas_risky)
df.is_seas_risky.value_counts()
"""
2.0    9468
4.0    7630
1.0    5974
5.0    2958
3.0     677
#Club and convert these 5 into 3 categories , 
#respondents who think there is no risk - 0
#respondents who think there is risk - 1
#respondents who are uncertain whether it is risk or no risk - 2
df['is_seas_risky']=df.get('is_seas_risky').replace(1.0,0)
df['is_seas_risky']=df.get('is_seas_risky').replace(2.0,1)
df['is_seas_risky']=df.get('is_seas_risky').replace(3.0,2)
df['is_seas_risky']=df.get('is_seas_risky').replace(4.0,1)
df['is_seas_risky']=df.get('is_seas_risky').replace(5.0,1)
df.is_seas_risky.value_counts()
1.0    20056
0.0     5974
2.0      677
Name: is_seas_risky, dtype: int64
seas_risk = pd.crosstab(df['is_seas_risky'], df['h1n1_vaccine']) 
seas_risk
h1n1_vaccine       0     1
is_seas_risky             
0.0             5468   506
1.0            15024  5032
2.0              541   136
c, p, dof, expected = chi2_contingency(seas_risk)
p #Out[274]: 7.493309218217299e-166
#P_value < 0.05 , so this is a significant variable
"""
is_seas_risky_0 = df[df.h1n1_vaccine == 0]
is_seas_risky_1 = df[df.h1n1_vaccine == 1]
import scipy
scipy.stats.ttest_ind(is_seas_risky_0.is_seas_risky, is_seas_risky_1.is_seas_risky)
#Out[114]: Ttest_indResult(statistic=-43.25403475203991, pvalue=0.0)
#significant

#Variable 21 - sick_from_seas_vacc
df.sick_from_seas_vacc.describe()
df.sick_from_seas_vacc.value_counts()
df.sick_from_seas_vacc.value_counts().sum()
26707-26170  #537 are NA
df['sick_from_seas_vacc'].fillna(df['sick_from_seas_vacc'].mode()[0],inplace=True)
plt.hist(df.sick_from_seas_vacc, bins = 'auto', facecolor = 'purple')
plt.xlabel('sick_from_seas_vacc')
plt.ylabel('counts')
plt.title('Histogram of sick_from_seas_vacc')
sns.countplot(df.sick_from_seas_vacc)
df.sick_from_seas_vacc.value_counts()
"""
1.0    12407
2.0     7633
4.0     4852
5.0     1721
3.0       94
Name: sick_from_seas_vacc, dtype: int64
#Club and convert these 5 into 3 categories , 
#respondents who are not worried at all - 0
#respondents who are worried - 1
#respondents who are uncertain  - 2
df['sick_from_seas_vacc']=df.get('sick_from_seas_vacc').replace(1.0,0)
df['sick_from_seas_vacc']=df.get('sick_from_seas_vacc').replace(2.0,1)
df['sick_from_seas_vacc']=df.get('sick_from_seas_vacc').replace(3.0,2)
df['sick_from_seas_vacc']=df.get('sick_from_seas_vacc').replace(4.0,1)
df['sick_from_seas_vacc']=df.get('sick_from_seas_vacc').replace(5.0,1)
df.sick_from_seas_vacc.value_counts()
"""
1.0    14206
0.0    12407
2.0       94
"""
seas_sick = pd.crosstab(df['sick_from_seas_vacc'], df['h1n1_vaccine']) 
seas_sick
h1n1_vaccine             0     1
sick_from_seas_vacc             
0.0                   9691  2716
1.0                  11256  2950
2.0                     86     8
c, p, dof, expected = chi2_contingency(seas_sick)
p #Out[292]: 0.0008447992242921546
#P_value < 0.05 , so this is a significant variable
"""
sick_from_seas_vacc_0 = df[df.h1n1_vaccine == 0]
sick_from_seas_vacc_1 = df[df.h1n1_vaccine == 1]
import scipy
scipy.stats.ttest_ind(sick_from_seas_vacc_0.sick_from_seas_vacc, sick_from_seas_vacc_1.sick_from_seas_vacc)
#Out[119]: Ttest_indResult(statistic=-1.546822736573835, pvalue=0.12191784569504283)
#not a significant variable

#Variable 22 - age_bracket
df.age_bracket.describe()
"""
count         26707
unique            5
top       65+ Years
freq           6843
Name: age_bracket, dtype: object
"""
df.age_bracket.value_counts()
"""
65+ Years        6843
55 - 64 Years    5563
45 - 54 Years    5238
18 - 34 Years    5215
35 - 44 Years    3848
"""
df.age_bracket.value_counts().sum()
26707-26707  #NO NA
plt.hist(df.age_bracket, bins = 'auto', facecolor = 'purple')
plt.xlabel('age_bracket')
plt.ylabel('counts')
plt.title('Histogram of age_bracket')
sns.countplot(df.age_bracket)
df.age_bracket.value_counts()
"""
65+ Years        6843
55 - 64 Years    5563
45 - 54 Years    5238
18 - 34 Years    5215
35 - 44 Years    3848
Name: age_bracket, dtype: int64
"""
#Club and convert these 5 into 3 categories , 
#18-34 years - 1
#35-55 years - 2
#55+ - 3
df['age_bracket']=df.get('age_bracket').replace('18 - 34 Years',1)
df['age_bracket']=df.get('age_bracket').replace('35 - 44 Years',2)
df['age_bracket']=df.get('age_bracket').replace('45 - 54 Years',2)
df['age_bracket']=df.get('age_bracket').replace('55 - 64 Years',3)
df['age_bracket']=df.get('age_bracket').replace('65+ Years',3)
df.age_bracket.value_counts()
"""
3    12406
2     9086
1     5215
Name: age_bracket, dtype: int64
"""
age_bracket_0 = df[df.h1n1_vaccine == 0]
age_bracket_1 = df[df.h1n1_vaccine == 1]
import scipy
scipy.stats.ttest_ind(age_bracket_0.age_bracket, age_bracket_1.age_bracket)
#Out[129]: Ttest_indResult(statistic=-7.488711573264192, pvalue=7.170289718650563e-14)
#significant variable

#Variable 23 - qualification
df.qualification.describe()
"""
count                25300
unique                   4
top       College Graduate
freq                 10097
Name: qualification, dtype: object
"""
df.qualification.value_counts()

College Graduate    10097
Some College         7043
12 Years             5797
< 12 Years           2363
Name: qualification, dtype: int64

df.qualification.value_counts().sum()
26707-25300  #1407 NA
df['qualification'].fillna(df['qualification'].mode()[0],inplace=True)
plt.hist(df.qualification, bins = 'auto', facecolor = 'purple')
plt.xlabel('qualification')
plt.ylabel('counts')
plt.title('Histogram of qualification')
sns.countplot(df.qualification)
df.qualification.value_counts()
"""
College Graduate    11504
Some College         7043
12 Years             5797
< 12 Years           2363
Name: qualification, dtype: int64
"""
#Club and convert these 5 into 2 categories , 
#group of age <=12 - 0
#group who is attending or has attended college - 1
df['qualification']=df.get('qualification').replace('< 12 Years',0)
df['qualification']=df.get('qualification').replace('12 Years',1)
df['qualification']=df.get('qualification').replace('Some College',2)
df['qualification']=df.get('qualification').replace('College Graduate',3)
df.qualification.value_counts()
df.qualification.value_counts().sum()
"""
3    11504
2     7043
1     5797
0     2363
Name: qualification, dtype: int64
"""
qualification_0 = df[df.h1n1_vaccine == 0]
qualification_1 = df[df.h1n1_vaccine == 1]
import scipy
scipy.stats.ttest_ind(qualification_0.qualification, qualification_1.qualification)
#Out[186]: Ttest_indResult(statistic=-10.108451777813876, pvalue=5.59748641872047e-24)
#significant

#Variable 24 - race
df.race.describe()
"""
count     26707
unique        4
top       White
freq      21222
Name: race, dtype: object
"""
df.race.value_counts()
"""
White                21222
Black                 2118
Hispanic              1755
Other or Multiple     1612
Name: race, dtype: int64
"""
df.race.value_counts().sum()
#26707 No NA
plt.hist(df.race, bins = 'auto', facecolor = 'purple')
plt.xlabel('race')
plt.ylabel('counts')
plt.title('Histogram of race')
sns.countplot(df.race)
df.race.value_counts()
"""
White                21222
Black                 2118
Hispanic              1755
Other or Multiple     1612
Name: race, dtype: int64
"""
21222/26707 #80% dominance by white race , so this variable can be skipped

#Variable 25 - sex
df.sex.describe()
"""
count      26707
unique         2
top       Female
freq       15858
Name: sex, dtype: object
"""
df.sex.value_counts()
"""
Female    15858
Male      10849
Name: sex, dtype: int64
"""
df.sex.value_counts().sum()
#26707 NO NA
plt.hist(df.sex, bins = 'auto', facecolor = 'purple')
plt.xlabel('sex')
plt.ylabel('counts')
plt.title('Histogram of sex')
sns.countplot(df.sex)
df.sex.value_counts()
"""
Female    15858
Male      10849
Name: sex, dtype: int64
"""
#Use chisquare test to check the significance
sex = pd.crosstab(df['sex'], df['h1n1_vaccine']) 
sex
"""
h1n1_vaccine      0     1
sex                      
Female        12378  3480
Male           8655  2194
"""
c, p, dof, expected = chi2_contingency(sex)
p #Out[351]: 0.0007709155489949327
#P_value < 0.05 , so this is a significant variable
df['sex']=df.get('sex').replace('Male',0)
df['sex']=df.get('sex').replace('Female',1)

#Variable 26 - income_level
df.income_level.describe()
"""
count                         22284
unique                            3
top       <= $75,000, Above Poverty
freq                          12777
Name: income_level, dtype: object
"""
df.income_level.value_counts()
"""
<= $75,000, Above Poverty    12777
> $75,000                     6810
Below Poverty                 2697
Name: income_level, dtype: int64
"""
df.income_level.value_counts().sum()
26707-22284  #4423 NA , replace with mode
df['income_level'].fillna(df['income_level'].mode()[0],inplace=True)
plt.hist(df.income_level, bins = 'auto', facecolor = 'purple')
plt.xlabel('income_level')
plt.ylabel('counts')
plt.title('Histogram of income_level')
sns.countplot(df.income_level)
df.income_level.value_counts()
"""
<= $75,000, Above Poverty    17200
> $75,000                     6810
Below Poverty                 2697
Name: income_level, dtype: int64
"""
#Club and convert these 3 into 3 ordinal , 
#Below Poverty - 1
#Above Poverty - 0
df['income_level']=df.get('income_level').replace('<= $75,000, Above Poverty',0)
df['income_level']=df.get('income_level').replace('> $75,000',1)
df['income_level']=df.get('income_level').replace('Below Poverty',2)
df.income_level.value_counts()
"""
0    17200
1     6810
2     2697
Name: income_level, dtype: int64
"""
#use independent t test to see the significance
income_level_0 = df[df.h1n1_vaccine == 0]
income_level_1 = df[df.h1n1_vaccine == 1]
import scipy
scipy.stats.ttest_ind(income_level_0.income_level, income_level_1.income_level)
#Out[139]: Ttest_indResult(statistic=-3.6177372921783095, pvalue=0.00029773785688863753)
#significant

#Variable 27 - marital_status
df.marital_status.describe()
"""
count       25299
unique          2
top       Married
freq        13555
Name: marital_status, dtype: object
"""
df.marital_status.value_counts()
"""
Married        13555
Not Married    11744
Name: marital_status, dtype: int64
"""
df.marital_status.value_counts().sum()
26707 - 25299 #1408 NA , replace by mode
df['marital_status'].fillna(df['marital_status'].mode()[0],inplace=True)
plt.hist(df.marital_status, bins = 'auto', facecolor = 'purple')
plt.xlabel('marital_status')
plt.ylabel('counts')
plt.title('Histogram of marital_status')
sns.countplot(df.marital_status)
df.marital_status.value_counts()
"""
Married        14963
Not Married    11744
Name: marital_status, dtype: int64
"""
#Use chisquare test to check the significance
marital = pd.crosstab(df['marital_status'], df['h1n1_vaccine']) 
marital
"""
h1n1_vaccine        0     1
marital_status             
Married         11539  3424
Not Married      9494  2250
"""
c, p, dof, expected = chi2_contingency(marital)
p #Out[377]: 1.6985751321912323e-13
#P_value < 0.05 , so this is a significant variable
df['marital_status']=df.get('marital_status').replace('Not Married',0)
df['marital_status']=df.get('marital_status').replace('Married',1)

#Variable 28 - housing_status
df.housing_status.describe()
"""
count     24665
unique        2
top         Own
freq      18736
Name: housing_status, dtype: object
"""
df.housing_status.value_counts()
"""
Own     18736
Rent     5929
Name: housing_status, dtype: int64
"""
df.housing_status.value_counts().sum()
26707 - 24665 #2402 NA , replace by mode
df['housing_status'].fillna(df['housing_status'].mode()[0],inplace=True)
plt.hist(df.housing_status, bins = 'auto', facecolor = 'purple')
plt.xlabel('housing_status')
plt.ylabel('counts')
plt.title('Histogram of housing_status')
sns.countplot(df.housing_status)
df.housing_status.value_counts()
"""
Own     20778
Rent     5929
Name: housing_status, dtype: int64
"""
df['housing_status']=df.get('housing_status').replace('Own',1)
df['housing_status']=df.get('housing_status').replace('Rent',0)
#Use chisquare test to check the significance
housing = pd.crosstab(df['housing_status'], df['h1n1_vaccine']) 
housing
"""
h1n1_vaccine        0     1
housing_status             
0                4810  1119
1               16223  4555
"""
c, p, dof, expected = chi2_contingency(housing)
p #Out[391]: 4.5507157863887266e-07
#P_value < 0.05 , so this is a significant variable
df['housing_status']=df.get('housing_status').replace('Own',0)
df['housing_status']=df.get('housing_status').replace('Rent',1)


#Variable 29 - employment
df.employment.describe()
"""
count        25244
unique           3
top       Employed
freq         13560
Name: employment, dtype: object
"""
df.employment.value_counts()
"""
Employed              13560
Not in Labor Force    10231
Unemployed             1453
Name: employment, dtype: int64
"""
df.employment.value_counts().sum()
26707 - 25244 #1463 NA , replace by mode
df['employment'].fillna(df['employment'].mode()[0],inplace=True)
plt.hist(df.employment, bins = 'auto', facecolor = 'purple')
plt.xlabel('employment')
plt.ylabel('counts')
plt.title('Histogram of employment')
sns.countplot(df.employment)
df.employment.value_counts()
"""
Employed              15023
Not in Labor Force    10231
Unemployed             1453
Name: employment, dtype: int64
"""
#Club these 3 categories into 3 , employed - 1 , unemployed - 0
df['employment']=df.get('employment').replace('Employed',1)
df['employment']=df.get('employment').replace('Not in Labor Force',2)
df['employment']=df.get('employment').replace('Unemployed',0)
#Use chisquare test to check the significance
emp = pd.crosstab(df['employment'], df['h1n1_vaccine']) 
emp
"""
h1n1_vaccine      0     1
employment               
0              1216   237
1             11829  3194
2              7988  2243
"""
c, p, dof, expected = chi2_contingency(emp)
p #Out[406]: 6.274688521364308e-06
#P_value < 0.05 , so this is a significant variable

#Variable 30 - census_msa
df.census_msa.describe()
"""
count                        26707
unique                           3
top       MSA, Not Principle  City
freq                         11645
Name: census_msa, dtype: object
"""
df.census_msa.value_counts()
"""
MSA, Not Principle  City    11645
MSA, Principle City          7864
Non-MSA                      7198
Name: census_msa, dtype: int64
"""
df.census_msa.value_counts().sum()
26707 #No NA 
plt.hist(df.census_msa, bins = 'auto', facecolor = 'purple')
plt.xlabel('census_msa')
plt.ylabel('counts')
plt.title('Histogram of census_msa')
sns.countplot(df.census_msa)
df.census_msa.value_counts()
"""
MSA, Not Principle  City    11645
MSA, Principle City          7864
Non-MSA                      7198
Name: census_msa, dtype: int64
"""
#Club these 3 categories into 2 , MSA - 1 , Non-MSA - 0
df['census_msa']=df.get('census_msa').replace('MSA, Not Principle  City',1)
df['census_msa']=df.get('census_msa').replace('MSA, Principle City',1)
df['census_msa']=df.get('census_msa').replace('Non-MSA',0)
#Use chisquare test to check the significance
msa = pd.crosstab(df['census_msa'], df['h1n1_vaccine']) 
msa
"""
h1n1_vaccine      0     1
census_msa               
0              5672  1526
1             15361  4148
"""
c, p, dof, expected = chi2_contingency(msa)
p #Out[419]: 0.9263581190403177
#P_value > 0.05 , so this is a not significant variable

#Variable 31 - no_of_adults
df.no_of_adults.describe()
"""
count    26707.000000
mean         0.887558
std          0.749980
min          0.000000
25%          0.000000
50%          1.000000
75%          1.000000
max          3.000000
Name: no_of_adults, dtype: float64
"""
df['no_of_adults'].fillna(df['no_of_adults'].mode()[0],inplace=True)
df.no_of_adults.value_counts()
"""
1.0    14723
0.0     8056
2.0     2803
3.0     1125
Name: no_of_adults, dtype: int64
"""
df.no_of_adults.value_counts().sum()
26707 #NO NA 
plt.hist(df.no_of_adults, bins = 'auto', facecolor = 'purple')
plt.xlabel('no_of_adults')
plt.ylabel('counts')
plt.title('Histogram of no_of_adults')
sns.countplot(df.no_of_adults)
df.no_of_adults.value_counts()
"""
1.0    14723
0.0     8056
2.0     2803
3.0     1125
Name: no_of_adults, dtype: int64
#Club these 4 categories into 2 , adults are there or not - Yes / No
df['no_of_adults']=df.get('no_of_adults').replace(1.0,1)
df['no_of_adults']=df.get('no_of_adults').replace(2.0,1)
df['no_of_adults']=df.get('no_of_adults').replace(3.0,1)
df['no_of_adults']=df.get('no_of_adults').replace(0.0,0)
df.no_of_adults.value_counts()
"""
1.0    18651
0.0     8056
"""
#Use chisquare test to check the significance
ad = pd.crosstab(df['no_of_adults'], df['h1n1_vaccine']) 
ad
h1n1_vaccine      0     1
no_of_adults             
0.0            6471  1585
1.0           14562  4089
c, p, dof, expected = chi2_contingency(ad)
p #Out[453]: 3.9969241703939635e-05
#P_value < 0.05 , so this is a significant variable

no_of_adults_0 = df[df.h1n1_vaccine == 0]
no_of_adults_1 = df[df.h1n1_vaccine == 1]
import scipy
scipy.stats.ttest_ind(no_of_adults_0.h1n1_vaccine, no_of_adults_1.h1n1_vaccine)
#Out[147]: Ttest_indResult(statistic=-inf, pvalue=0.0)
#significant

#Variable 32 - no_of_children
df.no_of_children.describe()
"""
count    26458.000000
mean         0.534583
std          0.928173
min          0.000000
25%          0.000000
50%          0.000000
75%          1.000000
max          3.000000
Name: no_of_children, dtype: float64
df['no_of_children'].fillna(df['no_of_children'].mode()[0],inplace=True)
"""
df.no_of_children.value_counts()
"""
0.0    18672
1.0     3175
2.0     2864
3.0     1747
Name: no_of_children, dtype: int64
"""
df.no_of_children.value_counts().sum()
26707-26458 #249 NA ,replace with mode
df['no_of_children'].fillna(df['no_of_children'].mode()[0],inplace=True)
plt.hist(df.no_of_children, bins = 'auto', facecolor = 'purple')
plt.xlabel('no_of_children')
plt.ylabel('counts')
plt.title('Histogram of no_of_children')
sns.countplot(df.no_of_children)
df.no_of_children.value_counts()
"""
0.0    18921
1.0     3175
2.0     2864
3.0     1747
Name: no_of_children, dtype: int64
"""
"""
#Club these 4 categories into 2 , childrens are there or not - Yes / No
df['no_of_children']=df.get('no_of_children').replace(1.0,1)
df['no_of_children']=df.get('no_of_children').replace(2.0,1)
df['no_of_children']=df.get('no_of_children').replace(3.0,1)
df['no_of_children']=df.get('no_of_children').replace(0.0,0)
df.no_of_children.value_counts()
"""
0.0    18921
1.0     7786
Name: no_of_children, dtype: int64
"""
#Use chisquare test to check the significance
chi = pd.crosstab(df['no_of_children'], df['h1n1_vaccine']) 
chi
"""
h1n1_vaccine        0     1
no_of_children             
0.0             14899  4022
1.0              6134  1652
"""
c, p, dof, expected = chi2_contingency(chi)
p #Out[473]: 0.9563133750418122
#P_value > 0.05 , so this is not a significant variable
"""
no_of_children_0 = df[df.h1n1_vaccine == 0]
no_of_children_1 = df[df.h1n1_vaccine == 1]
import scipy
scipy.stats.ttest_ind(no_of_children_0.h1n1_vaccine, no_of_children_1.h1n1_vaccine)
#Out[154]: Ttest_indResult(statistic=-inf, pvalue=0.0)
#significant

#Variable 33 - h1n1_vaccine
df.h1n1_vaccine.describe()
"""
count    26707.000000
mean         0.212454
std          0.409052
min          0.000000
25%          0.000000
50%          0.000000
75%          0.000000
max          1.000000
Name: h1n1_vaccine, dtype: float64

df.h1n1_vaccine.value_counts()
"""
Out[477]: 
0    21033
1     5674
Name: h1n1_vaccine, dtype: int64
"""
df.h1n1_vaccine.value_counts().sum()
#26707 - no NA values

###############################################################################
#now check the if the data is balanced or imbalanced
df.h1n1_vaccine.value_counts()

0    21033
1     5674
Name: h1n1_vaccine, dtype: int64

#data is imbalanced , so use smote to balance the data

import statsmodels.api as sm
import statsmodels.formula.api as smf
log_mod = smf.glm(formula='''h1n1_vaccine~h1n1_worry+h1n1_awareness+contact_avoidance+
          avoid_large_gatherings+reduced_outside_home_cont+avoid_touch_face+
          dr_recc_h1n1_vacc+dr_recc_seasonal_vacc+chronic_medic_condition+is_h1n1_vacc_effective+
          is_h1n1_risky+sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky+
          age_bracket+qualification+sex+income_level+marital_status+housing_status+
          employment+no_of_adults+no_of_children''',data = df, family=sm.families.Binomial())
result = log_mod.fit()
print(result.summary())
"""
=============================================================================================
                                coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------------------
Intercept                    -6.7163      0.148    -45.495      0.000      -7.006      -6.427
h1n1_worry                   -0.1172      0.023     -4.994      0.000      -0.163      -0.071
h1n1_awareness                0.2003      0.031      6.488      0.000       0.140       0.261
contact_avoidance            -0.0751      0.044     -1.719      0.086      -0.161       0.011
avoid_large_gatherings       -0.1979      0.046     -4.315      0.000      -0.288      -0.108
reduced_outside_home_cont    -0.0308      0.046     -0.663      0.507      -0.122       0.060
avoid_touch_face              0.0847      0.042      1.998      0.046       0.002       0.168
dr_recc_h1n1_vacc             2.0263      0.052     38.768      0.000       1.924       2.129
dr_recc_seasonal_vacc        -0.4518      0.051     -8.807      0.000      -0.552      -0.351
chronic_medic_condition       0.0733      0.040      1.818      0.069      -0.006       0.152
is_h1n1_vacc_effective        0.6088      0.024     24.862      0.000       0.561       0.657
is_h1n1_risky                 0.3878      0.017     23.200      0.000       0.355       0.421
sick_from_h1n1_vacc          -0.0458      0.014     -3.185      0.001      -0.074      -0.018
is_seas_vacc_effective        0.0994      0.022      4.468      0.000       0.056       0.143
is_seas_risky                 0.1490      0.016      9.230      0.000       0.117       0.181
age_bracket                   0.1779      0.029      6.151      0.000       0.121       0.235
qualification                 0.1122      0.020      5.651      0.000       0.073       0.151
sex                          -0.0863      0.037     -2.331      0.020      -0.159      -0.014
income_level                  0.0054      0.028      0.197      0.844      -0.049       0.059
marital_status                0.1537      0.042      3.660      0.000       0.071       0.236
housing_status                0.0871      0.048      1.821      0.069      -0.007       0.181
employment                    0.0346      0.033      1.036      0.300      -0.031       0.100
no_of_adults                 -0.0140      0.027     -0.514      0.607      -0.067       0.039
no_of_children               -0.0575      0.022     -2.615      0.009      -0.101      -0.014
=============================================================================================

"""
#Remove those whose p value is greater than 0.05
"""
no_of_adults
income_level
chronic_medic_condition
reduced_outside_home_cont
contact_avoidance
housing_status
employment
avoid_touch_face
"""


log_mod = smf.glm(formula='''h1n1_vaccine~h1n1_worry+h1n1_awareness+
          avoid_large_gatherings+
          dr_recc_h1n1_vacc+dr_recc_seasonal_vacc+is_h1n1_vacc_effective+
          is_h1n1_risky+sick_from_h1n1_vacc+is_seas_vacc_effective+is_seas_risky+
          age_bracket+qualification+sex+marital_status+no_of_children''',data = df, family=sm.families.Binomial())
result = log_mod.fit()
print(result.summary())


import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X = df[['h1n1_worry','h1n1_awareness','avoid_large_gatherings',
          'dr_recc_h1n1_vacc','dr_recc_seasonal_vacc','is_h1n1_vacc_effective',
          'is_h1n1_risky','sick_from_h1n1_vacc','is_seas_vacc_effective','is_seas_risky',
          'age_bracket','qualification','sex','marital_status','no_of_children']]
y = df['h1n1_vaccine']

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

from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()
lg = log_regression.fit(X_train,y_train)

#use model to make predictions on test data
y_pred = log_regression.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.8236615499812804
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.show()
#81.23
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
print('Log loss = {:.5f}'.format(log_loss(y_test, y_pred_proba)))
#Log loss = 0.40648
print('AUC = {:.5f}'.format(roc_auc_score(y_test, y_pred_proba)))
#AUC = 0.81235
print('Average Precision = {:.5f}'.format(average_precision_score(y_test, y_pred_proba)))
#Average Precision = 0.58226
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, y_pred)))
#Accuracy = 0.82366
print('Precision = {:.5f}'.format(precision_score(y_test, y_pred)))
#Precision = 0.66667
print('Recall = {:.5f}'.format(recall_score(y_test, y_pred)))
#Recall = 0.38974
print('F1 score = {:.5f}'.format(f1_score(y_test, y_pred)))
#F1 score = 0.49191
print(classification_report(y_test, y_pred))

