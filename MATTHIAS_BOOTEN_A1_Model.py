#!/usr/bin/env python
# coding: utf-8

# <h3>Read in the dataset.</h3><br>
# Importing appropriate libraries, setting Panda print options and reading in the file as a dataset.

# In[1]:


# Import libraries

import pandas                  as pd                      # data science essentials
import matplotlib.pyplot       as plt                     # essential graphical output
import seaborn                 as sns                     # enhanced graphical outputimport pandas as pd
import statsmodels.formula.api as smf                     # regression modeling

from   sklearn.model_selection import train_test_split    # train/test split
from   sklearn.linear_model    import LinearRegression    # linear regression (scikit-learn)
import sklearn.linear_model
from   sklearn.neighbors       import KNeighborsRegressor # KNN for Regression
from   sklearn.preprocessing   import StandardScaler      # standard scaler

# Set pandas print options

pd.set_option('display.max_rows'   , 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width'      , 1000)

# Specify file name
file = 'Apprentice_Chef_Dataset.xlsx'


# Read the file into Python
chefdf = pd.read_excel(file)


# <h3>Feature Engineering: Outlier Analysis</h3><br>

# In[2]:


#TOTAL_MEALS_ORDERED
TOTAL_MEALS_ORDERED_lo = 30
TOTAL_MEALS_ORDERED_hi = 200

#UNIQUE_MEALS_PURCH
UNIQUE_MEALS_PURCH_lo = 1
UNIQUE_MEALS_PURCH_hi = 9

#CONTACTS_W_CUSTOMER_SERVICE
CONTACTS_W_CUSTOMER_SERVICE_lo = 4.0
CONTACTS_W_CUSTOMER_SERVICE_hi = 8.0

#AVG_TIME_PER_SITE_VISIT
AVG_TIME_PER_SITE_VISIT_lo = 0
AVG_TIME_PER_SITE_VISIT_hi = 200

#WEEKLY_PLAN
WEEKLY_PLAN_lo = 0
WEEKLY_PLAN_hi = 15

#EARLY_DELIVERIES
EARLY_DELIVERIES_lo = 0
EARLY_DELIVERIES_hi = 4 

#LATE_DELIVERIES
LATE_DELIVERIES_lo = 0
LATE_DELIVERIES_hi = 6

#AVG_PREP_VID_TIME
AVG_PREP_VID_TIME_lo = 100
AVG_PREP_VID_TIME_hi = 200

#LARGEST_ORDER_SIZE
LARGEST_ORDER_SIZE_lo = 2
LARGEST_ORDER_SIZE_hi = 6

#MEDIAN_MEAL_RATING
MEDIAN_MEAL_RATING_lo = 2
MEDIAN_MEAL_RATING_hi = 4

#AVG_CLICKS_PER_VISIT
AVG_CLICKS_PER_VISIT_lo = 11
AVG_CLICKS_PER_VISIT_hi = 17

##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# Developing features (columns) for outliers

# TOTAL_MEALS_ORDERED

chefdf['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = chefdf.loc[0:,'out_TOTAL_MEALS_ORDERED'][chefdf['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]
condition_lo = chefdf.loc[0:,'out_TOTAL_MEALS_ORDERED'][chefdf['TOTAL_MEALS_ORDERED'] < TOTAL_MEALS_ORDERED_lo]

chefdf['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# UNIQUE_MEALS_PURCH

chefdf['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = chefdf.loc[0:,'out_UNIQUE_MEALS_PURCH'][chefdf['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]
condition_lo = chefdf.loc[0:,'out_UNIQUE_MEALS_PURCH'][chefdf['UNIQUE_MEALS_PURCH'] < UNIQUE_MEALS_PURCH_lo]

chefdf['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE

chefdf['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = chefdf.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][chefdf['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]
condition_lo = chefdf.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][chefdf['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_lo]

chefdf['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)


# AVG_TIME_PER_SITE_VISIT

chefdf['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = chefdf.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][chefdf['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]
condition_lo = chefdf.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][chefdf['AVG_TIME_PER_SITE_VISIT'] < AVG_TIME_PER_SITE_VISIT_lo]

chefdf['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# WEEKLY_PLAN

chefdf['out_WEEKLY_PLAN'] = 0
condition_hi = chefdf.loc[0:,'out_WEEKLY_PLAN'][chefdf['WEEKLY_PLAN'] > WEEKLY_PLAN_hi]
condition_lo = chefdf.loc[0:,'out_WEEKLY_PLAN'][chefdf['WEEKLY_PLAN'] < WEEKLY_PLAN_lo]

chefdf['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_WEEKLY_PLAN'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# EARLY_DELIVERIES
chefdf['out_EARLY_DELIVERIES'] = 0
condition_hi = chefdf.loc[0:,'out_EARLY_DELIVERIES'][chefdf['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]
condition_lo = chefdf.loc[0:,'out_EARLY_DELIVERIES'][chefdf['EARLY_DELIVERIES'] < EARLY_DELIVERIES_lo]

chefdf['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_EARLY_DELIVERIES'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# LATE_DELIVERIES

chefdf['out_LATE_DELIVERIES'] = 0
condition_hi = chefdf.loc[0:,'out_LATE_DELIVERIES'][chefdf['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]
condition_lo = chefdf.loc[0:,'out_LATE_DELIVERIES'][chefdf['LATE_DELIVERIES'] < LATE_DELIVERIES_lo]

chefdf['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_LATE_DELIVERIES'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# AVG_PREP_VID_TIME

chefdf['out_AVG_PREP_VID_TIME'] = 0
condition_hi = chefdf.loc[0:,'out_AVG_PREP_VID_TIME'][chefdf['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]
condition_lo = chefdf.loc[0:,'out_AVG_PREP_VID_TIME'][chefdf['AVG_PREP_VID_TIME'] < AVG_PREP_VID_TIME_lo]

chefdf['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# LARGEST_ORDER_SIZE

chefdf['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = chefdf.loc[0:,'out_LARGEST_ORDER_SIZE'][chefdf['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]
condition_lo = chefdf.loc[0:,'out_LARGEST_ORDER_SIZE'][chefdf['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_lo]

chefdf['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# MEDIAN_MEAL_RATING

chefdf['out_MEDIAN_MEAL_RATING'] = 0
condition_hi = chefdf.loc[0:,'out_MEDIAN_MEAL_RATING'][chefdf['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]
condition_lo = chefdf.loc[0:,'out_MEDIAN_MEAL_RATING'][chefdf['MEDIAN_MEAL_RATING'] < MEDIAN_MEAL_RATING_lo]

chefdf['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)

# AVG_CLICKS_PER_VISIT

chefdf['out_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = chefdf.loc[0:,'out_AVG_CLICKS_PER_VISIT'][chefdf['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]
condition_lo = chefdf.loc[0:,'out_AVG_CLICKS_PER_VISIT'][chefdf['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]

chefdf['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

chefdf['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_lo,
                                    value      = 1,
                                    inplace    = True)


# <h3>Feature Engineering: Trend Analysis</h3><br>
# Developing trend based features

# **Developing thresholds based on observations from scatterplots.**

# **Assumptions and actions:**
# 
# CROSS_SELL_SUCCESS is binary and used entirely. 
# AVG_CLICKS_PER_VISIT is used entirely even though there is a very thin presence of data before 8.
# PRODUCT_CATEGORIES_VIEWED is used entirely as there is a continuous trend.
# MOBILE_NUMBER this binary variable was used entirely.
# TASTES_AND_PREFERENCES is used entirely without thresholds.
# MOBILE_LOGINS was used entirely.
# PC_LOGINS was used entirely.
# EARLY_DELIVERIES was used entirely
# PACKAGE_LOCKER this binary variable was used entirely.
# REFRIGERATED_LOCKER this binary variable was used entirely.
# No thresholds were placed for FOLLOWED_RECOMMENDATIONS_PCT and MASTER_CLASSES_ATTENDED .

# In[3]:


# Setting trend-based thresholds

change_TOTAL_MEALS_ORDERED_hi            = 250 #Data scatters above this value
change_UNIQUE_MEALS_PURCH_hi             = 9   #Data scatters above this value
change_TOTAL_PHOTOS_VIEWED_hi            = 500 #Data scatters above this value
change_CONTACTS_W_CUSTOMER_SERVICE_hi    = 10  #Start of a downward trend then trend stops to flat line
change_AVG_TIME_PER_SITE_VISIT_hi        = 300 #Data scatters above this value 
change_CANCELLATIONS_BEFORE_NOON_hi      = 8   #Data scatters above this value
change_LATE_DELIVERIES_hi                = 10  #Data scatters above this value
change_AVG_PREP_VID_TIME_hi              = 290 #Data scatters above this value
change_LARGEST_ORDER_SIZE_hi             = 9   #Data scatters above this value
change_AVG_CLICKS_PER_VISIT_hi           = 10  #Starts from 8 to then and then has downward trend

# Change takes place at

change_MOBILE_NUMBER_at                  = 1 # According to graph it has more points present in higher revenue ranges for value = 1
change_TOTAL_PHOTOS_VIEWED_at            = 0 #strong concentration
change_WEEKLY_PLAN_at                    = 0 #High density around zero
change_TOTAL_PHOTOS_VIEWED_at            = 0 #heavy concentration
change_MEDIAN_MEAL_RATING_at             = 4 #discovered through categorical var analysis
change_UNIQUE_MEALS_PURCH_at             = 1 #strong concentration at 1 with some very high values for revenue
change_CANCELLATIONS_AFTER_NOON_at       = 0 #strongly zero inflated with some higher revenue values around zero


# Develop trend based features, using earlier chosen tresholds.

# In[4]:


# trend-based feature template

#change_TOTAL_MEALS_ORDERED_hi

chefdf['change_TOTAL_MEALS_ORDERED_hi'] = 0
condition = chefdf.loc[0:,'change_TOTAL_MEALS_ORDERED_hi'][chefdf['TOTAL_MEALS_ORDERED'] > change_TOTAL_MEALS_ORDERED_hi]

chefdf['change_TOTAL_MEALS_ORDERED_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#UNIQUE_MEALS_PURCH_hi
chefdf['change_UNIQUE_MEALS_PURCH_hi'] = 0
condition = chefdf.loc[0:,'change_UNIQUE_MEALS_PURCH_hi'][chefdf['UNIQUE_MEALS_PURCH'] > change_UNIQUE_MEALS_PURCH_hi]

chefdf['change_UNIQUE_MEALS_PURCH_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_TOTAL_PHOTOS_VIEWED_hi
chefdf['change_TOTAL_PHOTOS_VIEWED_hi'] = 0
condition = chefdf.loc[0:,'change_TOTAL_PHOTOS_VIEWED_hi'][chefdf['TOTAL_PHOTOS_VIEWED'] > change_TOTAL_PHOTOS_VIEWED_hi]

chefdf['change_TOTAL_PHOTOS_VIEWED_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_CONTACTS_W_CUSTOMER_SERVICE_hi
chefdf['change_CONTACTS_W_CUSTOMER_SERVICE_hi'] = 0
condition = chefdf.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE_hi'][chefdf['CONTACTS_W_CUSTOMER_SERVICE'] > change_CONTACTS_W_CUSTOMER_SERVICE_hi]

chefdf['change_CONTACTS_W_CUSTOMER_SERVICE_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)
#change_AVG_TIME_PER_SITE_VISIT_hi
chefdf['change_AVG_TIME_PER_SITE_VISIT_hi'] = 0
condition = chefdf.loc[0:,'change_AVG_TIME_PER_SITE_VISIT_hi'][chefdf['AVG_TIME_PER_SITE_VISIT'] > change_AVG_TIME_PER_SITE_VISIT_hi]

chefdf['change_AVG_TIME_PER_SITE_VISIT_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_CANCELLATIONS_BEFORE_NOON_hi
chefdf['change_CANCELLATIONS_BEFORE_NOON_hi'] = 0
condition = chefdf.loc[0:,'change_CANCELLATIONS_BEFORE_NOON_hi'][chefdf['CANCELLATIONS_BEFORE_NOON'] > change_CANCELLATIONS_BEFORE_NOON_hi]

chefdf['change_CANCELLATIONS_BEFORE_NOON_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_LATE_DELIVERIES_hi
chefdf['change_LATE_DELIVERIES_hi'] = 0
condition = chefdf.loc[0:,'change_LATE_DELIVERIES_hi'][chefdf['LATE_DELIVERIES'] > change_LATE_DELIVERIES_hi]

chefdf['change_LATE_DELIVERIES_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_AVG_PREP_VID_TIME_hi
chefdf['change_AVG_PREP_VID_TIME_hi'] = 0
condition = chefdf.loc[0:,'change_AVG_PREP_VID_TIME_hi'][chefdf['AVG_PREP_VID_TIME'] > change_AVG_PREP_VID_TIME_hi]

chefdf['change_AVG_PREP_VID_TIME_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_LARGEST_ORDER_SIZE_hi
chefdf['change_LARGEST_ORDER_SIZE_hi'] = 0
condition = chefdf.loc[0:,'change_LARGEST_ORDER_SIZE_hi'][chefdf['LARGEST_ORDER_SIZE'] > change_LARGEST_ORDER_SIZE_hi]

chefdf['change_LARGEST_ORDER_SIZE_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_AVG_CLICKS_PER_VISIT_hi
chefdf['change_AVG_CLICKS_PER_VISIT_hi'] = 0
condition = chefdf.loc[0:,'change_AVG_CLICKS_PER_VISIT_hi'][chefdf['AVG_CLICKS_PER_VISIT'] > change_AVG_CLICKS_PER_VISIT_hi]

chefdf['change_AVG_CLICKS_PER_VISIT_hi'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

########################################
## change at threshold                ##
########################################

#change_MOBILE_NUMBER_at

chefdf['change_MOBILE_NUMBER_at'] = 0
condition = chefdf.loc[0:,'change_MOBILE_NUMBER_at'][chefdf['MOBILE_NUMBER'] == change_MOBILE_NUMBER_at ]

chefdf['change_MOBILE_NUMBER_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_TOTAL_PHOTOS_VIEWED_at

chefdf['change_TOTAL_PHOTOS_VIEWED_at'] = 0
condition = chefdf.loc[0:,'change_TOTAL_PHOTOS_VIEWED_at'][chefdf['TOTAL_PHOTOS_VIEWED'] == change_TOTAL_PHOTOS_VIEWED_at ]

chefdf['change_TOTAL_PHOTOS_VIEWED_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)


#change_WEEKLY_PLAN_change_at

chefdf['change_WEEKLY_PLAN_at'] = 0
condition = chefdf.loc[0:,'change_WEEKLY_PLAN_at'][chefdf['WEEKLY_PLAN'] == change_WEEKLY_PLAN_at ]

chefdf['change_WEEKLY_PLAN_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_TOTAL_PHOTOS_VIEWED_at

chefdf['change_TOTAL_PHOTOS_VIEWED_at'] = 0
condition = chefdf.loc[0:,'change_TOTAL_PHOTOS_VIEWED_at'][chefdf['TOTAL_PHOTOS_VIEWED'] == change_TOTAL_PHOTOS_VIEWED_at ]

chefdf['change_TOTAL_PHOTOS_VIEWED_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)
#change_UNIQUE_MEALS_PURCH_at

chefdf['change_UNIQUE_MEALS_PURCH_at'] = 0
condition = chefdf.loc[0:,'change_UNIQUE_MEALS_PURCH_at'][chefdf['UNIQUE_MEALS_PURCH'] == change_UNIQUE_MEALS_PURCH_at ]

chefdf['change_UNIQUE_MEALS_PURCH_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)

#change_MEDIAN_MEAL_RATING_at

chefdf['change_MEDIAN_MEAL_RATING_at'] = 0
condition = chefdf.loc[0:,'change_MEDIAN_MEAL_RATING_at'][chefdf['MEDIAN_MEAL_RATING'] == change_MEDIAN_MEAL_RATING_at ]

chefdf['change_MEDIAN_MEAL_RATING_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)
#change_CANCELLATIONS_AFTER_NOON_at
chefdf['change_CANCELLATIONS_AFTER_NOON_at'] = 0
condition = chefdf.loc[0:,'change_CANCELLATIONS_AFTER_NOON_at'][chefdf['CANCELLATIONS_AFTER_NOON'] == change_CANCELLATIONS_AFTER_NOON_at ]

chefdf['change_CANCELLATIONS_AFTER_NOON_at'].replace(to_replace = condition,
                                       value      = 1,
                                       inplace    = True)


# <h3> Modelling </h3><br>

# The data is prepared and the train test split is performed in the code below.

# The x-variables that will be used in the model to predict the y-variable REVENUE are declared below.
# 
# **Note:**
# Following variables have been removed to avoid overfitting;
# 'AVG_TIME_PER_SITE_VISIT'
# 'TOTAL_PHOTOS_VIEWED'
# 'out_CONTACTS_W_CUSTOMER_SERVICE'
# 'change_TOTAL_PHOTOS_VIEWED_hi'

# In[5]:


x_variables = ['TOTAL_MEALS_ORDERED',
               'CONTACTS_W_CUSTOMER_SERVICE',
               'LATE_DELIVERIES',
               'AVG_PREP_VID_TIME',
               'LARGEST_ORDER_SIZE',
               'MASTER_CLASSES_ATTENDED',
               'MEDIAN_MEAL_RATING',
               'out_LATE_DELIVERIES',
               'out_AVG_PREP_VID_TIME',
               'change_TOTAL_MEALS_ORDERED_hi',
               'change_CONTACTS_W_CUSTOMER_SERVICE_hi',
               'change_AVG_PREP_VID_TIME_hi',
               'change_LARGEST_ORDER_SIZE_hi',
               'change_UNIQUE_MEALS_PURCH_at',
               'change_MEDIAN_MEAL_RATING_at']


# In the code below modeling was performed using scikit-learn

# In[6]:


# Applying modelin scikit-learn

# Preparing x-variables
chefdf_data = chefdf.loc[:,x_variables]

# Preparing response variable
chefdf_target = chefdf.loc[:,'REVENUE']

# Running train/test split again
X_train,X_test,y_train,y_test = train_test_split(
                chefdf_data, chefdf_target,test_size = 0.25, random_state = 222)


# **KNeighbors**

# In[7]:


# Instantiating a StandardScaler() object
scaler = StandardScaler()

# Fitting the scaler with chefdf
scaler.fit(chefdf_data)

# Transforming data after fit (interpreting the data as below(negative) or above(positive) average)
X_scaled = scaler.transform(chefdf_data)

# Converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

#Indicating columns for each variable
X_scaled_df.columns = chefdf_data.columns


# In[8]:


#Train Test split with standardized data

X_train, X_test, y_train,y_test = train_test_split(
            X_scaled_df,
            chefdf_target,
            test_size = 0.25,
            random_state = 222)


# In[9]:


# Creating lists for training set accuracy and test set accuracy

training_accuracy = []
test_accuracy = []

# Bilding a visualization of 1 to 21 neighbors
neighbors_settings = range(1, 21)

for n_neighbors in neighbors_settings:
    
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Finding the optimal number of neighbors
opt_neighbors = test_accuracy.index(max(test_accuracy)) + 1
print(f"""The optimal number of neighbors is {opt_neighbors}""")


# In[10]:


# Instantiate a model with the optimal number of neighbors
knn_stand = KNeighborsRegressor(algorithm = 'auto',
                   n_neighbors = opt_neighbors)

# Fitting model based on the training data
knn_stand.fit(X_train, y_train)

# Predicting
knn_stand_pred = knn_stand.predict(X_test)

# Scoring
KNN_training_score=(knn_stand.score(X_train,y_train).round(3))
KNN_test_score=(knn_stand.score(X_test, y_test).round(3))

#Printing the results
print('Training Score:', knn_stand.score(X_train,y_train).round(3))
print('Testing Score:',  knn_stand.score(X_test, y_test).round(3))

# Saving data for future use
knn_stand_score_train =  knn_stand.score(X_train,y_train).round(3)
knn_stand_score_test  = knn_stand.score(X_test, y_test).round(3)


# In[11]:


# Showing results

print(f"""
Model      Train Score      Test Score
-----      -----------      ----------
KNN        {KNN_training_score}           {KNN_test_score}
""")


# In[ ]:




