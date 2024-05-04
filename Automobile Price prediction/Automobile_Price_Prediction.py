#!/usr/bin/env python
# coding: utf-8

# ``` ```
# # Automobile Price Prediction
# 
# In this Jupyter Notebook, We're going to learn how to use machine learning to predict automobile prices.
# 
# ### 1. Data Loading and Preprocessing:
#    - Import necessary libraries.
#    - Load the automobile dataset.
#    - Drop irrelevant columns like 'symboling', 'normalized-losses', and 'aspiration'.
#    - Handle missing values by replacing '?' with appropriate values and converting data types.
# 
# ``` ``` ``` ```

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Load the dataset
filename = 'Automobile_data.csv'
df = pd.read_csv(filename)
df


# In[3]:


# Drop certain columns
columns_to_drop = ['symboling', 'normalized-losses', 'aspiration']
df = df.drop(columns=columns_to_drop)
df


# ``` ```
# 
# ## 2. Data Exploration:
# 
#    - Examine basic information about the dataset.
#     
#     
# ```python
# 
# # Handle missing values and data type conversion
# # ...
# 

# In[4]:


# Display information about the DataFrame
df.info()


# In[5]:


df.iloc[:,10:23].head(3)
#df.iloc[row index,col index]


# In[6]:


df.head(2)


# In[7]:


df['price'].value_counts()


#The columns mentioned below have wrong dtypes.
#these columns have ? symbol as a value.we replace the ? with 0.The we change the dtype.


# In[8]:


# df['bore'] = df['bore'].replace('?', 0)
# df['stroke'] = df['stroke'].replace('?', 0)
# df['horsepower'] = df['horsepower'].replace('?', 0)


# In[9]:


# Convert the below columns to the appropriate data types
#convert them to int,float dtypes using astype fun.
df['bore'] = df['bore'].replace('?', 0).astype('float64')
df['stroke'] = df['stroke'].replace('?', 0).astype('float64')
df['horsepower'] = df['horsepower'].replace('?', 0).astype('int64')
df['peak-rpm'] = df['peak-rpm'].replace('?', 0).astype('int64')
df['price'] = df['price'].replace('?', 0).astype('int64')
df.info()


# In[10]:


# Check for missing values
df.isnull().sum()


# In[11]:


# Create a copy of the DataFrame
df1 = df.copy()
df1.head()


# In[12]:


df1.head(3)


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[14]:


#Univariate analysis


# In[15]:


# plt.ylabel("Count of cars manufactured by each company")


# In[29]:


#I plot a figure with Width of 20 inches and height of 6 inches.
sns.countplot(x='make',data=df1)
plt.figure(figsize=(20,30
                   ))
plt.show()


# In[17]:


#count of cars manufactured by each company is displayed.
#Most number of cars are produced by toyota.The count is above 30.


# In[18]:


sns.countplot(x='fuel-type',data=df1)
plt.figure(figsize=(20, 6))
plt.show()


# In[19]:


#Most of the cars run on gas.There are few cars that consume diesel.


# In[20]:


#Bivariate analysis


# In[21]:


sns.scatterplot(x='horsepower',y='price',data=df1);


# In[22]:


#When the horsepower ranges between 100 to 150,the cars price lies between 10k to 20 k us dollars.
#for high horsepower values,price of the cars lies beyond 30k us dollars.(units..)


# In[ ]:





# In[23]:


sns.scatterplot(x='horsepower',y='peak-rpm',data=df1)


# In[28]:


df1.corr()


# ``` ```
# 
# ## 3. Feature Engineering:
#    - Standardize numeric features.
#    - Encode categorical features using one-hot encoding.
#     
# ```
# ```

# In[ ]:


# Select numeric columns
num_df1 = df1.select_dtypes(include=['int64', 'float64'])#int64,float64
num_df1.head()


# In[ ]:


# Standardize numeric features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1_num_sc = pd.DataFrame(sc.fit_transform(num_df1), columns=num_df1.columns)
df1_num_sc


# In[ ]:


# Select categorical columns
str_df1 = df1.select_dtypes(include=['object'])
str_df1


# In[ ]:


# Convert categorical columns into dummy variables
dummy = pd.get_dummies(str_df1, drop_first=False)
dummy.head(3)


# In[ ]:


# Combine standardized numeric data and dummy variables
df1_pre = pd.concat([dummy,df1_num_sc], axis=1)
df1_pre.head(5)


# In[ ]:


df1_pre.iloc[:,0:-1]


# ``` ```
# 
# ## 4. Data Splitting:
#    - Split the dataset into training and testing sets.
# ```
# ```
# 
# <!-- X = df1_pre.drop('price', axis=1)
#   -->

# In[ ]:


# Define X (features) and Y (target)
x = df1_pre.iloc[:,0:-1]
x


# In[ ]:


#Y = df1.iloc[:, [-1]]
y=df1_pre.iloc[:,[-1]]
y


# In[ ]:


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[ ]:


# Sort the index of training and testing sets
x_train.sort_index(ascending=True, inplace=True)
x_test.sort_index(ascending=True, inplace=True)
y_train.sort_index(ascending=True, inplace=True)
y_test.sort_index(ascending=True, inplace=True)


# ``` ```
# 
# ## 5. Model Building:
#    - Train a Multiple Linear Regression (MLR) model using the training data.
# 
# ```
# ```

# In[ ]:


# Build a Multiple Linear Regression (MLR) model
import statsmodels.api as sm
MLR_model1 = sm.OLS(y_train, x_train).fit()
MLR_model1.summary()


# In[ ]:


# Make predictions on the test set
y_test_pred = MLR_model1.predict(x_test)
y_test_pred.count()


# In[ ]:


# Sort the index of predicted values
#y_test_pred.sort_index(ascending=True, inplace=True)


# In[ ]:


y_test_pred


# ``` ```
# 
# ## 6. Model Evaluation:
#    - Make predictions on the test set.
#    - Calculate metrics such as Mean Squared Error (MSE) and R-squared (R2) score
#    
# ```
# ```

# In[ ]:


# Calculate the error between actual and predicted values
error = y_test.price - y_test_pred
error


# In[ ]:


# Calculate the mean squared error (MSE)
(y_test.price - y_test_pred) ** 2


# In[ ]:


# Calculate the mean squared error (MSE)
import numpy as np
np.sum((y_test.price - y_test_pred) ** 2) / len(y_test.price)



# In[ ]:


MSE is close to 0.


# In[ ]:


# Calculate the R-squared (R2) score
from sklearn.metrics import r2_score
r2_score(y_true=y_test.price, y_pred=y_test_pred)


# In[ ]:


R_square value is 0.78.MSE is close to 0.
The linear regression model made accurate predictions.It learned
well from the train data.


# ``` ```
# 
# ## 7. Visualization:
#    - Plot a scatter plot to visualize actual vs. predicted prices.
# 
# ```
# ```

# In[ ]:


# Plot a scatter plot of actual vs. predicted prices
import matplotlib.pyplot as plt
plt.scatter(y_test.price, y_test_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# 
# This markdown document outlines the step-by-step process of performing automobile price prediction using a Jupyter Notebook with explanations for each code segment.
# 

# ```
# ```
