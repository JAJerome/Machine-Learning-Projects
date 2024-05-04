#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("airline_price_Dataset.csv")
# df1=df.copy()
df


# In[3]:


df.drop(columns=['Unnamed: 0','flight'], inplace=True)


# In[52]:


df


# In[53]:


df.shape


# In[54]:


df.info()


# In[55]:


df.describe()


# In[56]:


df.columns


# In[57]:


df[df.columns[0]].dtype


# In[58]:


df.dtypes


# In[59]:


df.isnull().sum()


# In[60]:


df1=df.copy


# In[61]:


df


# In[13]:


df_num=df.select_dtypes(include=['int64','float64'])
df_num


# In[14]:


df_obj=df.select_dtypes(include=['object'])
df_obj


# In[15]:


for i in df_obj.columns:
    print(df_obj[i].value_counts())


# In[16]:


for i in df_num.columns:
    print(df_num[i].value_counts())


# In[17]:


df_num_price=df_num.iloc[:,-1:]
df_num_price


# In[18]:


df_num1=df_num.iloc[:,:2]
df_num1


# In[19]:


from sklearn.preprocessing import MinMaxScaler


# In[20]:


mn=MinMaxScaler()
c=mn.fit_transform(df_num1)
df_num_mn=pd.DataFrame(c,columns=df_num1.columns)
df_num_mn.head(5)


# In[21]:


dummy=pd.get_dummies(data=df_obj,dtype=int)
dummy


# In[22]:


df_pre=pd.concat([df_num_mn,dummy,df_num_price],axis=1)
df_pre.head()


# In[23]:


x=df_pre.loc[:,df_pre.columns!='price']
y=df_pre.loc[:,['price']]


# In[24]:


x


# In[25]:


y


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=45,test_size=0.3)


# In[27]:


x_train.sort_index(ascending=True,inplace=True)


# In[28]:


x_train.head(5)


# In[29]:


x_test.sort_index(ascending=True, inplace=True)


# In[30]:


x_test.head(3)


# In[31]:


y_train.sort_index(ascending=True, inplace=True)


# In[32]:


y_train


# In[33]:


y_test.sort_index(ascending=True, inplace=True)


# In[34]:


y_test


# In[35]:


aa=df_num_mn


# In[36]:


scaler = MinMaxScaler()
df_scaled =pd.DataFrame(scaler.fit_transform(aa),columns=aa.columns)
print("Min-Max Scaled Data:\n", df_scaled)


# In[37]:


df_scaled


# In[38]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


# In[39]:


scaler = StandardScaler()
scaled_data_standardized = scaler.fit_transform(df_num_mn)
print("Standardized Data:\n", scaled_data_standardized)


# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


x_train


# In[42]:


y_train


# In[43]:


import statsmodels.api as sm
MLR_model1=sm.OLS(y_train,x_train).fit()
print(MLR_model1.summary())


# In[44]:


y_test_pred = MLR_model1.predict(x_test)
y_test_pred.count()


# In[45]:


y_test_pred


# In[46]:


from sklearn.metrics import mean_absolute_error,r2_score


# In[47]:


mean_absolute_error(y_true=y_test,y_pred=y_test_pred)


# In[48]:


from sklearn.metrics import r2_score
r2_score(y_true=y_test.price, y_pred=y_test_pred)


# In[66]:


len(x_test.columns)


# In[82]:


y_test


# In[77]:


x_test.iloc[:,28:40]


# In[81]:


input_data = (0.027,0.0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1)
len(input_data)
# Change the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# Reshape the array for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = MLR_model1.predict(input_data_reshaped)
print(prediction)


# In[79]:


len(input_data)


# In[85]:


y_train.price


# In[91]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=250,criterion='squared_error',min_samples_split=2,min_samples_leaf=1,random_state=45)
rforest=rf.fit(x_train,y_train.price)


# In[ ]:


input_data = (0.027,0.0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1)
len(input_data)
# Change the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# Reshape the array for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
rforest=rf.fit(x_train,y_train.price)
print(prediction)


# In[ ]:





# In[ ]:




