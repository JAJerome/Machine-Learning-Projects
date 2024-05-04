#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("fraudTest.csv")
df1=df.copy()
df


# In[3]:


df.drop(columns=['Unnamed: 0','trans_num','street','zip','unix_time'],inplace=True)


# In[4]:


df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'],utc=True)


# In[5]:


df['dob'] = pd.to_datetime(df['dob'],utc=True)


# In[6]:


#df['age']=df['trans_date_trans_time']-df['dob']


# In[7]:


df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# Extract month and year and set it as 'trans_date'
df['trans_date'] = df['trans_date_trans_time'].dt.year


# In[8]:


df['dob'] = pd.to_datetime(df['dob'])

# Extract month and year and set it as 'trans_date'
df['Dob'] = df['dob'].dt.year


# In[9]:


df['age']=  df['trans_date']-df['Dob']


# In[10]:


df['age']


# In[11]:


df


# In[12]:


df


# In[13]:


df.dtypes


# In[14]:


df.drop(columns=['trans_date','Dob'],inplace=True)


# In[15]:


df


# In[16]:


df.info()


# In[17]:


df.columns


# In[18]:


df.shape


# In[19]:


df.describe()


# In[20]:


df[df.columns[1]].dtype


# In[21]:


df.dtypes


# In[22]:


df.isnull().sum()


# In[23]:


df


# In[99]:


df_num=df.select_dtypes(include=['int64','float64','int32'])
df_num


# In[101]:


df_num.drop(columns=['cc_num','merch_lat','merch_long'],inplace=True)


# In[102]:


df_num


# In[28]:


#amt,age,transdate,(merch lat,long,)


# In[103]:


df_obj=df.select_dtypes(include=['object'])
df_obj


# In[104]:


df_obj.drop(columns=['merchant','first','last','city','state','job'],inplace=True)


# In[105]:


df_obj


# In[106]:


for i in df_num.columns:
    print(df_num[i].value_counts())


# In[107]:


for i in df_obj.columns:
    print(df_obj[i].value_counts())


# In[123]:


df_num


# In[111]:


#df_num_fraud=df_num['is_fraud']
#df_num_fraud


# In[112]:


#df_num1=df_num.iloc[:,:1]
#df_num1


# In[124]:


df_num


# In[126]:


df_num['fraud']=df_num['is_fraud']
df_num['fraud']


# In[121]:


df_num.drop(columns=['is_fraud'],inplace=True)


# In[127]:


df_num


# In[135]:


df_num1=df_num.iloc[:,:5]
df_num1


# In[136]:


#feature engineering cat columns to numericals
from sklearn.preprocessing import MinMaxScaler


# In[137]:


mn=MinMaxScaler()
c=mn.fit_transform(df_num1)
df_num_mn=pd.DataFrame(c,columns=df_num1.columns)
df_num_mn


# In[138]:


df.dtypes


# In[139]:


from sklearn.preprocessing import StandardScaler


# In[140]:


scaler = MinMaxScaler()
df_scaled =pd.DataFrame(scaler.fit_transform(df_num1),columns=df_num1.columns)
print("Min-Max Scaled Data:\n", df_scaled)


# In[141]:


df_obj


# In[42]:


#onehot encoding the gender & categorical  column


# In[142]:


columns_to_encode=['gender','category']
df_obj_encoded=pd.get_dummies(df_obj,columns=columns_to_encode,dtype=int)
df_obj_encoded


# In[44]:


#label encoding df_obj expect the gender column


# In[45]:


#from sklearn.preprocessing import LabelEncoder


# In[46]:


#exclude_columns=['gender_F','gender_M']


# In[47]:


#label_encoder=LabelEncoder()


# In[48]:


#for column in df_obj.columns:
    #if column != exclude_columns and df_obj[column].dtype == 'object':
     #   df_obj[column] = label_encoder.fit_transform(df_obj[column])


# In[144]:


df_obj


# In[145]:


df


# In[146]:


#df_trans_date=df['trans_date_trans_time']
#df_trans_date


# In[147]:


df_date = df['trans_date_trans_time'].dt.date
df_date


# In[148]:


df_age=df['age']
df_age


# In[149]:


df_num


# In[55]:


#concardinating the fearture engg data


# In[91]:


df_date=pd.to_datetime(df_date,utc=True)


# In[151]:


df_num_mn


# In[153]:


df_num


# In[176]:


df_pre=pd.concat([df_num_mn,df_obj_encoded],axis=1)
df_pre.head()


# In[177]:


x=df_pre


# In[178]:


#x=df_pre.loc[:,df_pre.columns!='is_fraud']
y=df_num.loc[:,['fraud']]


# In[179]:


x,y


# In[180]:


x.dtypes


# In[181]:


y


# In[182]:


#splitting the dataset into train & test


# In[183]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=45,test_size=0.3)


# In[184]:


x_train.sort_index(ascending=True,inplace=True)


# In[185]:


x_train.head(20)


# In[186]:


x_test.sort_index(ascending=True, inplace=True)


# In[187]:


x_test.head(5)


# In[188]:


y_train.sort_index(ascending=True, inplace=True)


# In[189]:


y_train


# In[190]:


y_test.sort_index(ascending=True, inplace=True)


# In[191]:


y_test


# In[192]:


#prediction
from sklearn.linear_model import LogisticRegression


# In[193]:


lr=LogisticRegression()


# In[194]:


logreg=lr.fit(x_train,y_train)


# In[195]:


logreg


# In[200]:


y_predprob=logreg.predict_proba(X=x_test)
y_predprob
#prob value s of label 0     prob value of lavbel 1


# In[204]:


y_predprob[:,1]


# In[205]:


#yy=[return 0 else if i>0.5 return 1 for i in y_predprob[:,1]]


# In[201]:


y_pred = logreg.predict(x_test)
len(y_pred)


# In[202]:


y_pred[0:10]


# In[207]:


from sklearn.metrics import confusion_matrix


# In[208]:


confusion_matrix(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




