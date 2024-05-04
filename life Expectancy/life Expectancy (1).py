#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[126]:


df=pd.read_csv("Life_Expectancy.csv")
df1=df.copy()
df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[127]:


df.describe()


# In[128]:


df.info()


# In[129]:


df.columns


# In[130]:


df.dtypes


# In[131]:


df.isnull().sum()


# In[132]:


#adding missing values


# In[133]:


df['Alcohol'].fillna(df['Alcohol'].mean(), inplace=True)
df['Hepatitis B'].fillna(df['Hepatitis B'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df['GDP'].fillna(df['GDP'].mean(), inplace=True)
df['Malnourished10_19'].fillna(df['Malnourished10_19'].mean(), inplace=True)
df['Malnourished5_9'].fillna(df['Malnourished5_9'].mean(), inplace=True)
df['Income_Index'].fillna(df['Income_Index'].mean(), inplace=True)
df['Schooling'].fillna(df['Schooling'].mean(), inplace=True)
df['Life_Expectancy'].fillna(df['Life_Expectancy'].mean(), inplace=True)
df['Adult_Mortality'].fillna(df['Adult_Mortality'].mean(), inplace=True)
df['Population'].fillna(df['Population'].mean(), inplace=True)


# In[134]:


df.isnull().sum()


# In[135]:


df


# In[136]:


#DATA VISUALIZATION


# In[137]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[138]:


#histplot
numerical_columns = ['Adult_Mortality', 'Infant_Deaths', 'Alcohol', 'BMI', 'Polio', 'HIV', 'GDP', 'Population', 'Malnourished10_19', 'Malnourished5_9', 'Income_Index', 'Schooling', 'Life_Expectancy']
plt.figure(figsize=(12, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[column], bins=20, kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()


# In[139]:


#barplot


# In[140]:


for column in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()


# In[141]:


sns.countplot(data=df,x="Status",)


# In[142]:


sns.scatterplot(data=df,x='Income_Index',y='GDP')


# In[143]:


sns.scatterplot(data=df,y='Infant_Deaths',x='GDP')


# In[144]:


sns.scatterplot(data=df,y='BMI',x='Life_Expectancy')


# In[145]:


plt.hist(df, bins=10)  
plt.title('Histogram of Data')
plt.xlabel('Count')
plt.ylabel('Infant_Deaths')
plt.show()


# In[146]:


df.columns


# In[147]:


df.drop(columns=['Year','Country','Infant_Deaths'],inplace=True)


# In[148]:


df.drop(columns=['Underfive_Deaths '],inplace=True)


# In[149]:


df.drop(columns=['Measles '],inplace=True)


# In[150]:


df


# In[152]:


df.corr()


# In[ ]:





# In[153]:


df


# In[154]:


df_num=df.select_dtypes(include=['int64','float64','int32'])
df_num


# In[155]:


df_num1=df_num.iloc[:,-1:]
df_num1


# In[156]:


df_num2=df_num.copy()


# In[157]:


df_num2.drop(columns=['Population','GDP','Life_Expectancy'],inplace=True)


# In[158]:


df_num2


# In[159]:


for i in df_num2.columns:
    print(df_num2[i].value_counts())


# In[160]:


from sklearn.preprocessing import MinMaxScaler


# In[161]:


mn=MinMaxScaler()
c=mn.fit_transform(df_num2)
df_num_mn=pd.DataFrame(c,columns=df_num2.columns)
df_num_mn


# In[162]:


from sklearn.preprocessing import StandardScaler


# In[163]:


scaler = MinMaxScaler()
df_scaled =pd.DataFrame(scaler.fit_transform(df_num2),columns=df_num2.columns)
print("Min-Max Scaled Data:\n", df_scaled)


# In[164]:


df_GDP=df['GDP']


# In[165]:


df_population=df['Population']


# In[166]:


df_obj=df.select_dtypes(include=['object'])
df_obj


# In[167]:


for i in df_obj.columns:
    print(df_obj[i].value_counts())


# In[170]:


#df_obj.drop(columns=['Country'],inplace=True)


# In[171]:


#onehot encoding 


# In[172]:


df_obj


# In[173]:


columns_to_encode=['Status']
df_obj_encoded=pd.get_dummies(df_obj,columns=columns_to_encode,dtype=int)
df_obj_encoded


# In[175]:


# from sklearn.preprocessing import LabelEncoder
# df_country=df['Country']
# # Assuming 'Country' is the column containing country names
# label_encoder = LabelEncoder()
# df_obj_encoded1=label_encoder.fit_transform(df['Country'])


# In[176]:


#target encoding
# from category_encoders import TargetEncoder

# # Assuming 'Country' is the column containing country names and 'Life_Expectancy' is the target variable
# target_encoder = TargetEncoder()
# df['Country_Encoded'] = target_encoder.fit_transform(df['Country'], df['Life_Expectancy'])


# In[177]:


df_pre=pd.concat([df_num_mn,df_GDP,df_population,df_obj_encoded],axis=1)
df_pre.head()


# In[178]:


x=df_pre.astype(float)


# In[179]:


y=df_num1.astype(float)


# In[180]:


x,y


# In[181]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=45,test_size=0.3)


# In[182]:


x_train.sort_index(ascending=True,inplace=True)


# In[183]:


x_train.head(5)


# In[184]:


x_test.sort_index(ascending=True, inplace=True)


# In[185]:


x_test.head(5)


# In[186]:


y_train.sort_index(ascending=True, inplace=True)


# In[187]:


y_train


# In[188]:


y_test.sort_index(ascending=True, inplace=True)


# In[189]:


y_test.head(5)


# In[190]:


#prediction


# In[191]:


from sklearn.linear_model import LogisticRegression


# In[192]:


lr=LogisticRegression()


# In[194]:


logreg=lr.fit(x_train,y_train)


# In[195]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


# In[196]:


model = LinearRegression()


# In[197]:


model.fit(x_train, y_train)


# In[198]:


y_pred = model.predict(x_test)


# In[199]:


y_pred


# In[200]:


#evaluating the model


# In[201]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[202]:


from sklearn.metrics import mean_squared_error


# In[203]:


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[204]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[205]:


rfc = RandomForestClassifier(n_estimators=100, random_state=42)


# In[206]:


rfc.fit(x_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:




