#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Objective:
#To predict the House rent amount based on various attributes using
#Multiple ML algorithms and compare the evaluation metrics of ml models.


# In[2]:


#Attribute information


# In[3]:


# BHK: Number of Bedrooms, Hall, Kitchen.

# Rent: Price of the Houses/Apartments/Flats.

# Size: Size of the Houses/Apartments/Flats in Square Feet.

# Floor: Houses/Apartments/Flats situated in which Floor and Total Number of Floors (Example: Ground out of 2, 3 out of 5, etc.)

# Area Type: Size of the Houses/Apartments/Flats calculated on either Super Area or Carpet Area or Build Area.

# Area Locality: Locality of the Houses/Apartments/Flats.

# City: City where the Houses/Apartments/Flats are Located.

# Furnishing Status: Furnishing Status of the Houses/Apartments/Flats, either it is Furnished or Semi-Furnished or Unfurnished.

# Tenant Preferred: Type of Tenant Preferred by the Owner or Agent.

# Bathroom: Number of Bathrooms.

# Point of Contact: Whom should you contact for more information regarding the Houses/Apartments/Flats


# In[4]:


import pandas as pd
df=pd.read_csv("House_Rent_Dataset.csv")
# df1=df.copy()
df


# In[5]:


df.shape


# In[6]:


df.info()

#I perform groupby on month column and display the average rent paid by each tenant in every month.
#I can do digital marketing.I can provide offers like 10% reduction in rent to tenants.
#A particular month will have high average rent.say december avg rent paid by each tentant is 20k
#We can provide 10% reduction in rent for each tentant.10% discount.Then many people will come to my 
#apartment and live in my apaertment houses.My business will increase.


# In[7]:


df.describe()


# In[8]:


########################################
#bhk increases means rent increases    #
#size inc means rent inc               #
#Area type inc means rent inc          #
#based on city price varies            #
#based on furnishing price varies      #
#based on bathroom price varies        #
#tnenant inc price increases           #
########################################


# In[ ]:





# In[9]:


#No null values in the dataset.No need to perform null value treatment.


# In[10]:


df.columns


# In[ ]:





# In[11]:


#convert date colum datatype from object to datetime
#create a new col named as month.retrieve month from date column.
#Month col is used to analyse.


# In[ ]:





# In[12]:


# df['Posted On']=pd.to_datetime(df['Posted On'],utc=True)


# In[13]:


df[df.columns[0]].dtype


# In[14]:


# df['Posted On'].str.slice(0,4)
#AttributeError: Can only use .str accessor with string values!
#It measn that u can use slice fun to slice substring from a string d type column.object
#We cannot slice substring from a datetime column.


# In[15]:


# df['Year_of_ad_post']=df['Posted On'].str.slice(0,4)


# In[16]:


Date column gives u details about the date of job ad post.
Date column does not mean date of payment of the rent.
SO date is irrelevant column.I cannot predict rent amount based on date of ad posted by houseowner.


# In[ ]:


#df.col1.str.slice() 
##df.col1.str.replace() 


# In[17]:


df.drop(columns=['Posted On'], inplace=True)
#do not drop it


# In[18]:


df.dtypes


# In[19]:


df.isnull().sum()


# In[20]:


#removed ir relevent features
#no need to change d types ..astype fun all cols have correct dtypes
#no null valuess
#create a copy of the raw clean dataset to perform data visualization.


# In[21]:


df1=df.copy()


# In[22]:


#df1 will be used for data visualization


# In[23]:


#Select num vars 


# In[24]:


df_num=df.select_dtypes(include=['float64', 'int64'])
# df1_num
# df1_num1=df1_num.iloc[:,1:-1]
# df1_num1.head(5)
df_num.head(4)


# In[25]:


df_cat=df.select_dtypes(include=['object'])
df_cat


# In[26]:


#display the value counts in each cate vars.Why?
#to replace different values using logic
for i in df_cat.columns:
    print(df_cat[i].value_counts())


# In[27]:


#There are no different values like '-1' and '?'.All fine.


# In[28]:


#to display -1 or different values in each num var we have to type the following code for each num var
# df_num.iloc[df_num.numvar==-1]
#-1 or any other value like '?'


# In[29]:


df_cat.dtypes
#Floor means the floor number at which the house is available for rent
#The houses are available for rent in various areas in various cities across india.
#Many house owners prefer bachelor or family.Any thing is ok for them.There are 472 house owners who specificly 
#want family tenants


# In[30]:


#Floor has many cate values .So i wwill label encode them.
#Label encode the foloowing columns:1)Floor2)area locality
#one hot encode the folowing columns:1)City2)furnishing status3)Tenant4)Point of contact5)Area type


# In[31]:


#Feature Engineering on cate vars


# In[32]:


#some columns has 4 cate values while some columns have 10+cate values.
#perform one hot encoding of these columns
#and perform label encoding on columns with 10+cate values.


# In[ ]:





# In[33]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[34]:


df_cat.Floor= le.fit_transform(df_cat.Floor)
#We can label encode a column in the same column only.
#We cannot put integer values in new column.


# In[35]:


df_cat


# In[36]:


# df1_cat['Area Type']= le.fit_transform(df1_cat['Area Type'])


# In[37]:


df_cat['Area Locality']= le.fit_transform(df_cat['Area Locality'])


# In[38]:


df_cat


# In[39]:


df_cat.columns


# In[ ]:





# In[40]:


a=df_cat.iloc[:,[1,3,4,5,6]]
a


# In[41]:


dummy=pd.get_dummies(data=a,dtype=int)
dummy


# In[42]:


dummy1=pd.concat([dummy,df_cat],axis=1)
dummy1


# In[43]:


dummy1.drop(columns=['Area Type','City','Furnishing Status','Tenant Preferred','Point of Contact'],inplace=True)


# In[44]:


dummy1.head(10)


# In[45]:


#Feature Engineering Numerical columns
from sklearn.preprocessing import MinMaxScaler


# In[ ]:





# In[46]:


#Applying min max scaling on num vars
mn=MinMaxScaler()
c=mn.fit_transform(df_num)
df_num_mn=pd.DataFrame(c,columns=df_num.columns)
df_num_mn.head(5)
#TV named Rent must be scaled when u scaled inp num vars.


# In[47]:


df_pre=pd.concat([df_num_mn,dummy1],axis=1)
df_pre.head()


# In[48]:


#df_pre has scaled num vars + dummy vars


# In[49]:


#Train test split
x=df_pre.loc[:,df_pre.columns!='Rent']


# In[50]:


y=df_pre.loc[:,['Rent']]


# In[51]:


# y.unique()


# In[52]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=45,test_size=0.3)


# In[53]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[54]:


x_train.sort_index(ascending=True,inplace=True)


# In[55]:


x_train.head(4)


# In[56]:


x_test.sort_index(ascending=True, inplace=True)


# In[57]:


x_test.head(3)


# In[58]:


y_train.sort_index(ascending=True, inplace=True)


# In[59]:


y_train
#when y is a df with one col,ytrai must be a df with one col
#ytest must be a df with on col which is tv


# In[60]:


y_test.sort_index(ascending=True, inplace=True)


# In[61]:


y_test


# In[62]:


#from sklearn import linear regression


# In[63]:


import statsmodels.api as sm
MLR_model1=sm.OLS(y_train,x_train).fit()
print(MLR_model1.summary())


# In[ ]:





# In[64]:


#calculate mse and r-square

y_test_pred = MLR_model1.predict(x_test)
y_test_pred.count()


# In[65]:


y_test_pred


# In[66]:


y_test_pred[33]


# In[67]:


y_test.Rent[33]


# In[68]:


#actual op value of a dp at row index 33 is 0.001
#predicted op value of a dp at row index 33 is 0.0008
#error is less.Ml model has made almost accurate prediction for this dp.


# In[69]:


#MSE manually


s1=(y_test_pred-y_test['Rent'])**2
type(s1)
s1


# In[70]:


import numpy as np


# In[71]:


np.sum(s1)/len(y_test)


# In[72]:


from sklearn.metrics import mean_squared_error


# In[73]:


mean_squared_error(y_true=y_test.Rent,y_pred=y_test_pred)


# In[74]:


#MSE is close to 0.
#THe OLS model which is Linear Regression model is a good model.


# In[75]:


y_test.Rent


# In[ ]:





# In[76]:


# for r square
from sklearn.metrics import r2_score
r2_score(y_true=y_test.Rent, y_pred=y_test_pred)
#in the y_true u can pass a df with one column or a series data.
#YOu can pass either of the two data.


# In[77]:


#to calc mae


# In[78]:


from sklearn.metrics import mean_absolute_error,r2_score


# In[79]:


mean_absolute_error(y_true=y_test,y_pred=y_test_pred)


# In[80]:


s2=np.abs(y_test.Rent-y_test_pred)
s2


# In[81]:


np.sum(s2)/len(y_test)


# In[82]:


len(y_test)


# In[83]:


#MAE is named as mean absolute error.error is difference between actuual op value and predicted op value 
#for all the datapoints.
#Mae is mean of absolute difference between act op value and predicted op value.


# In[84]:


#MAe is close to 0.good model.


# In[85]:


r2_score(y_true=y_test.Rent,y_pred=y_test_pred)



# In[86]:


#Data visualization
import seaborn as sns


# In[87]:


df1.tail()


# In[88]:


#BI VARIATE ANALYSIS


# In[89]:


import matplotlib.pyplot as plt


# In[90]:


df1.groupby(by=['City'])['Rent'].max()


# In[91]:


df_cr=pd.DataFrame(df1.groupby(by=['City'])['Rent'].mean())
df_cr


# In[92]:


df_cr['City']=df_cr.index


# In[93]:


df_cr


# In[94]:


plt.bar(x='City',height='Rent',data=df_cr)
# plt.figsize()


# In[95]:


The average rent at a house in Mumbai is 85k.It is higher than avg rent in other cities.
So Cost of living is high in mumbai compared  to other cities.


# In[96]:


#scatter plot 1
sns.scatterplot(data=df1,x=df1.BHK,y=df1.Rent)


# In[97]:


df1.loc[df1.BHK==3]['Rent'].max()
#Select the records where bhk equal to 3.rent col is selected.
#max rent for 3 bhk house is 35 lakh.


# In[98]:


#For a 3 BHK house,the maximum rent is 35,00,000.It is displayed in graph too.
#4 BHK houses have high rents compared to other bhk houses.


# In[99]:


#scatter plot 2


# In[100]:


sns.scatterplot(data=df1,x=df1.Size,y=df1.Rent)


# When the size of the house is high ,rent is high.
# big house have high rent.small house have less rent.
# 
# 

# In[101]:


df1.loc[(df1.Size>=3000) & (df1.Size<=4000)]['Rent'].mean()


# In[102]:


#Mean rent amount of houses whose sizes range btw 3000 to 4000 units is 2.06 lakh rupees.(units)
#70k inc


# In[103]:


df1.loc[(df1.Size>=2000) & (df1.Size<=3000)]['Rent'].mean()


# In[104]:


#Mean rent amount of houses whose sizes range btw 2000 to 3000 units is 1.28 lakh rupees.(units)


# In[105]:


#df.groupby
#df.loc[where cond 1,2]['num var'].agg function.


# In[106]:


df1.corr()


# In[ ]:


#As the size of the house increases,rent increases.
#A tenant needs to pay high rent for a big house.


# In[ ]:





# In[112]:


import sklearn
from sklearn.tree import DecisionTreeRegressor


# In[113]:


dt=DecisionTreeRegressor(criterion='squared_error',min_samples_split=2,min_samples_leaf=1,random_state=45)


# In[114]:


dtree=dt.fit(x_train,y_train)


# In[115]:


y_pred2=dtree.predict(x_test)
y_pred2


# In[116]:


###mse,mae,r squared error value
#mse,mae,r squared error value
#calculate the mse,mae,r squared value of test data.


# In[117]:


y_test


# In[118]:


#mse on test data when we build decision tree regressor model
mean_squared_error(y_true=y_test.Rent,y_pred=y_pred2)


# In[119]:


mean_absolute_error(y_true=y_test.Rent,y_pred=y_pred2)


# In[120]:


r2_score(y_true=y_test.Rent,y_pred=y_pred2)


# In[121]:


from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=250,criterion='squared_error',min_samples_split=2,min_samples_leaf=1,random_state=45)


rforest=rf.fit(x_train,y_train)


# In[124]:


y_pred3=rforest.predict(x_test)
#predict function reads the xtest data and predicts the op values of the test data.
#Predict function uses trained random forest model object named as rforest object.


# In[125]:


y_pred3


# In[126]:


#mse on test data when we build Random forest regressor model
mean_squared_error(y_true=y_test.Rent,y_pred=y_pred3)


# In[127]:


mean_absolute_error(y_true=y_test.Rent,y_pred=y_pred3)


# In[128]:


r2_score(y_true=y_test.Rent,y_pred=y_pred3)


# In[129]:


x_test


# In[130]:


x_test.iloc[:,8:]


# In[131]:


y_test


# In[132]:


input_data = (0.2,0.136421,0.11,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,455,221)
# Change the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# Reshape the array for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = MLR_model1.predict(input_data_reshaped)
print(prediction)


# In[133]:


# #if (prediction[0] == 0):
#   print('The person will not get H.A')
# else:
#   print('The person will get H.A')


# In[134]:


Conclusion:
    
Models:                     MSE        MAE       R_squared
1)Linear Regression         0.00012    0.0064    0.546
2)DecisionTreeRegressor     0.00078    0.0047
3)RandomForestRegressor     0.00016    0.0037    0.38


# In[135]:


Linear Regression model(OLS) is the best model because it has MSE close to 0,
MAE close to 0,RMSE close to 0 and High R_Sqaured value.
IT has made accurate predictions.

