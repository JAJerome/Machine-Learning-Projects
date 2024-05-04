#!/usr/bin/env python
# coding: utf-8

# This Dataset consists list of AI Jobs in india .
# 
# Job titles which are interchangeably used ('Data Scientist', 'Data Engineer' ,'Machine Learning Engineer','Data Analyst')
# 
# Number of rows in data - 885
# 
# Number of columns/attribute in data - 19
# 
# 

# 
# 

# Objective:
# Finding Companies most probable to hire an ML Engineer/Data Analyst Applicant in respect to his/her skillset.
# 
# To analyse Machine Learning Job Market in India with respect to the given problem statement using Segmentation analysis and outline the segments most optimal to apply or prepare for Machine Learning Jobs.

# In[1]:


# Techniques and Algorithms used - Machine learning using python with numpy , pandas and Basic maths.
# In this project we took Dataset consists list of AI Jobs in india and
# analyzed the dataset using various data analysis techniques and using various python libraries
# and Various Plotting techniques.


# In[2]:


# Attribute Information:

# 'Job.Title' : Job title

# 'Job.Description': Job description

# 'Rating': Job rating

# 'Company.Name': Company name

# 'Location': Location of the company

# 'Headquarters': Company headquarters

# 'Size': the Size of the company

# 'Founded': The year it was founded

# 'Type.of.ownership': Public or private ownership

# 'Industry': Type of industry company belongs

# 'Sector': Type of sector company belongs

# 'Revenue': Yearly revenue by company

# 'Competitors': Company competitors

# 'Python': Does description has python keyword in it 1 if yes 0 if no

# 'R.Prog': Does description has R prog keyword in it 1 if yes 0 if no

# 'Excel': Does description has Excel keyword in it 1 if yes 0 if no

# 'Hadoop': Does description has Hadoop keyword in it 1 if yes 0 if no

# 'SQL': Does description has SQL keyword in it 1 if yes 0 if no

# 'SAS: Does description has SAS keyword in it 1 if yes 0 if no


# In[ ]:





# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# from google.colab import drive
# drive.mount("Mydrive")


# In[5]:


#upload dataset in googledrive
#Connect google drive with google colab 
#Read the datset in googlecolab


# In[6]:


df1=pd.read_csv(r'cleaned_job_dataset.csv',encoding='ISO-8859-1')
df1.head()


# In[7]:


df1.shape


# In[8]:


#change date column to datetime format
#change Programming Leanguages columns 6 to object dtype


# In[ ]:





# In[9]:


df1['Founded']=pd.to_datetime(df1["Founded"],utc=True)


# In[10]:


#Founded column is converted to datetime.
#When I set utc=True i convert all the datetime values in same time zone i.e GMT greenwich
#i.e GMT 0:00


# In[11]:


df1['Python']=df1['Python'].astype(object)


# In[ ]:





# In[12]:


df1['Rating'].dtype


# In[13]:


df1['Rating']=df1['Rating'].astype('float64')
#Rating col has float values.ITs d type must be float.


# In[14]:


df1['R Prog']=df1['R Prog'].astype(object)


# In[15]:


df1['Excel']=df1['Excel'].astype(object)


# In[16]:


df1['Excel'].dtype


# In[17]:


df1['SQL']=df1['SQL'].astype(object)


# In[18]:


df1['SAS']=df1['SAS'].astype(object)


# In[19]:


df1['Hadoop']=df1['Hadoop'].astype(object)


# In[20]:


df1.dtypes


# In[21]:


#All the columns have correct data types.


# In[22]:


df1.isnull().sum()
#nonull values


# In[23]:


# df.dtypes
#founded must be obj as it is date column


# In[24]:


##f.isnull().sum()


# In[25]:


#Step1:No irrelevent columns
#Step2:Change dtype of some columns.
#Step3:Null value Treatment


# In[26]:


# df_cat=df.select_dtypes(include=object)
# df_cat.head(2)


# In[ ]:





# In[27]:


# #The below code checks if there are special charaters in a cate column or not(like $,#,-1)
# for i in df_cat.columns:
#     print(df_cat[i].value_counts())


# In[28]:


# df_cat.dtypes
#The columns in df_cat df are object dtype.All columns have correct dtypes.


# In[ ]:





# In[29]:


#Job.Title has no special chars...no -1
#Job.Description has no -1 value
#Company.Name has no -1 value
#Location col has -1 value.count is 12
##Headquarters col has -1 value.count is 72.
#Size col has -1 value.count is 134.
#Type.of.ownership col has -1 value.count is 101
#Industry col has -1 value.count is 249.
#Sector col has -1 value.count is 248.
#Revenue col has -1 value.Count is 498.
#Competitors col has -1 value.count is 639.


# In[ ]:





# In[ ]:





# In[30]:


# df[df.columns[0]]


# In[ ]:





# In[31]:


#no location means how will job title exist?
#Replace -1 with Locationn not known,found


# In[ ]:





# In[32]:


df1.loc[df1.Location=='-1','Location']='not mentioned'


# In[ ]:





# In[33]:


df1.loc[df1.Location=='-1']
#I have replaced -1 with not mentioned


# In[34]:


df1.loc[df1.Headquarters=='-1',['Headquarters']]='not mentioned'


# In[35]:


df1.loc[df1.Location=='-1']


# In[36]:


df1.loc[df1.Size=='-1',['Size']]='not mentioned'


# In[37]:


df1.loc[df1.Size=='-1']
#I have replaced -1 with not mentioned.
#Number of employees in the company is not mentioned.it is the meaning of size=-1.
#size of the company is not mentioned.


# In[38]:


df1.loc[df1.Revenue=='-1',['Revenue']]='not mentioned'


# In[39]:


#There are some job titles or vacancies in some companies where the companies revenue is not mentioned.


# In[40]:


df1.loc[df1.Competitors=='-1',['Competitors']]='not mentioned'
#There are some job tiles in some companies whose competetors is not mentioned  and their revenue is also not mentioned.
#I have repalce -1 values in cate vars using not mentioned string.


# In[41]:


# for i in df_num.columns:
#     print(df_num[i].value_counts())


# In[42]:


# df.columns


# In[43]:


df1.loc[df1[df1.columns[8]]=='-1',df1.columns[8]]='not mentioned'


# In[44]:


df1.loc[df1[df1.columns[8]]=='-1',df1.columns[8]]


# In[45]:


# df.loc[df.Type.of.ownership=='-1',['Size']]='not mentioned'


# In[46]:


df1.loc[df1['Industry']=='-1',['Industry']]='not mentioned'
#where industry is -1 it means that industry type is not mentioned.


# In[47]:


df1.loc[df1['Industry']=='-1']
#For some job titles the industry type is not mentioned.


# In[48]:


df1.loc[df1.Sector=='-1',['Sector']]='not mentioned'
#There are some job titles  whose Sector is not mentioned


# In[49]:


#there are many companies which do not have headquarters.say 72.
#These 72 companies are registered,legal without headquarters.
#Set -1 with 'No Headquarters'.I cannot replace -1 with mode value.I cannot replace -1 with say Banglore.
#A company which does not have hq cannot be set to say banglore.
#I cannot assign any headquarter city name to a company.


# In[50]:


# df.Location


# In[51]:


# df.Headquarters


# In[52]:


# df.columns


# In[53]:


#df.loc[df.Size=='-1']['Job.Title']
#I am selecting the job titles with size==-1.
#These job titles are there in company with no employees.This is logically wrong.
#generally,when job titles are there vacancies are there then company will have employees.


# In[54]:


# df.columns[8]


# In[55]:


# df[df.columns[8]]


# In[56]:


# df[df.columns[8]].value_counts()

# for i in df_cat.columns:
#     print(i)

#check for -1 values in type of ownership,Rating column,

# for i in df_cat.columns:
#     print(df_cat[i].value_counts())
# df1_num=df1.select_dtypes(include=['int64','float64'])
# df1_num
#The rating column has float values its dtype is float.When i add not mentioned striing then
#its dtype must be object.
#There are no -1 values in 2 columns in numerical dataframe.
# df_num.Hadoop.value_counts()#Hadoop col has 0 and 1.its d type must be object.
#its fine.
# df_num.Rating.value_counts().head(5)
# df.loc[df.Rating=='-1',['Rating']]
# df1.Rating.value_counts()
# #


# In[57]:


#We replaced -1 values in rating column with 0.Only then we can plot pie chart.
#when we repl -1 to not mentioned we can not p pie ch.


# In[58]:


df1.loc[df1.Rating==-1,['Rating']]=0
#Rating column had -1 not '-1'.other col had '-1'
#There are some JOb titles or vacancies in some companies whose Rating is not mentioned.
#There ratings of the companies are not mentioned.
#CHanges are saved in original df named df not in df_cat or df_num.


# In[59]:


#In df dataframe,The rating column has not mentioned string.
#df_num df has -1 values.let it have.not an issue.


#  The df1 dataframe has no '-1' values.The cate cols do not have '-1'.The num vars also do not have
#  '-1' values.All the  '-1' values are replaced by 'not mentioned'.

# In[60]:


# for i in df1.columns:
#     print(df1[i].value_counts())correct


# In[61]:


df1.head(4)


# In[ ]:





# In[62]:


#Piechart 


# In[63]:


df1[df1[df1.columns[0]]=='Data Scientist']['Rating'].mean()


# In[64]:


df1[df1[df1.columns[0]]=='Machine Learning Engineer']['Rating'].mean()


# In[65]:


rr=[3.015,3.63,3.862,2.456]


# In[66]:


df1[df1.columns[0]].value_counts()


# In[67]:


cc=df1[df1.columns[0]].value_counts().index


# In[68]:


cc


# In[ ]:





# In[69]:


df1['Rating'].describe()


# In[ ]:





# In[70]:


#Pie chart plots the distribution of rating for each job titles


# In[71]:


plt.pie(x=rr,labels=cc,autopct='%.2f%%')#'%.2f'
plt.show()


# In[72]:


Inference:
Data Engineer Job  have vacancies are present in high Rating companies.


# In[ ]:


# df_num.head(3)
#There are no -1 values in any of the columns in the df.
#no -1 values in numerical and cate gorical df.
# df.loc[df[df.columns[8]]=='-1']['Job.Title'].value_counts()
#Where type of ownership is -1 the job titles are displayed.
#There are 69 Data Scientist job vacancies  where type of ownership is -1.
#So,replace the -1 values in TO ownership column with 'NOt Known'.
# df.Industry.value_counts().head(8)
# df.shape
#103 job titles are there in IT services industry
#249 job titles are there in -1 type of industry
#I cannot replace 249 job titles in -1 type of industry with IT services
#it is logically wrong.with a particuly type only.
#set -1 with Not Known,not registered.
#these 249 job titles ka industry type is not known.
#company itself isnot known then how will i now the industry type?
# df_num=df.select_dtypes(include=['int64','float64'])
# df_num
#rating of company
#loc and headquarters of the company
#size of the company
#year in which the company was founded
#revenue and competitors of the company
#Python to SAS are features which have categories in it.yes means 1 and no means 0.so these 6 columns are
#categorical columns with dtype object.
# def unique():
#     for i in df.columns:
#         print(df[i].unique())
# unique()
#competitors,founded,job description columns have many unique values.
# df['Job.Title'].unique()
#df.Rating
#df['Company.Name'].unique()
# df.groupby(by=['Company.Name'])['Rating'].mean().sort_values(ascending=False).head(5)
# df_num=df.select_dtypes(include=['int64','float64'])
#df1.Rating
#df_cat=df.select_dtypes(include=['object'])
# df_cat.head()
# for i in df_num:
#print(df_num[i].value_counts())
# df_cat
#number of categories in job title colum are 4.
##number of categories in Job.Description column are many.
#The paragraphs length is high.
#number of categories in Company.Name column are around 10.
#number of categories in location column is around 30.
#number of categories in Headquarters column is around 30.
#Size
##number of categories in Size column is around 5.
# for i in df_cat:
#     print(df_cat[i].value_counts())
# for i in df.columns:
#     print(df[i].value_counts())


# Machstatz is highly rated company.

# In[ ]:


# df.Size.value_counts()


# count of each and every value in Size column
# number of times 1 to 50 empp value occuring in Size column is 226
# many companies hire employee btw 1 to 50 only.
# There are very few companies hiring in bulk.
# 226 companies hire few companies(less than 50)
# 168 companies hire more employees(more than 10000)
# nice inference.Contrasting extreme inference.
# 

# In[ ]:


df1.groupby(by=['Company.Name'])['Size'].value_counts().sort_values(ascending=False).head(20)
#Bi=pd.DataFrame((bind))
#bind.loc[bind.Size=='51 to 200 employees']
#df2=pd.DataFrame()
#df2['company']=list(Bi.index)#multi(2 indexes are there..)


# In[ ]:





# In[ ]:


Sanofi,Amazon,EY,Citi are the companies which have  a huge size of 10000+ employees.


# In[ ]:


ind=df1.groupby(by=['Company.Name'])['Job.Title'].value_counts().sort_values(ascending=False).head(6)
ind


# In[ ]:


#Bi variate analysis


# In[ ]:


ind.plot(kind='bar',figsize=(8,8))


# In[ ]:





# Number of data scientist job vacancies in Sanofi is 12
# Number of Data Analyst job vacancies in citi is 12.

# In[ ]:


#each and every company has say 4 job titles like d analyst,ml engineer,d scientist...
#some companies might have only data scientist roles say 4
#some company named abc may have  1 d.a,2 ml engineer etc
#Industry,sector,revenue vs location
#job title too


# In[ ]:


df.columns


# In[ ]:


#replacing -1 (a value in Industry column)with IT Services as it is 2 nd most frequent value
#-1 was the most frequent value but it has no meaning.SO We did above step for our convenience.
#what does -1 industry mean??


# In[ ]:


df1.Industry.value_counts().head(2)


# In[ ]:


# df['Industry']=df['Industry'].str.replace('-1',"IT Services")


# In[ ]:


df.Industry.mode()


# In[ ]:


df.Industry.value_counts().head(2)


# In[ ]:





# Both Industry and Location columns have -1 value which has been replaced by mode value
# in respective columns using str.replace().

# In[ ]:


#df.fillna(df.Industry) not possible as -1 is not a null value.fillna@null values
#only


# In[ ]:


len(df.loc[df.Location=='-1'])


# In[ ]:


len(df.Location)


# There are 12 locations with unidentified -1 value out of 885 values in location column
# Let us replace -1 with most frequent value in location column that is bengaluru.

# In[ ]:


# df.Location.value_counts().head(3)


# In[ ]:


# df['Location']=df['Location'].str.replace('-1','Bengaluru')


# In[ ]:


df.Location.value_counts().head(2)


# In[ ]:


df1.groupby(by=['Location'])['Industry'].value_counts().sort_values(ascending=False).head(10)
#we can type another code
#df.locdf.loc==bangalore[industry].value_counts()
#there are 156 companies in bengaluru based on IT Services industry...
#Count of IT services value in industry column in Bengaluru is 156.


# In[ ]:


df1.loc[df1.Location=='Bengaluru']['Company.Name'].value_counts().head()


# Quantzig,Walmart,String BIo re some companies which have many job vacancies in Bengaluru.
# 

# In[ ]:


df1.loc[df1.Location=='Bengaluru']['Industry'].value_counts().head()


# Many companies in Bengaluru belong to IT Services industry.

# In[ ]:


#Industry,sector,revenue vs location
#job title too


# In[ ]:


df.head(5)


# In[ ]:


df.Sector.value_counts().head(2)


# In[ ]:


#note to put '-1' not -1 as the value to be replaced in replace() because
#-1 is a category/class in Sector column.category/class means object or string
#only.string must be represented in ''.


# In[ ]:


# df['Sector']=df['Sector'].str.replace('-1','Information Technology')


# In[ ]:


df.Sector.value_counts().head(2)


# In[ ]:


df1.groupby(by=['Job.Title'])['Sector'].value_counts().sort_values(ascending=False).head(2)


# In[ ]:


Many Data Scientist jobs are in IT and Business Services sectors.


# In[ ]:


#revenue vs industry,company...


# In[ ]:


df1.columns


# In[ ]:


#plt.figure(figsize=(25,25))
#plt.bar(x=df.Revenue,height=df.Industry,width=0.9)
#width ranges from 0 to 1.0.9 means the bars are fatter.0.5 means bars are thinner.
#the xlabels and y labels become smaller in size as figsize changes from 10,10 to 25,25.
#bar plot plotted btw freq/count/num var and cate var not btw two cate var.


# In[ ]:


#pd.pivot_table(df,columns='Revenue',index='Industry',aggfunc='count')
df1["Revenue"].value_counts()


# In[ ]:





# In[ ]:


df["Revenue"]=df["Revenue"].str.replace('-1','No Revenue')
# -1 in revenue means the data is error or null values or company revenue should be zero 
#so we consider here it has NO revenue company


# In[ ]:


#I must not replace -1 in Revenue column with 2 nd most frequesntly occuring value using mode function.
#500+ billion (INR)  means that company's turnover is 5000 crore rs plus.
#Logically,I have to replace -1 with No revenue.I cannot replace -1 with 5000 crores.


# In[ ]:


pid=pd.crosstab(index=df1.Revenue,columns=df1.Industry)
pid#its a df
#revenue cate var's values are in row indexes
#industry cate var's values are in column indexes
#freq/count calculated for both cate vars involved


# In[ ]:


pid.iloc[-1].sort_values(ascending=False).head(3)
#loc[-1] gives error#did not replace -1 class in revenue column
#with mode value of that column using str.replace()
# Here -1 value in industry means the data is deleted or null values or industry does not have any particular belonging means it doesnot belong any category it doesnot have any partcular identification


# Investment Banking and Asset Management is the Industry in which many companies get a revenue of 500+ billion INR
# Rich companies are based on Investment banking and asset management

# In[ ]:





# In[ ]:


dip=pd.crosstab(index=df1['Company.Name'],columns=df1['Revenue'])
dip.head()


# In[ ]:


dip.iloc[:,-1].sort_values(ascending=False).head()
#in df,i am locating all rows of 1 st column from last and that last columns values are sorted in desc order and displayed


# In[ ]:


#Sanofi,EY,Amazon,Citi,Walmart are high revenue generating companies.
#count of 500+ billion inr value in revenue column of sanofi is 12...


# In[ ]:


idp=pd.crosstab(index=df1['Job.Title'],columns=df1['Revenue'])
idp.head()


# In[ ]:


idp.iloc[:,-1].sort_values(ascending=False).head()


# Many Data scientist job vacancies are there in high revenue generating companies.
# count of 500+billion inr value in revenue column corresponding to Data Scientist is 93.

# In[ ]:


#change dtype of 6 columns using astype fun.this needs to be done before using pandas,numpy operations...
df.columns


# In[ ]:


df.dtypes.tail()


# In[ ]:


#df[['Python','R Prog','Excel','Hadoop','SQL','SAS']]=df[['Python','R Prog','Excel','Hadoop','SQL','SAS']].astype('object')

#(Important code)-In one line of code i can change dtype of multiple columns to any datatype.


# In[ ]:


df.dtypes.tail(6)


# In[ ]:


df.SAS.value_counts()#SAS and python have high imbalanced data.lets take 2 vars for np,pd operations..


# In[ ]:


vin=df1.loc[df1.Python==1]['Job.Title'].value_counts()
vin#vin is a series with job titles as keys/indexes and counts as values
#count of data scientist job opportunities is 376 where job description has python keyword
#If u want to become a ds,you need to have good python knowledge.


# In[ ]:


vin.values #not vin.values()# as vin is a series.if it was df,use vin.values()


# In[ ]:


plt.bar(list(vin.keys()),list(vin.values),color=['r','b','Orange','g'],width=0.1)
#as width reduces from 1 to 0,bars become thin.


# In[ ]:


niv=df1.loc[df1.SAS==1]['Job.Title'].value_counts()
niv


# In[ ]:


#Piechart plot


# In[ ]:


Conlusion
Data Scientist,Data Analyst,Data Enginner , Machine Learning Engineer job titles are interchangeably used for search AI jobs.
Data Scientist , Data Analyst are most frequent job titles used.


# In[ ]:


gb = df1.groupby(['Job.Title'])[['Python','R Prog','Excel','Hadoop','SQL','SAS']].sum()
print(gb)


# In[ ]:


#bar plot for above analysis:
gb.plot(kind='bar',figsize=(20,10) ,title= 'Skills which has more demand Based on Titles ')


# Conclusions
# Python is top most demanded skill is requred for data scientist job.
# For data Analyst Excel, python are most demanded skills required .
# For Machine Learning Engineer Most demanded skill is Python
# For Data Engineer Most demanded skills are Sql, python and excel.

# Number of job vacancies job title wise and city wise

# In[ ]:


#For analysis,prefer value_counts() over count()


# In[ ]:


#df.groupby(by=['Location','Job.Title']).value_counts()
df1.groupby(by=['Location'])['Job.Title'].value_counts().sort_values(ascending=False).head(10)


# In[ ]:


#.head() is applicable on series and df


# There are 258 Vacancies for Data Scientist roles in Bengaluru
# There are 87 Vacancies for Data Scientist roles in Mumbai
# There are 45 Vacancies for Data Scientist roles in Hyderabad

# Final Conclusions:
# 
# Q.1) Find Companies most probable to hire an ML Engineer/Data Analyst Applicant in respect to his/her skillset.
# 
# Ans. Top Most Companies which are hiring are 'ZoomRx','Amazon','EY','Walmart','Citi','Sanfoi','Matelabs Innovations Pvt. Ltd.    etc.
# 
# Q.2) To analyse Machine Learning/Data Analyst Job Market in India and outline the segments most optimal to apply or prepare for Data Analyst/ Machine Learning Jobs.
# 
# Ans.Top Most Locations Which Are Hiring Machine Learning/Data Analyst in India Are Beangaluru,Mumbai,Pune,Chennai,Hyderbad.Top Most Demanded Skills for prepare Data Analyst/ Machine Learning Jobs are Python,Excel,SQL.

#                   Python  R Prog  Excel  Hadoop  SQL  SAS
# Job.Title  
# 
# Data Analyst
# 
#                      101       3    163      12  122   28
# 
# Data Engineer
# 
#                      58       1     54      44   67    2
# 
# Data Scientist   
#                      
#                      376      22    207     149  277   93
# 
# Machine Learning Engineer
# 
#                       17       1      9       3    6    0
# 
# 
# =>From the data we can see that Data analyst must have the skills are python,Excel,sql
# 
# =>Data analyst must have skill should be Python,excel,sql
# 
# =>Data engineer must have skill should be python,exel,sql
# 
# =>machine learning engineer should have skills are python,excel

# In[ ]:


task:u have read the dataset
3 cate vars have -1 4 num vars have -1
drop null values,
irrelevent columns,change d types.
store the cate vars in separate df and num vars in separate df
then do df_cat[i].value_counts()
df_num[i].value_counts()
u have to replace -1 wth logic
replace all 7 columns with -1 values using logic
use str.replace function
Now u will analyse the dataset.
do.iloc,df.loc barplots,piechart,scatterplots etc
no tv 
Build K means model
put data points in say 10 clusters
Analyse each cluster
Analyse each cluster
#We need to replace -1 in cate columns using logic.Then build K means model and
#analyse the data points in say 10 cluster one by one.


# In[ ]:


#df.loc[df.Rating=='-1']


# In[ ]:


df1.head(3)
#This df has no -1 values.This is a clean raw dataset required to build K means model.


# In[ ]:


# df.columns


# In[ ]:


# from sklearn.cluster import KMeans


# In[ ]:


#Scaling is not neeeded when we build K means model.(avoid scaling).Because we cannot analyse the scaled numerical
#data properly.We can analyse the numerical data properly.
#Cluster labels will be assogned for scaled num data too.Cluster labels will be assigned for unscaled num data too.
#But analysis is useless or difficult in case of un scaled data.


# In[ ]:


#Feature engineering on cate vars is must in 


# In[ ]:


df1_cat=df1.select_dtypes(include=object)
#df1 doesnot have -1 values


# In[ ]:





# In[ ]:


# df1_cat.Rating.value_counts()


# In[ ]:


#df1_cat.head(2)#keep r col alone.do not touch it.


# In[ ]:


#To calculate the count of cate levels in each cate var.
#When the count is <=4,perform one hot encoding.
#When the count is >=4,perform label encoding.
#Job description has many cate levels so u must perform  label encoding in it or remove that column.
#I prefer to remove the Job.Description column.


# In[ ]:


df1_cat.head(4)


# In[ ]:


# df1_cat=df1.select_dtypes(include=(object))


# In[ ]:





# In[ ]:


# for i in df_cat.columns:
#     print(df_cat[i].value_counts())


# In[ ]:


# df.columns[0]


# In[ ]:





# In[ ]:





# In[ ]:


df1_cat.columns[0]


# In[ ]:


df1_cat=pd.get_dummies(data=df1_cat,columns=[df1_cat.columns[0]])
#Dummy variables are stored in  df_cat dataframe.In place of that column 4 new columns will be created.
#There is no need to drop the old cate var.


# In[ ]:





# In[ ]:


df1_cat.columns[0]


# In[ ]:


df1_cat.drop(columns=[df1_cat.columns[0]],axis=1,inplace=True)
#I cannot take mode,mean,count data so i removed Job description column.


# In[ ]:


#df1_cat.head(3)


# In[ ]:


#Location column has 20 plus cate vars and all 20 cate levels are important.So we perform one hot encoding here.
#


# In[ ]:


# df_cat.drop(labels=df_cat.columns[1],axis=1,inplace=True)
import sklearn.preprocessing as p


# In[ ]:


label_encoder = p.LabelEncoder()   
# Encode labels in column 'species'. 
df1_cat['Location']= label_encoder.fit_transform(df1_cat['Location']) 


# In[ ]:


df1_cat.head(2)


# In[ ]:


#to create dummy variables for company name column


# In[ ]:


df1_cat.columns[1]


# In[ ]:





# In[ ]:


df1_cat=pd.get_dummies(data=df1_cat,columns=[df1_cat.columns[1]])


# In[ ]:


# df1_cat.head(2)


# In[ ]:





# In[ ]:


df1_cat.head(3)


# In[ ]:


df1_cat.columns[2]


# In[ ]:


df1_cat.drop(columns=[df1_cat.columns[2]],axis=1,inplace=True)


# In[ ]:


df1_cat.head(3)


# In[ ]:


# df1_cat=pd.get_dummies(data=df1_cat,columns=[df1_cat.columns[1]])


# In[ ]:


# df1_cat.head(2)


# In[ ]:





# In[ ]:


# df_cat=pd.get_dummies(data=df_cat,columns=['Location'])


# In[ ]:


df1_cat=pd.get_dummies(data=df1_cat,columns=['Size'])


# In[ ]:


df1_cat.head(2)


# In[ ]:


# df_cat.columns


# In[ ]:


# df_cat.columns[0]


# In[ ]:


df1_cat.columns[2]


# In[ ]:





# In[ ]:


df1_cat=pd.get_dummies(data=df1_cat,columns=[df1_cat.columns[2]])


# In[ ]:


df1_cat.head(4)


# In[ ]:


df1_cat.columns[2]


# In[ ]:


label_encoder = p.LabelEncoder()   
# Encode labels in column 'species'. 
df1_cat['Industry']= label_encoder.fit_transform(df1_cat['Industry']) 


# In[ ]:


df1_cat.head(3)


# In[ ]:





# In[ ]:


# len(df_cat.Competitors.value_counts())
#Compettitors col has names of competitors of the companies in which job vacancies are posted.
#It is not an important column.LEts drop it as it ir relevent and has 100 cate levels.


# In[ ]:


df1_cat.drop(labels=['Competitors'],axis=1,inplace=True)


# In[ ]:


df1_cat=pd.get_dummies(data=df1_cat,columns=['Revenue'])


# In[ ]:


df1_cat.shape


# In[ ]:


df1_cat.head(4)
#industry must be l.e not ohe.


# In[ ]:


df1_cat=pd.get_dummies(data=df1_cat,columns=['Sector'])


# In[ ]:


df1_cat.head(2)


# In[ ]:


df1_cat.Rating.value_counts().head(2)


# In[ ]:


#Rating col has not mentioned string


# In[ ]:


df1_cat.loc[df1_cat.Rating=='not mentioned',['Rating']]=1


# In[ ]:


#I have set not mentioned string with 1 integer.
#When i pass string data into the k means model model will not understand string data.


# In[ ]:


#df1.select_dtypes(include=['int64','float64']).head(3) #no num vars


# In[ ]:


df1_cat.head(2)


# In[ ]:


# for i in df1_cat.columns:
#     print(i)


# In[ ]:


#df1_cat=pd.get_dummies(data=df1_cat,columns=[df1_cat.columns[2]])


# In[ ]:


#Only ind and loc col have integer labels from 1 to say 20.its fine.label encoded


# In[ ]:


df1_cat.shape


# In[ ]:


df1_cat.head(3)


# In[ ]:


df1_cat is a df which has F.E cate vars+F.E num vars.generaly df1_cat has f.e cate vars
b its fine.Never scale num vars when u build k means algorithm
df1_cat is a raw clean dataset
We cannot train Kmeans model using string data.cate varss.error displayed.


# In[ ]:


# df_k=pd.concat([df_cat,df_num],axis=1)


# In[ ]:


# df_k.shape


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


wcss=[]
for i in range(2,20):
#     print("hi")    
    kmeans = KMeans(n_clusters = i, random_state = 10)
    kmeans.fit(df1_cat)
    wcss.append(kmeans.inertia_)


# In[ ]:


plt.plot(range(2,20), wcss)

# set the axes and plot labels
# set the font size using 'fontsize'
plt.title('Elbow Plot', fontsize = 15)
plt.xlabel('No. of clusters (K)', fontsize = 15)
plt.ylabel('WCSS', fontsize = 15)

# display the plot
plt.show()


# In[ ]:


from sklearn.metrics import silhouette_score


# In[ ]:


n_clusters = [5,6,7,8]

for K in n_clusters:
    cluster = KMeans (n_clusters= K, random_state= 10)
    predict = cluster.fit_predict(df1_cat)
    score = silhouette_score(df1_cat, predict, random_state= 10)
    print("For {} clusters the silhouette score is {})".format(K, score))


# In[ ]:


# for i in df1_cat.columns:
#     print(i)
    
#there are so many companies.w.k.t amazon,city,sanofi,city 4 have high size and generate high 
#revenue something.Focus on these 3 cities and implement them in df.loc[] in cluster 0.


# In[ ]:


# Build the Clusters
# Let us build the 6 clusters using K-means clustering.


# In[ ]:


#I set kmeans model with 6 clusters.
new_clusters = KMeans(n_clusters =6,random_state = 10)

#Kmeans model gets trained on observations of the df named as df1_cat.
#It has both num and cat vars.
#The data is grouped in 6 clusters.
new_clusters.fit(df1_cat)

#all dp are assigned cluster number
df1_cat['Cluster'] = new_clusters.labels_


# In[ ]:


df1_cat.Cluster.value_counts()


# In[ ]:


#We will split out raw clean dataset into 6 clusters and analyse all the cluster.


# In[ ]:


df1_cat.tail()
#df1_cat is raw clean dataset


# In[ ]:





# In[ ]:


# All the observations are clustered in 6 clusters.
# We will analyse 


# In[ ]:


df1_cat.loc[df1_cat['Cluster']==0].describe()


# In[ ]:


#There are 228 datapoints in cluster 0.
#The job related data in cluster 0 is different from job related data in cluster 1 and other clusters.
#Industry is label encoded.SO do not make inference from mean,median value .It will give wrong inference.
#For all teh encoded cate variables u must focus on min and max values only.
#Mean,median values of encoded cate vars will give wrog inference.
#


# In[ ]:


df_0=df1_cat.loc[df1_cat['Cluster']==0]


# In[ ]:


df_0.head()#this df has data points in cluster 0


# In[ ]:


# df_0.loc[df_0['Job.Title_Data Scientist']==1].head(3)


# In[ ]:


#df_0.loc[df_0['Job.Title_Data Scientist']==1]['Job.Title_Data Scientist']


# In[ ]:


len(df_0.loc[df_0['Job.Title_Data Scientist']==1]['Job.Title_Data Scientist'])


# In[ ]:


#There are 161 Data Scientist roles in cluster 0.


# In[ ]:


len(df_0.loc[((df_0['Job.Title_Data Scientist']==1)&(df_0['Python']==1))])


# In[ ]:


print("Percentage of Data Scientist job roles where python skill is must ",(100/161*100))
#In cluster 0,62 percentage of DS roles require python skill.


# Inference 1:
# In cluster 0,the count of data scientist jobs which require python skill(compulsary) is 100.

# In[ ]:





# In[ ]:


#df_0.columns[450:500]


# In[ ]:


len(df_0.loc[((df_0['Job.Title_Data Scientist']==1)&(df_0['Revenue_500+ billion (INR)']==1))])


# In[ ]:


5/161


# In[ ]:


print("Percentage of data scientist jobs in high revenue generating companies wrt count of ds jobs in cluster 0 ",(5/161)*100)


# Inference 2:
# In cluster 0,count of data scientist job roles in high revenue companies is just 5.

# In[ ]:


#df1.Sector.value_counts()
#How many data Scientist jobs are there in IT sector
#Howmany D Scientits jobs are there in Business Services sector?
#use this concept in clustrer 0 
#apply df.loc[with 2 conditions] three times in cluster 0
#apply df.loc[with 2 conditions] three times in cluster 1
#In cluster 0,count of d scientitst job vacancies in sector of IT is say 20
#In cluster 1,count of d scientist job vacanciser in IT is say 30
##In cluster 0,count of d scientitst job vacancies in sector of Business is say 20
#In cluster 1,count of d scientist job vacanciser in business is say 30
#calculate the percentage value for previsou s4 sentences.


# In[ ]:


#df_0.columns[400:500]


# In[ ]:


len(df_0.loc[((df_0['Job.Title_Data Scientist']==1)&(df_0['Size_10000+ employees']==1))])


# In[ ]:


In cluster 0,count of Data Scientist roles in companies with Size_10000+ employees is just 6.
Few Ds job titles are present in bulk companies.(companies with many employees)


# Inference 3:
# In cluster0,count of Data Scientist roles in high size companies is just 6.

# In[ ]:


#df_0.head(2)# df1.Size.value_counts() # df_0['Company.Name_Amazon'].value_counts() 
#In cluster 0 no data scientist job is available in amazon company.
# df_0.loc[(df_0['Company.Name_Amazon']==1) # df_0.loc[((df_0['Job.Title_Data Scientist']==1)&(df_0['Company.Name_Amazon']==1))]


# In[ ]:


df1_cat.loc[df1_cat['Cluster']==1].describe()


# In[ ]:


df_1=df1_cat.loc[df1_cat['Cluster']==1]


# In[ ]:


len(df_1)


# In[ ]:


#There are 154 data points in cluster 1


# In[ ]:


#df_1 is a df which has cluster 1 ka data.


# In[ ]:


df_1.loc[df_1['Job.Title_Data Scientist']==1]['Job.Title_Data Scientist']


# In[ ]:


len(df_1.loc[df_1['Job.Title_Data Scientist']==1])
#In cluster 1 i display data scientist data.
#Count of data scientists in cluster 1 are 102 while there are 161 ds roles in cluster 0.clustering is good and proper.


# In[ ]:





# There are 102 Data Scientist roles in cluster 1.

# In[ ]:


len(df_1.loc[((df_1['Job.Title_Data Scientist']==1)&(df_1['Python']==1))])


# In[ ]:


print("Percentage of data Scientist job roles  where python skill is must is ",(77/102)*100)


# In cluster 1,the count of data scientist jobs which require python skill(compulsary) is 77.

# In[ ]:


#COunt of Data Scientist jobs which require python skill is just 77 compared too 100 in cluster 0.
#The data points in cluster 0 have different properties compared to data points in cluster 1.


# In[ ]:


len(df_1.loc[((df_1['Job.Title_Data Scientist']==1)&(df_1['Revenue_500+ billion (INR)']==1))])


# In[ ]:


#2 cate vars.cross tab?
#pd.cross tab?
####pd.cross tab function


# In[ ]:


#df1_cat.iloc[[9],[15]]#i access row index 9 value in column index 15.df1_cat.iloc[:,:60]


# In[ ]:


#161:5  cluster 0
#102:31 cluster 1


# In[ ]:


#focus on cluster number 3 and 4


# In[ ]:


df1_cat.Cluster.value_counts()


# In[ ]:


df_3=df1_cat.loc[df1_cat['Cluster']==3]


# In[ ]:


df_3.head(3)


# In[ ]:


len(df_3)


# In[ ]:


len(df_3.loc[((df_3['Job.Title_Data Scientist']==1)&(df_3['Python']==1))])


# In[ ]:


len(df_3.loc[df_3['Job.Title_Data Scientist']==1])


# In[ ]:





# In[ ]:


print("Percentage of Data Scientist job roles where python skill is must ",(35/40)*100,"percentage")
#In cluster 3,87.5 percentage of DS roles require python skill


# In[ ]:


len(df_3.loc[((df_3['Job.Title_Data Scientist']==1)&(df_3['Size_10000+ employees']==1))])


# In[ ]:


len(df_3.loc[((df_3['Job.Title_Data Scientist']==1)&(df_3['Size_5001 to 10000 employees']==1))])


# In[ ]:


#In cluster 3,only 9 data scientist job roles are present in large size companies.
#


# In[ ]:


#df_3.columns[400:500]


# In[ ]:


df_4=df1_cat.loc[df1_cat['Cluster']==4]


# In[ ]:


len(df_4)


# In[ ]:


len(df_4.loc[((df_4['Job.Title_Data Scientist']==1)&(df_4['Size_10000+ employees']==1))])


# In[ ]:


len(df_4.loc[((df_4['Size_5001 to 10000 employees']==1)&(df_4['Size_5001 to 10000 employees']==1))])


# In[ ]:


# # plot the lmplot to visualize the clusters
# # pass the different markers to display the points in each cluster with different shapes
# # the 'hue' parameter returns colors for each cluster
# sns.lmplot(x = 'Cust_Spend_Score', y = 'Yearly_Income', data = df1_num, hue = 'Cluster', 
#                 markers = ['*', ',', '^', '.', '+',',','+'], fit_reg = False, size = 10)

# # set the axes and plot labels
# # set the font size using 'fontsize'
# plt.title('K-means Clustering (for K=7)', fontsize = 15)
# plt.xlabel('Spending Score', fontsize = 15)
# plt.ylabel('Annual Income', fontsize = 15)

# # display the plot
# plt.show()


# In[ ]:


#In cluster 4,24 data scientist job vacancies are present in large size companies.24 out of 57 .
#almost half.high fraction value in cluster 4 compared to cluster 3.


# Innovative techniques applied in new project:

# 
# We have build K means algorithm in this DS project.In existing system Data preprocessing,Feature engineering and Exploratory Data
# Analysis was performed.We have used Elbow plot,Silhouette scores to analyse the properties of data in each cluster.
# Data patterns in each clusters are different.We can analyse the properties of data in each clusters to obtain meaningfull
# insights for business.

# In[ ]:




