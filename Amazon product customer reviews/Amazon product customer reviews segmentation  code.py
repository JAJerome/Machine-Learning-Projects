#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("amazon_books_Data.csv")
df


# In[3]:


data=df.iloc[:,[-6,-4]]


# In[4]:


data


# In[5]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
data.iloc[:,-1]=label.fit_transform(data.iloc[:,-1])


# In[6]:


data


# In[7]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[8]:


corpus=[]
for i in range(0,100):
    reviews=re.sub("[^A-Za-z]",' ',data['review_body'][i])
    reviews=reviews.lower()
    reviews=reviews.split()
    pst=PorterStemmer()
    reviews=[pst.stem(word)for word in reviews if word not in set(stopwords.words("english"))]
    reviews=''.join(reviews)
    corpus.append(reviews)


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
X=tf.fit_transform(corpus).toarray()
y=data.iloc[:,-1].values


# In[10]:


X


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[13]:


from sklearn.naive_bayes import BernoulliNB
ber=BernoulliNB()
ber.fit(X_train,y_train)
y_pred=ber.predict(X_train)


# In[14]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[15]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[16]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[17]:


from sklearn.svm import SVC
svc=SVC(C= 0.1, gamma= 'auto', kernel='rbf')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_train)


# In[18]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[19]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[20]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[21]:


from sklearn.model_selection import GridSearchCV
params={"C":[0.1,0.001],"kernel":["rbf","linear"],"gamma":["auto","scale"]}
grid=GridSearchCV(estimator=svc,param_grid=params,cv=5)
grid.fit(X_train,y_train)
print("best params:",grid.best_params_)
print("best score:",grid.best_score_)


# In[22]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_train)


# In[23]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[24]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[25]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[26]:


from sklearn.model_selection import GridSearchCV
params={"n_neighbors":[5,7,10],"weights":["uniform","distance"]}
grid=GridSearchCV(estimator=knn,param_grid=params,cv=5)
grid.fit(X_train,y_train)
print("best_score:",grid.best_score_)
print("best_params:",grid.best_params_)


# In[27]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_train)


# In[28]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[29]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[30]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[31]:


from sklearn.model_selection import RandomizedSearchCV
params={'criterion': ['gini', 'entropy'],
    'max_depth':[1,3,5,7,10],
    'min_samples_split':[2,5,6,10,15, 20],
    'min_samples_leaf':[2,5,6,10]}
random=RandomizedSearchCV(estimator=dtree,param_distributions=params,cv=5)
random.fit(X_train,y_train)
print("best_score:",random.best_score_)
print("best_params:",random.best_params_)


# In[32]:


from sklearn.ensemble import RandomForestClassifier
rndf=RandomForestClassifier()
rndf.fit(X_train,y_train)
y_pred=rndf.predict(X_train)


# In[33]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[34]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[35]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[36]:


from sklearn.model_selection import RandomizedSearchCV
params={'n_estimators':[100, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth':[1,3,5,7,10],
    'min_samples_split':[2,5,6,10,15, 20],
    'min_samples_leaf':[2,5,6,10]}
random=RandomizedSearchCV(estimator=rndf,param_distributions=params,cv=5)
random.fit(X_train,y_train)
print("best_score:",random.best_score_)
print("best_params:",random.best_params_)


# In[37]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_train)


# In[38]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[39]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[40]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# Amazon Grocery DataSet

# In[41]:


data2=pd.read_csv(r"Downloads\amazon_grocery_Data.csv")
data2


# In[42]:


df2=data2.iloc[:,[-3,-1]]
df2


# In[43]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df2.iloc[:,-1]=label.fit_transform(df2.iloc[:,-1])


# In[44]:


df2


# In[45]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[46]:


corpus2=[]
for i in range(0,100):
    reviews=re.sub("^A-Za-z",' ',df2['review_body'][i])
    reviews=reviews.lower()
    reviews=reviews.split()
    pst=PorterStemmer()
    reviews=[pst.stem(word)for word in reviews if word not in set(stopwords.words("english"))]
    reviews=' '.join(reviews)
    corpus2.append(reviews)


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus2).toarray()
y=df2.iloc[:,-1].values


# In[48]:


from sklearn.naive_bayes import BernoulliNB
ber=BernoulliNB()
ber.fit(X_train,y_train)
y_pred=ber.predict(X_train)


# In[49]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[50]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[51]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[52]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_train)


# In[53]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[54]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[55]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[56]:


from sklearn.model_selection import GridSearchCV
params={"C":[0.1,0.001],"kernel":["rbf","linear"],"gamma":["auto","scale"]}
grid=GridSearchCV(estimator=svc,param_grid=params,cv=5)
grid.fit(X_train,y_train)
print("best params:",grid.best_params_)
print("best score:",grid.best_score_)


# In[57]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_train)


# In[58]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[59]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[60]:


from sklearn.model_selection import GridSearchCV
params={"n_neighbors":[5,7,10],"weights":["uniform","distance"]}
grid=GridSearchCV(estimator=knn,param_grid=params,cv=5)
grid.fit(X_train,y_train)
print("best_score:",grid.best_score_)
print("best_params:",grid.best_params_)


# In[61]:


from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred=tree.predict(X_train)


# In[62]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[63]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[64]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[65]:


from sklearn.model_selection import RandomizedSearchCV
params={'criterion': ['gini', 'entropy'],
    'max_depth':[1,3,5,7,10],
    'min_samples_split':[2,5,6,10,15, 20],
    'min_samples_leaf':[2,5,6,10]}
random=RandomizedSearchCV(estimator=dtree,param_distributions=params,cv=5)
random.fit(X_train,y_train)
print("best_score:",random.best_score_)
print("best_params:",random.best_params_)


# In[66]:


from sklearn.ensemble import RandomForestClassifier
rndf=RandomForestClassifier()
rndf.fit(X_train,y_train)
y_pred=rndf.predict(X_train)


# In[67]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[68]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[69]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[70]:


from sklearn.model_selection import RandomizedSearchCV
params={'n_estimators':[100, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth':[1,3,5,7,10],
    'min_samples_split':[2,5,6,10,15, 20],
    'min_samples_leaf':[2,5,6,10]}
random=RandomizedSearchCV(estimator=rndf,param_distributions=params,cv=5)
random.fit(X_train,y_train)
print("best_score:",random.best_score_)
print("best_params:",random.best_params_)


# In[71]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_train)


# In[72]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,y_pred)
acc


# In[73]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
cm


# In[74]:


from sklearn.metrics import classification_report
report=classification_report(y_train,y_pred)
report


# In[ ]:




