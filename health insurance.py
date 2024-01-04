#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np #numarical data 
import pandas as pd #processing the data 
import matplotlib.pyplot as plt #graphic representation 
import seaborn as sns #graphic representation 


get_ipython().run_line_magic('matplotlib', 'inline')

# %matplotlib inline will lead to static images of your plot embedded in the same  notebook


# In[7]:


df_train=pd.read_csv("C:\\Users\\dell\\OneDrive\\Desktop\\train .csv")


# In[8]:


df_train.head(10)


# In[9]:


df_train.isnull().sum()
# checking for null values


# In[10]:


df_train.info()
# checking for datatypes and other details


# In[11]:


sns.distplot(df_train['Region_Code'])
# plot for visualisation (distribution plot)


# In[12]:


sns.distplot(df_train['Holding_Policy_Type'])
# plot for visualisation (distribution plot)


# In[13]:


sns.distplot(df_train['Reco_Policy_Premium'])
# plot for visualisation (distribution plot)


# In[14]:


df_train['Reco_Policy_Premium'].describe()
# this feature describes the premium for policy(vluecnt)


# In[15]:


df_train['Accomodation_Type'].value_counts().plot(kind = 'pie')
#visualisation of features


# In[16]:


df_train.Accomodation_Type.value_counts()
# this is object datatype


# In[17]:


df_train['Accomodation_Type']=df_train['Accomodation_Type'].map({'Rented':0,
                             'Owned':1
    
})
df_train.Accomodation_Type.value_counts()

# maping features in order to make all int values(cat>num value)


# In[18]:


df_train['Reco_Insurance_Type'].value_counts()


# In[19]:


df_train['Reco_Insurance_Type'].value_counts().plot(kind = 'pie')


# In[20]:


df_train['Reco_Insurance_Type']=df_train['Reco_Insurance_Type'].map({'Individual':0,
                             'Joint':1
    
})
df_train.Reco_Insurance_Type.value_counts()

# maping features in order to make all int values


# In[21]:


df_train['Is_Spouse'].value_counts()


# In[22]:


df_train['Is_Spouse']=df_train['Is_Spouse'].map({'No':0,
                             'Yes':1
    
})
df_train.Reco_Insurance_Type.value_counts()


# In[23]:


df_train['City_Code'].value_counts().plot(kind = 'bar')#


# In[24]:


df_train['City_Code'].unique()


# In[25]:


from sklearn.preprocessing import LabelEncoder#cat to num
LE=LabelEncoder()
df_train['City_Code']=LabelEncoder().fit_transform(df_train['City_Code'])


# In[26]:


df_train.head(10)


# In[27]:


df_train['Holding_Policy_Duration'].unique()
# here we found that 14+(object) value needs to be changed


# In[28]:


df_train['Holding_Policy_Duration']=df_train['Holding_Policy_Duration'].replace('14+',15.0)
# value replaced
df_train['Holding_Policy_Duration'].replace(np.nan,df_train['Holding_Policy_Duration'].mode()[0],inplace=True)
# treatment of null values
df_train['Holding_Policy_Duration']=df_train['Holding_Policy_Duration'].astype(float)
# changing datatype str to float


# In[29]:


df_train.info()


# In[30]:


df_train['Holding_Policy_Duration'].unique()
# checking unique value


# In[31]:


df_train['Health Indicator'].unique()


# In[32]:


df_train['Health Indicator'].replace(np.nan,df_train['Health Indicator'].mode()[0],inplace=True)#inpls save changes
df_train['Health Indicator'].value_counts()
# treatment of null values and crosschecking it


# In[33]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df_train['Health Indicator']=LabelEncoder().fit_transform(df_train['Health Indicator'])
df_train['Health Indicator'].value_counts()

# LabelEncoder can be used to normalize labels.  


# In[34]:


df_train['Holding_Policy_Type'].replace(np.nan,df_train['Holding_Policy_Type'].mode()[0],inplace=True)
df_train['Holding_Policy_Type'].value_counts()


# In[35]:


df_train.isnull().sum()


# In[36]:


df_train.info()
 # cleaning isdone 


# In[37]:


for col in df_train.iloc[:,:-1].columns:
    print(col)
    sns.boxplot(x=df_train[col],data=df_train)
    plt.show()
    
# boxplot for checking of Outliers


# In[38]:


def boxoutlier(var):
    for x in var.iloc[:,:-1].columns :        
        Q1=var[x].quantile(0.25)
        Q3=var[x].quantile(0.75)
        IQR=Q3-Q1
        Lower = Q1-(1.5*IQR)
        Upper = Q3+(1.5*IQR)
        var.loc[:,x]=np.where(var[x].values > Upper,Upper,var[x].values)
        var.loc[:,x]=np.where(var[x].values < Lower,Lower,var[x].values)
        
    return var
df1=boxoutlier(df_train)


# In[39]:


for col in df1.iloc[:,:-1].columns:
    print(col)
    sns.boxplot(x=df1[col],data=df1)
    plt.show()


# In[40]:


X=df1.drop('Response',axis=1)#indpndt vari


# In[41]:


y=df1['Response']#target


# In[42]:


X.info()


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4 )


# In[45]:


display(X_train.head(),y_train.head(),'Testing Data',X_test.head(),y_test.head())


# # ** Checking model on unseen data ('Test.csv' file)**

# In[46]:


df_test=pd.read_csv("C:\\Users\\dell\\OneDrive\\Desktop\\train .csv")


# In[47]:


df_test.info()


# In[48]:


df_test.isnull().sum()
# checking for null values


# In[49]:


df_test['Accomodation_Type']=df_test['Accomodation_Type'].map({'Rented':0,
                             'Owned':1
    
})
df_test.Accomodation_Type.value_counts()

# maping features in order to make all int values


# In[50]:


df_test['Reco_Insurance_Type']=df_test['Reco_Insurance_Type'].map({'Individual':0,
                             'Joint':1
    
})
df_test.Reco_Insurance_Type.value_counts()

# maping features in order to make all int values


# In[51]:


df_test['Is_Spouse']=df_test['Is_Spouse'].map({'No':0,
                             'Yes':1
    
})
df_test.Reco_Insurance_Type.value_counts()


# In[52]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df_test['City_Code']=LabelEncoder().fit_transform(df_test['City_Code'])


# In[53]:


df_test.sample(n=5)


# In[54]:


df_test['Holding_Policy_Duration'].unique()
# here we found that 14+ value needs to be changed


# In[55]:


df_test['Holding_Policy_Duration']=df_test['Holding_Policy_Duration'].replace('14+',15.0)
# value replaced
df_test['Holding_Policy_Duration'].replace(np.nan,df_test['Holding_Policy_Duration'].mode()[0],inplace=True)#oind
# treatment of null values
df_test['Holding_Policy_Duration']=df_test['Holding_Policy_Duration'].astype(float)
# changing datatype str to float


# In[56]:


df_test['Holding_Policy_Duration'].unique()


# In[57]:


df_test['Health Indicator'].replace(np.nan,df_test['Health Indicator'].mode()[0],inplace=True)
df_test['Health Indicator'].value_counts()


# In[58]:


df_test['Health Indicator'].isnull().sum()


# In[59]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df_test['Health Indicator']=LabelEncoder().fit_transform(df_test['Health Indicator'])
df_test['Health Indicator'].value_counts()

# LabelEncoder can be used to normalize labels.
# It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels


# In[60]:


df_test['Holding_Policy_Type'].replace(np.nan,df_test['Holding_Policy_Type'].mode()[0],inplace=True)
df_test['Holding_Policy_Type'].value_counts()


# In[61]:


df_test.isnull().sum()


# In[62]:


df_test.info()


# In[63]:


for col in df_test.iloc[:,:-1].columns:
    print(col)
    sns.boxplot(x=df_test[col],data=df_test)
    plt.show()


# In[64]:


df2=boxoutlier(df_test)


# In[65]:


for col in df2.iloc[:,:-1].columns:
    print(col)
    sns.boxplot(x=df2[col],data=df2)
    plt.show()


# In[66]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a Logistic Regression model
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
logistic_predictions = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, logistic_predictions)
print(f"Logistic Regression Accuracy: {accuracy}")


# In[67]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a Random Forest model
random_forest_model = RandomForestClassifier()

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
random_forest_predictions = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, random_forest_predictions)
print(f"Random Forest Accuracy: {accuracy}")


# In[69]:


from sklearn.neighbors import KNeighborsClassifier

# Create a KNN model
knn_model = KNeighborsClassifier()

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the test set
knn_predictions = knn_model.predict(X_test)

# Evaluate the model
accuracy_knn = accuracy_score(y_test, knn_predictions)
print(f"KNN Accuracy: {accuracy_knn}")


# In[ ]:




