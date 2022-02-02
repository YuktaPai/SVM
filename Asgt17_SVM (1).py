#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importig Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# ### Data

# In[2]:


# Loading data
train_data = pd.read_csv('C:/Users/17pol/Downloads/SalaryData_Train(1).csv')
test_data = pd.read_csv('C:/Users/17pol/Downloads/SalaryData_Test(1).csv')


# ### EDA & Data Preprocessing

# In[3]:


train_data.shape


# In[4]:


test_data.shape


# In[5]:


train_data.info()


# In[6]:


test_data.info()


# In[7]:


train_data.isna().sum()


# In[8]:


test_data.isna().sum()


# In[9]:


train_data.head()


# In[10]:


test_data.head()


# In[11]:


# frequency for categorical fields 
category_col =['workclass', 'education','maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native', 'Salary'] 
for c in category_col:
    print (c)
    print (train_data[c].value_counts())
    print('\n')


# In[12]:


# countplot for all categorical columns
import seaborn as sns
sns.set(rc={'figure.figsize':(15,8)})
cat_col = ['workclass', 'education','maritalstatus', 'occupation', 'relationship', 'race', 'sex','Salary']
for col in cat_col:
    plt.figure() #this creates a new figure on which your plot will appear
    sns.countplot(x = col, data = train_data, palette = 'Set2');


# In[13]:


# printing unique values from each categorical columns

print('workclass',train_data.workclass.unique())
print('education',train_data.education.unique())
print('maritalstatus',train_data['maritalstatus'].unique())
print('occupation',train_data.occupation.unique())
print('relationship',train_data.relationship.unique())
print('race',train_data.race.unique())
print('sex',train_data.sex.unique())
print('native',train_data['native'].unique())
print('Salary',train_data.Salary.unique())


# In[14]:


train_data[['Salary', 'age']].groupby(['Salary'], as_index=False).mean().sort_values(by='age', ascending=False)


# In[15]:


plt.style.use('seaborn-whitegrid')
x, y, hue = "race", "prop", "sex"
#hue_order = ["Male", "Female"]
plt.figure(figsize=(20,5)) 
f, axes = plt.subplots(1, 2)
sns.countplot(x=x, hue=hue, data=train_data, ax=axes[0])

prop_df = (train_data[x]
           .groupby(train_data[hue])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())

sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=axes[1])


# In[16]:


g = sns.jointplot(x = 'age', 
              y = 'hoursperweek',
              data = train_data, 
              kind = 'hex', 
              cmap= 'hot', 
              size=10)

#http://stackoverflow.com/questions/33288830/how-to-plot-regression-line-on-hexbins-with-seaborn
sns.regplot(train_data.age, train_data['hoursperweek'], ax=g.ax_joint, scatter=False, color='grey')


# ### Feature encoding

# In[17]:



from sklearn.preprocessing import LabelEncoder
train_data = train_data.apply(LabelEncoder().fit_transform)
train_data.head()


# In[18]:


test_data = test_data.apply(LabelEncoder().fit_transform)
test_data.head()


# ### Test-Train-Split

# In[19]:



drop_elements = ['education', 'native', 'Salary']
X = train_data.drop(drop_elements, axis=1)
y = train_data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)





# ### Building SVM Model

# In[20]:


from sklearn import metrics
svc = SVC()
svc.fit(X_train, y_train)


# In[21]:


# make predictions
prediction = svc.predict(X_test)


# In[22]:


# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))


# ### Testing it on new test data from SalaryData_Test(1).csv

# In[23]:



drop_elements = ['education', 'native', 'Salary']
X_new = test_data.drop(drop_elements, axis=1)

y_new = test_data['Salary']


# In[24]:


# make predictions
new_prediction = svc.predict(X_new)


# In[25]:


# summarize the fit of the model
print(metrics.classification_report(y_new, new_prediction))
print(metrics.confusion_matrix(y_new, new_prediction))

print("Accuracy:",metrics.accuracy_score(y_new, new_prediction))
print("Precision:",metrics.precision_score(y_new, new_prediction))
print("Recall:",metrics.recall_score(y_new, new_prediction))


# ### Building SVM model with Hyper Parameters kernel='rbf',gamma=15, C=1

# In[27]:


model = SVC(kernel='rbf',gamma=15, C=1)



# In[28]:


model.fit(X_train, y_train)


# In[29]:


# make predictions
prediction = model.predict(X_test)


# In[30]:


# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))


# ### Testing above model on SalaryData_Test(1).csv

# In[31]:


# make predictions
new_prediction = model.predict(X_new)
# summarize the fit of the model
print(metrics.classification_report(y_new, new_prediction))
print(metrics.confusion_matrix(y_new, new_prediction))

print("Accuracy:",metrics.accuracy_score(y_new, new_prediction))
print("Precision:",metrics.precision_score(y_new, new_prediction))
print("Recall:",metrics.recall_score(y_new, new_prediction))


# ### Building SVM model with Hyper Parameters kernel='linear',gamma=0.22, C=0.1

# In[32]:



model_2 = SVC(kernel='linear',gamma=0.22, C=1)
model_2.fit(X_train, y_train)

# make predictions
prediction = model.predict(X_test)


# In[33]:


# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))


# ### Testing above model on SalaryData_Test(1).csv

# In[34]:


# make predictions
new_prediction = model_2.predict(X_new)
# summarize the fit of the model
print(metrics.classification_report(y_new, new_prediction))
print(metrics.confusion_matrix(y_new, new_prediction))

print("Accuracy:",metrics.accuracy_score(y_new, new_prediction))
print("Precision:",metrics.precision_score(y_new, new_prediction))
print("Recall:",metrics.recall_score(y_new, new_prediction))


# In[ ]:




