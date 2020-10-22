#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[29]:


data=pd.read_csv('C:\\Users\\kumar\\OneDrive\\Desktop\\Data_Mining\\Data_for_UCI_named.csv', delimiter = ',')
df = data.copy()
df_r=data.copy()
df


# In[30]:


print(df.shape)
df[0:]
df.columns
df.info()


# In[31]:


# set the style
sns.set_style("whitegrid")
# plot
splot = sns.countplot(df['stabf'])
# add the value counts on top of each bar
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + 
# align the value count text
p.get_width() / 2., p.get_height()-180), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
# further styling
plt.title("Class Balance", fontweight='bold')
plt.xlabel('Category', fontweight='bold', color='#505050')
plt.xticks(color='#606060')
plt.yticks(color='#606060')
plt.ylabel('Count', color='#505050', rotation=0, fontweight='bold', horizontalalignment='right', y=1.0)


# In[32]:


# drop the rows with na or missing values
from sklearn.preprocessing import LabelEncoder
df = df.dropna(axis=0) #chn

for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])
        df.head()

df.head()


# In[33]:


from numpy import int64
#df['stabf'].astype(str).astype('int64')
df['stabf']=df['stabf'].astype(np.int64)
#pd.to_numeric(df['stabf']).astype('int64')

df.info()


# In[34]:


#Correlation betweeen features
import seaborn as sns
correlations = df.corr()
plt.subplots(figsize=(10,8))
mask = np.zeros_like(correlations)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlations, xticklabels=data.columns, yticklabels=data.columns, 
            mask=mask,annot=True,linewidths=2, vmin=-1, fmt=".2f")
plt.show()


# In[35]:


del df['p1']


# In[36]:


def train_test_splitting(data,test_ratio):
    np.random.seed(42)
    shuffled_indices=np.random.permutation(len(data))
    test_size=int(len(data)*test_ratio)
    train_indices=shuffled_indices[:test_size]
    test_indices=shuffled_indices[test_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

X_train,X_test=train_test_splitting(df.iloc[:,:-2],0.2)
y_train,y_test=train_test_splitting(df['stabf'],0.2)

#print(X_train)
print(X_test)


# In[37]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

k_range=range(1,20)
scores={}
scores_list=[]
for k in  k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))
    
plt.plot(k_range,scores_list)
plt.xlabel("value of k for KNN")
plt.ylabel("Testing Accuracy")


# In[38]:


from sklearn.model_selection import train_test_split
X = np.array(df.drop(['stab','stabf'], axis=1))
y = np.array(df['stabf'])

# split data into training and testing datasets
X_train, X_test2, y_train, y_test2 =train_test_split(X, y, test_size=0.2, random_state=1)


# In[39]:


#plot the accuracy of KNN classifier against k value
knn=KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test2)
print("KNN(k=13):Acuuracy(train)=",metrics.accuracy_score(y_test2,y_pred))
print("Test: ")
print(metrics.classification_report(y_test2,y_pred))


# In[40]:


import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


X = np.array(df.drop(['stab','stabf'], axis=1))
y = np.array(df['stabf'])


# In[41]:


# define seed for reproducibility
seed = 42


# In[42]:


# split data into training and testing datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=seed)


# In[43]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# define scoring method
scoring = 'accuracy'

# Define models to train
names = ["Nearest Neighbors", "Decision tree","SVM Linear", "LogisticRegression","RandomForestClassifier"]

# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))

classifiers = [
    KNeighborsClassifier(n_neighbors = 9),
    DecisionTreeClassifier(criterion='entropy'),
    SVC(kernel = 'linear',gamma='auto'),
    LogisticRegression(),
    RandomForestClassifier()
        

]

models = zip(names, classifiers)

#evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('Test-- ',name,': ',accuracy_score(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))


# In[44]:


plt.plot(y_test,predictions)
plt.show()


# In[52]:


df.isnull().sum()


# In[45]:


results


# In[46]:


import pandas
import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Machine Learning Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[47]:


#REGRESSION


# In[48]:


df_r.info()
df_r


# In[49]:


#split our data
u = df_r.drop(["stabf","stab"], axis=1)
v = df_r["stab"]
u


# In[50]:


#Viewing the distribution of values for 'stab'
df_r['stab'].hist(bins=50)


# In[51]:


df_r.boxplot(column='stab')


# In[53]:


# split data into training and testing datasets
u_train, u_test, v_train, v_test = model_selection.train_test_split(u, v, test_size=0.25)
#random_state=seed


# In[54]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Define models to train
names_r = ["Linear Regression", "Decision tree Regression","Random Forest Regression"]
scoring=["r2","neg_mean_squared_error"]
Regression = [
    LinearRegression(),
    DecisionTreeRegressor(), 
    RandomForestRegressor()
]

models_r = zip(names_r, Regression)

# evaluate each model in turn
results_r = []
names_r = []

for name_r, model in models_r:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    for i in scoring:
        
        results = model_selection.cross_val_score(model, u, v, cv=kfold, scoring=i)
        print(str(i),results.mean(), results.std())
    #model.fit(u_train, v_train)
   # predictions = model.predict(u_test)
    #print(predictions)
    #print()
    # print(classification_report(y_test, predictions))


# In[ ]:





# In[ ]:




