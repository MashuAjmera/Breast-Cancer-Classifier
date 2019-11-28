#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score,confusion_matrix,f1_score
import statsmodels.formula.api as sm
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


df = pd.read_csv('wdbc.data')
df.drop('ID',axis = 1,inplace = True)


# In[20]:


df.head()


# In[21]:


le = LabelEncoder()
le.fit(df['Diagonsis'])
df['Diagonsis'] = le.transform(df['Diagonsis'])
df.head()


# In[22]:


X = df.iloc[:,1:]
y = df.iloc[:,0]


# In[23]:


l = ["Radius_mean","Texture_mean","Perimeter_mean","Area_mean","Area_se","Worst_Radius","Worst_Texture","Worst_Perimeter",'Worst_Area']

mms = MinMaxScaler()
mms.fit(X[l])
X[l] = mms.transform(X[l])

X.head()


# In[24]:


l = ["Malignant","Benign"]
m = y[y == 1].count()
b = y[y == 0].count()

plt.figure()
plt.bar(l[0],m)
plt.bar(l[1],b)
plt.xlabel("Category of Cancer")
plt.ylabel("Number of Cases")
plt.title("Count of Malignant and Benign Cancers")
plt.show()


# In[25]:


corr = X.corr()
plt.figure(figsize=(20,20))
ax = sns.heatmap(corr,annot = True)
plt.show()


# In[26]:


# All the data related to Perimeter (Perimeter_mean,Perimeter_se and Worst_Perimeter) are highly correlated to
# the respective values of area and radius.


# In[27]:


columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = X.columns[columns]

X = X[selected_columns]
selected_columns = list(selected_columns.values)
X.head()


# In[28]:


def p_threshold(X, y, sl, columns):
    numOfVars = len(columns)
    for i in range(0, numOfVars):
        regressor_OLS = sm.OLS(y, X).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numOfVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    t = columns[j]
                    X.drop(t,axis = 1,inplace = True)
                    columns.pop(j)
    print(regressor_OLS.summary())
    return X, columns

SL = 0.05
X, selected_columns = p_threshold(X, y, SL, selected_columns)


# In[29]:


plt.figure(figsize = (20, 20))
j = 0
for i in X.columns:
    plt.subplot(6, 3, j+1)
    j += 1
    sns.distplot(X[i][y==1], label = 'Malignant')
    sns.distplot(X[i][y==0], label = 'Benign')
    plt.legend(loc='best')
plt.suptitle('Breast Cancer Analysis')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


# In[30]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)


# In[31]:


svc = SVC(C = 1.0,kernel='rbf')
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)


# In[32]:


print("Accuracy of SVC:",round(accuracy_score(y_test,y_pred_svc)*100,2))
print("Precision of SVC:",round(precision_score(y_test,y_pred_svc)*100,2))
print("Recall of SVC:",round(recall_score(y_test,y_pred_svc)*100,2))
print("F- score of SVC:",round(f1_score(y_test,y_pred_svc)*100,2))


# In[33]:


plt.figure()
sns.heatmap(confusion_matrix(y_test,y_pred_svc),annot = True)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("Confusion Matrix of SVM Classifer")
plt.xticks([0.5,1.5],["Benign","Malignant"])
plt.yticks([0.5,1.5],["Benign","Malignant"])
plt.show()


# In[ ]:





# In[ ]:




