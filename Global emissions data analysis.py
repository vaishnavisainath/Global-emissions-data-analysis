#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as po
import plotly.graph_objs as go
import warnings


# In[3]:


df=pd.read_csv('emissions_by_country.csv')
df


# In[4]:


df=df.dropna() 


# In[5]:


df.drop_duplicates(inplace=True)


# In[6]:


df.describe()
df.head()


# In[7]:


df['Country'] = df['Country'].astype("category").cat.codes
df['ISO 3166-1 alpha-3'] = df['ISO 3166-1 alpha-3'].astype("category").cat.codes
#df['cut'] = df['cut'].astype("category").cat.codes


# In[8]:


df


# In[9]:


plt.hist(df['Country'])
plt.title(" Country Distribution in country dataset")
plt.ylabel("Per Capita")
plt.xlabel("Country")
plt.show()


# In[10]:


plt.figure(figsize = (10,10))
sns.heatmap(df.corr(),cmap='coolwarm', annot=True)
plt.show()


# In[11]:


df.corr()


# In[12]:


#Step 4: Renaming columns for uniformity
df.rename(columns={'Per Capita': 'Per_Capita'}, inplace=True)


# In[13]:


df


# In[14]:


plotcountry = df.groupby('Country').Per_Capita.mean().reset_index()
plotdata = [
    go.Bar(
        x=plotcountry['Country'],
        y=plotcountry['Per_Capita'],
        width = [0.3, 0.3,0.3,0.3],
        marker=dict(
        color=[])
    )
]
plotlayout = go.Layout(
        xaxis={"title": "Country"},
        yaxis={"title": "Per_Capita"},
        title='Per capita per country',
        
    )
fig = go.Figure(data=plotdata, layout=plotlayout)
po.iplot(fig)


# In[15]:


sns.violinplot(x='Cement', y='Per_Capita', data=df)
plt.xlabel('Cement')
plt.ylabel('Per_Capita')
plt.show()


# In[16]:


sns.jointplot(x='Cement',y='Per_Capita',data=df,kind='scatter')


# In[17]:


df.dtypes


# In[18]:


df=(df-df.mean())/df.std()
df=(df-df.min())/(df.max()-df.min())


# In[19]:


y = df['Per_Capita']
X = df.drop(columns=['Per_Capita'])


# In[20]:


ratio = 0.20
total_rows = df.shape[0]
test_size = int(total_rows*ratio)


# In[21]:


X_test = X[0:test_size]
X_train = X[test_size:]
y_test = y[0:test_size]
y_train = y[test_size:]


# In[22]:


ratio = 0.20
total_rows = df.shape[0]
test_size = int(total_rows*ratio)


# In[23]:


X_train.shape


# In[24]:


X_test.shape


# In[27]:


y_train.shape


# In[28]:


trans = np.transpose(X_train)
matmul= np.matmul(trans, X_train)
m = matmul + 0.8*np.identity(X_train.shape[1])
c = np.linalg.inv(m)
b = np.matmul(trans, y_train)
weight = np.matmul(c, b)


# In[29]:


y_pred = np.matmul(X_test, weight)


# In[30]:


mse = np.mean((y_test - y_pred)**2)
print(mse)


# In[31]:


plt.scatter(y_test, y_test, color='blue', label='Actual test data')
plt.plot(y_test, y_pred, color='pink', label='Predictions')
plt.legend()


# In[ ]:


#References
https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe
https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/

