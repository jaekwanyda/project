#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("C:/Users/jaekw/Documents/카카오톡 받은 파일/temp1.csv",encoding="euc-kr")


# In[3]:


df


# In[57]:


df.columns


# In[4]:


df.iloc[:,3:]


# In[5]:


df=df.iloc[:,3:]


# In[6]:


corr=df.corr(method="pearson")


# In[7]:


corr


# In[67]:


import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib


# In[68]:


plt.figure(figsize=(16, 16))
sns.heatmap(corr, vmin=-1, vmax=1, annot=False, cmap='BrBG')


# In[69]:


df.isnull().sum()


# In[8]:


df.fillna(method='ffill',inplace=True)


# In[9]:


df.isnull().sum()


# In[72]:


df["평균기온(°C)"].plot()


# In[ ]:




