#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy
from scipy import stats

marks=[50,40,90,60,50,40,80,75,95,30]

# mean , median, mode

print(numpy.mean(marks))
print(numpy.median(marks))
print(stats.mode(marks))


# In[13]:


pip install sciPy


# In[17]:


import numpy #Std deviation

marks=[50,40,90,60,50,40,80,75,95,30]
numpy.std(marks)


# In[22]:


import numpy #variance

marks=[50,40,90,60,50,40,80,75,95,30]
print(numpy.var(marks))
print(numpy.percentile(marks,75)) # dataset,value


# In[31]:


# Create an array containing 50 random plots between 0 to 5

import numpy
import matplotlib.pyplot as plt

x=numpy.random.uniform(0,5,50)

print(x)
plt.hist(x)


# In[33]:


import numpy
import matplotlib.pyplot as plt

x=numpy.random.uniform(0,5,10000)

plt.hist(x,100)


# In[36]:


import numpy
import matplotlib.pyplot as plt

x=numpy.random.normal(0,5,10000)

plt.hist(x,100)  #parameters are mean and std deviations.


# In[43]:


import numpy

marks=[50,40,90,60,50],[40,80,75,95,30]
np.shape(marks)


# In[50]:


#ndim

import numpy

arr = np.array([1, 2, 3, 4], ndmin=5)
np.shape(arr)


# In[89]:


import numpy

arr=np.array([[2,3,4,5],[5,6,2,1]])
arr1=np.array([[[2,3,4,5],[5,6,2,1],[8,9,0,1]]])
arr2=np.array([[[[2,3,],[5,6,],[8,9,],[9,0]]]])

ar=np.array([5,6,3,2,8,0,1,2,3,4,5,6])

newar = ar.reshape(3,4)
newar1 = ar.reshape(2,2,3)

print(arr.shape)
print(arr1.shape)
print(arr2.shape)
print(newar)
print(newar1)


# In[1]:


pip install pandas


# In[10]:


import pandas as pd #PANDAS DATAFRAME

data={
    'Cars':["Audi","Benz","Ford"],
    'Passings':[4,9,5]
}
dataset=pd.DataFrame(data)             #Dataframe is having multiple columns and rows.
print(dataset)
print(data)


# In[11]:


print(pd.__version__)


# In[14]:


import pandas as pd                 #Single column data is the series
a=[1,6,7]
dataset=pd.Series(a)
print(dataset)

print(dataset[0])


# In[17]:


import pandas as pd                 #Single column data is the series
a=[1,6,7]
dataset=pd.Series(a,index=["A","B","C"])
print(dataset)
print(dataset["A"])


# In[25]:


#Dataframe Creation -2D

import pandas as pd
data={
    "Calories":[420,380,390],
    "Duration":[50,40,45]    
}
df=pd.DataFrame(data)
print(df)
print(df.loc[1])                       #Location of the values 


# In[28]:


#Dataframe Creation -2D

import pandas as pd
data={
    "Calories":[420,380,390],
    "Duration":[50,40,45]    
}
df=pd.DataFrame(data,index=["Day 1","Day 2","Day 3"])
print(df)                    #Location of the values 


# In[29]:


#Dataframe Creation -2D

import pandas as pd
data={
    "Calories":[420,380,390],
    "Duration":[50,40,45]    
}
df=pd.DataFrame(data)
print(df)      
print(df.loc[0:1])   


# In[30]:


import pandas as pd

df=pd.read_csv('data.csv')

print(df.to_string())


# In[32]:


import pandas as pd

df=pd.read_csv('data.csv')

print(pd.options.display.max_rows)


# In[33]:


import pandas as pd

df=pd.read_csv('data.csv')
print(df)   #Head and tail


# In[38]:


import pandas as pd

df=pd.read_csv('data.csv')
print(df.head(10))   #Head and tail


# In[39]:


import pandas as pd

df=pd.read_csv('data.csv')
print(df.tail(10))   #tail


# In[40]:


import pandas as pd

df=pd.read_csv('data.csv')
print(df.info())   #info means infromation about the datasets


# In[42]:


import pandas as pd
df=pd.read_csv('dataset.csv')
print(df.to_string())


# In[43]:


print(df.info())


# In[46]:


new_df=df.dropna()
print(new_df.info())


# In[49]:


# Preprocessing the data

import pandas as pd
df=pd.read_csv('dataset.csv')

print(df.info())

print(df.to_string())

new_df=df.dropna()

print(new_df.info())

print(new_df.to_string())


# In[48]:


# Delete empty values
df.dropna(inplace=True)
print(df.to_string())


# In[51]:


df.fillna(130,inplace=True)
print(df.to_string())


# In[53]:


df["Calories"].fillna(130,inplace=True)
print(df.to_string())


# In[ ]:




