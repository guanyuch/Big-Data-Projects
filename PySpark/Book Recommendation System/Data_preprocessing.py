
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:

# Add the path of original_data please
user = pd.read_csv('BX-Users.csv', header = 0, sep = '";"', encoding = 'latin-1')
user.head()


# In[3]:


book = pd.read_csv('BX-Books.csv', header = 0, sep = '";"', encoding = 'latin-1')
book.tail()


# In[4]:


rating = pd.read_csv('BX-Book-Ratings.csv', header = 0, sep = '";"', encoding = 'latin-1')
rating.head()


# In[5]:


# Format the column name and records
user.columns = ['User-ID', 'Location', 'Age']
book.columns = book.columns = ['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher','Image-URL-S','Image-URL-M','Image-URL-L']
rating.columns = ['User-ID','ISBN','Book-Rating']


# In[6]:


user['User-ID'] = user['User-ID'].str.lstrip('"')
user['Age'] = user['Age'].str.rstrip('"')
book['ISBN'] = book['ISBN'].str.lstrip('"')
book['Image-URL-L'] = book['Image-URL-L'].str.rstrip('"')
rating['User-ID'] = rating['User-ID'].str.lstrip('"')
rating['Book-Rating'] = rating['Book-Rating'].str.rstrip('"')


# In[7]:


# Check missing values in each table
user.isnull().sum()


# In[8]:


book.isnull().sum()


# In[9]:


rating.isnull().sum()


# In[10]:


user.drop(user[user['Location'].isnull()].index, inplace = True)
user.drop(user[user['Age'].isnull()].index, inplace = True)
book.drop(book[book['Publisher'].isnull()].index, inplace = True)


# In[14]:


rating.tail()


# In[15]:


import seaborn as sns
get_ipython().magic('pylab inline')


# In[17]:


sns.violinplot(rating['Book-Rating']);

plt.ylabel('Amount')
plt.xlabel('Rating')
plt.title('Rating Distribution');


# In[18]:


rating = rating[rating['Book-Rating'] != '0']


# In[20]:


rating.head()


# In[19]:


# Get an overview of our data
print("The number of unique users in user table is "+str(len(user['User-ID'].unique())))
print("The number of unique books in book table is "+str(len(book['ISBN'].unique())))
print("The number of unique users in rating table is "+str(len(rating['User-ID'].unique())))
print("The number of unique books in rating table is "+str(len(rating['ISBN'].unique())))


# In[21]:


# Drop books that do not exist in both tables， same with users later
book = book.loc[book['ISBN'].isin(rating['ISBN'].unique())]
book.reset_index(drop=True,inplace=True)
book.tail()


# In[22]:


rating = rating.loc[rating['ISBN'].isin(book['ISBN'].unique())]
rating.reset_index(drop=True,inplace=True)
rating.tail()


# In[23]:


user = user.loc[user['User-ID'].isin(rating['User-ID'].unique())]
user.reset_index(drop=True,inplace=True)
user.tail()


# In[24]:


# Split Location to city, state and country
user['Location'] = user['Location'].str.split(',')
user.head()


# In[25]:


user['city'] = user['Location'].str[0]
user['state'] = user['Location'].str[1]
user['country'] = user['Location'].str[2]
user.drop('Location',axis = 1, inplace = True)
user.head()


# In[26]:


user['state'] = user['state'].str.strip(' ')
user['country'] = user['country'].str.strip(' ')
user = user[user.city != '']
user.head()


# In[27]:


# Focus on USA users only
user = user[user['country'] == 'usa']
user.head()


# In[28]:


# Drop records with city not starts with letters
import re
user = user[user['city'].map(lambda s: bool(re.match('[a-zA-Z]', s)))]


# In[29]:


rating = rating.loc[rating['User-ID'].isin(user['User-ID'].unique())]
rating.set_index('User-ID',inplace = True)
rating.tail()


# In[30]:


book.head()


# In[31]:


book = book.loc[book['ISBN'].isin(rating['ISBN'].unique())]


# In[32]:


book.set_index('ISBN', inplace = True)
book.head()


# In[33]:


user.to_csv('BX-Users.csv', index = True, header = True, encoding = 'utf-8')
book.to_csv('BX-Books.csv', index = True, header = True, encoding = 'utf-8')
rating.to_csv('BX-Book-Ratings.csv', index = True, header = True, encoding = 'utf-8')

