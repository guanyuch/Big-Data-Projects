
# coding: utf-8

# In[1]:


import findspark
findspark.init('C:/spark-2.2.0-bin-hadoop2.6')


# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[3]:


#Load book info
book_info = spark.read.csv('BX-Books.csv',header=True, inferSchema = True)
#Schema
book_info.printSchema()


# In[4]:


#Load user info
user_info = spark.read.csv('BX-Users.csv',header=True, inferSchema = True)
#Schema
user_info.printSchema()


# In[5]:


from pyspark.sql.types import IntegerType
user_info=user_info.withColumn("Age", user_info["Age"].cast(IntegerType()))
book_info=book_info.withColumn("Year-Of-Publication", book_info["Year-Of-Publication"].cast(IntegerType()))


# In[6]:


#Load books rating scores
book_rating = spark.read.csv('BX-Book-Ratings.csv',header=True, inferSchema = True)
#Schema
book_rating.printSchema()


# In[7]:


#Merge three datasets
#Merge book_rating and book_info by same ISBN
book_rating_join_book_info= book_rating.join(book_info,on=['ISBN'])


# In[8]:


#Merge book_rating_join_book_info and user_info by same user-id
#Final dataset
final_dataset= user_info.join(book_rating_join_book_info, on=['User-ID'])
final_dataset.na.drop()


# In[9]:


final_model_dataset=final_dataset.drop('User-ID', 'ISBN', 'Image-URL-S','Image-URL-M','Image-URL-L')


# In[10]:


#Encoding categorical features to numbers
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder,StringIndexer,IndexToString
from pyspark.ml import Pipeline


# In[11]:


author_indexer = StringIndexer(inputCol = 'Book-Author', outputCol = 'authorIndex').setHandleInvalid("skip")
author_encoder = OneHotEncoder(inputCol = 'authorIndex', outputCol = 'authorVector')

title_indexer = StringIndexer(inputCol = 'Book-Title', outputCol = 'titleIndex').setHandleInvalid("skip")
title_encoder = OneHotEncoder(inputCol = 'titleIndex', outputCol = 'titleVector')

publisher_indexer = StringIndexer(inputCol = 'Publisher', outputCol = 'publisherIndex').setHandleInvalid("skip")
publisher_encoder = OneHotEncoder(inputCol = 'publisherIndex', outputCol = 'publisherVector')

city_indexer = StringIndexer(inputCol = 'city', outputCol = 'cityIndex').setHandleInvalid("skip")
city_encoder = OneHotEncoder(inputCol = 'cityIndex', outputCol = 'cityVector')

state_indexer = StringIndexer(inputCol = 'state', outputCol = 'stateIndex').setHandleInvalid("skip")
state_encoder = OneHotEncoder(inputCol = 'stateIndex', outputCol = 'stateVector')

country_indexer = StringIndexer(inputCol = 'country', outputCol = 'countryIndex').setHandleInvalid("skip")
country_encoder = OneHotEncoder(inputCol = 'countryIndex', outputCol = 'countryVector')


# In[12]:


#Rename and delete
final_model_dataset=final_model_dataset.withColumnRenamed('Year-Of-Publication','year')
final_model_dataset=final_model_dataset.withColumnRenamed('Book-Rating','rating')
final_model_dataset=final_model_dataset.na.drop()


# In[13]:


final_model_dataset.printSchema()


# In[14]:


assembler = VectorAssembler(inputCols = ['authorVector','titleVector', 'publisherVector','cityVector', 'stateVector',  'year','Age'], outputCol = 'features')
labelIndexer = StringIndexer(inputCol="rating", outputCol="label")


# ##### Create demo dataset using 20% of full dataset (Hard to deal with large amount of data)

# In[15]:


fData, ffData = final_model_dataset.randomSplit([0.8,0.2], seed = 1)


# In[16]:


trainingData, testData = ffData.randomSplit([0.8,0.2],seed = 1)


# In[17]:


from pyspark.ml.regression import GBTRegressor


# In[18]:


gbt= GBTRegressor(maxIter=10)


# In[19]:


pipeline = Pipeline(stages=[labelIndexer,author_indexer,publisher_indexer,title_indexer,city_indexer,state_indexer,author_encoder, title_encoder,publisher_encoder, city_encoder, state_encoder, assembler,gbt])


# In[20]:


from pyspark.ml.regression import GBTRegressor


# In[21]:


get_ipython().run_cell_magic('time', '', 'model = pipeline.fit(trainingData)')


# In[22]:


#Make predictions on testData
predictions = model.transform(testData)


# In[23]:


from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")


# In[24]:


RMSE = evaluator.evaluate(predictions)


# In[25]:


print("Model RMSE = " + str(RMSE))

