
# coding: utf-8

# In[3]:


import findspark
findspark.init('C:/spark1/spark-2.2.1-bin-hadoop2.7')
import pandas as pd
import pyspark
from pyspark import SparkContext 
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
spark=SparkSession.builder.getOrCreate()
sqlContext=SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)
from pyspark import SparkConf
sc = SparkContext.getOrCreate(SparkConf())
import os
from pyspark.sql.functions import col
from pyspark.mllib.recommendation import ALS
import math


# # 1. load and process data

# ## 1.1 load 'BX-Book-Ratings.csv'(clean dataset) and convert it into RDD as well as DataFrame

# In[4]:


#convert ISBN into int
ratings_raw_RDD = sc.textFile('C:/Users/Xin Gu/Desktop/BA courses/758B/project/newdataset/BX-Book-Ratings.csv')
ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
ratings_RDD = ratings_raw_RDD.filter(lambda line: line!=ratings_raw_data_header)    .map(lambda line: line.split(","))    .map(lambda tokens: (int(tokens[0]), abs(hash(tokens[1])) % (10 ** 8), int(tokens[2]))).cache()


# In[170]:


ratings_RDD.take(2)


# In[4]:


#ratings_RDD= ratings_RDD.filter(lambda x: x[2] != 0)


# In[9]:


#build booking rate as dataframe
from pyspark.sql.functions import col
df_rate=sqlContext.createDataFrame(ratings_RDD)
mapping = dict(zip(['_1', '_2', '_3'],['user','ISBN','rating']))
df_rate=df_rate.select([col(c).alias(mapping.get(c, c)) for c in df_rate.columns])
df_rate.show(3)


# In[10]:


#check if there is null value
from pyspark.sql.functions import isnan, when, count, col
df_rate.select([count(when(isnan(c), c)).alias(c) for c in df_rate.columns]).show()


# In[92]:


### Show the number of users  who rated books in the dataset is approximately 95513
print("Number of different users "+ str(df_rate.select('user').distinct().count()))


# In[11]:


# Show sample number of ratings per user
grouped_ratings=df_rate.repartition('user').groupBy('user').count()
grouped_ratings.orderBy('count',ascending=False).show(5)


# ## 1.2 load 'BX-Books.csv'(clean data) and convert it into RDD and DataFrame

# In[15]:


# convert ISBN into int
books_raw_RDD = sc.textFile('C:/Users/Xin Gu/Desktop/BA courses/758B/project/newdataset/BX-Books.csv')
books_raw_data_header = books_raw_RDD.take(1)[0]
books_RDD = books_raw_RDD.filter(lambda line: line!=books_raw_data_header)    .map(lambda line: line.split(","))    .map(lambda tokens: (abs(hash(tokens[0])) % (10 ** 8), tokens[1], tokens[2], tokens[3], tokens[4], tokens[5])).cache()
books_titles_RDD = books_RDD.map(lambda x: (int(x[0]), x[1], x[2], (x[3]), x[4], x[5])).cache()


# In[95]:


books_titles_RDD.take(3)


# In[17]:


# covert rdd into datafram
df_book=sqlContext.createDataFrame(books_titles_RDD)
mapping = dict(zip(['_1', '_2', '_3', '_4', '_5', '_6'],['ISBN','Title',
 'Author',
 'Year-Of-Publication',
 'Publisher',
 'URLM']))
df_book=df_book.select([col(c).alias(mapping.get(c, c)) for c in df_book.columns])


# In[18]:


df_book.show(5)


# ## 2. Build recommendation system based on ALS

# In[173]:


#split data into test(0.2)and training (0.8)
test, train = ratings_RDD.randomSplit(weights=[0.2, 0.8], seed=1)
test1= test.map(lambda token: (token[0], token[1]))


# In[174]:


test1.count()


# ### 2.1 train ALS model and tune parameter based on the RMSE using the whole test dataset

# In[104]:


# tune paramter:rank [16,20,24,28,32,36]
seed = 5
iterations = 10
regularization_parameter = 0.1
rank_list=[]
rmse_list=[]
ranks = [16,20,24,28,32,36]
errors = [0]*len(ranks)
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(train, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(test1).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds =test.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    rank_list.append(rank)
    rmse_list.append(error)
    print('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print('The best model was trained with rank %s' % best_rank)


# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.xlabel('Rank')
plt.ylabel('RMSE')
sns.pointplot(x=rank_list,y=rmse_list)


# In[118]:


#tune parameter regularization_parameter = [0.01,0.05,0.10,0.15,0.20,0.25]
seed = 5
iterations = 10
rank_list=[]
rmse_list=[]
regular_list=[]
r = [0.01,0.05,0.10,0.15,0.20,0.25]
errors = [0]*len(r)
err = 0
tolerance = 0.02
min_error = float('inf')
best_rank = -1
best_iteration = -1
for i in r:
    model = ALS.train(train, rank=32, seed=seed, iterations=iterations,
                      lambda_=i)
    predictions = model.predictAll(test1).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds =test.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    regular_list.append(i)
    rank_list.append(rank)
    rmse_list.append(error)
    print('For regularization_parameter %s the RMSE is %s' % (i, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print('The best model was trained with regularization_parameter %s' % i)


# In[119]:


plt.xlabel('Lamda')
plt.ylabel('RMSE')
sns.pointplot(x=regular_list,y=rmse_list)


# In[121]:


#tune parameter iterations = [8,10,12,14,16,18,20]
seed = 5
iterations = [8,10,12,14,16,18,20]
rank_list=[]
rmse_list=[]
regular_list=[]
errors = [0]*len(iterations)
err = 0
tolerance = 0.02
min_error = float('inf')
best_rank = -1
best_iteration = -1

for i in iterations :
    model = ALS.train(train, rank=32, seed=seed, iterations=i,
                      lambda_=0.25)
    predictions = model.predictAll(test1).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds =test.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    regular_list.append(i)
    rank_list.append(rank)
    rmse_list.append(error)
    print('For iterations %s the RMSE is %s' % (i, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print('The best model was trained with iterations  %s' % i)


# In[178]:


plt.xlabel('iterations')
plt.ylabel('RMSE')
sns.pointplot(x=regular_list,y=rmse_list)


# ### 2.2choose the best model with tunned parameters

# In[ ]:


# the lowest RMSE is when rank =32 and lambda 0.25,iterations 20
als= ALS.train(ratings_RDD, rank=32, seed=5, iterations=20,
                      lambda_=0.25,nonnegative=True)


# ### 2.3  caculate RMSE again based on same test subset with other models instead of the whole test dataset
# 
# Attention!!: After tunning best model, we split test subset into test2(0.2),train2(0.8) again, the purpose for spliting it again is to caculate RMSE on test2 which dataset other models use to test their models so that we can compare RMSE at the same level.

# In[138]:


# split test subset into test2(0.2),train2(0.8)
test2, train2= test.randomSplit(weights=[0.2, 0.8], seed=1)
test22= test2.map(lambda token: (token[0], token[1]))


# In[139]:


#tune parameter regularization_parameter = [0.01,0.05,0.10,0.15,0.20,0.25]
model= ALS.train(ratings_RDD, rank=32, seed=5, iterations=20,
                      lambda_=0.25,nonnegative=True)
predictions = model.predictAll(test22).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds =test2.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
print('the RMSE is %s' % (error))


# ### 2.4 save our model

# In[ ]:


from pyspark.mllib.recommendation import MatrixFactorizationModel
model_path = os.path.join('..', 'model_recommendation', 'book_ALs')
# Save and load model
bestmodel.save(sc, model_path)
samemodel = MatrixFactorizationModel.load(sc, model_path)


# ## 3. Model2: generate the  baseline model

# In[140]:


## extract train rdd with (user, ISBN, actual rating)
from pyspark.sql.functions import col
df_rate1=sqlContext.createDataFrame(train)
mapping = dict(zip(['_1', '_2', '_3'],['user','ISBN','rating']))
df_rate1=df_rate1.select([col(c).alias(mapping.get(c, c)) for c in df_rate1.columns])


# In[141]:


df_rate1.take(4)


# In[143]:


## take average value for each book (item-based)
avg_rating=df_rate1.groupBy('ISBN').agg({'rating': 'mean'})
avg_rating.show(5)


# In[144]:


#conver two rdd into pd.dataframe
df_rate1=df_rate1.toPandas()


# In[145]:


# generate avg_rating dataframe (ISBN, avgrating)
avg_rating1=avg_rating.toPandas()
avg_rating1.head()


# In[146]:


## extract test2.rdd and convert it into dataframe (user, ISBN, acutal rating)
test_schema=['user','ISBN','rating']
pd_test2=test2.toDF(schema=test_schema).toPandas()


# In[157]:


## create dataframe with the schema of (user, ISBN, rating, avgrating)
df = pd_test2.merge(avg_rating1, on='ISBN')
df.head()


# In[172]:


#caculate MSE and RMSE  
df['Diff']=(df.rating-df['avg(rating)'])**2
df_mean=df.groupby(by='ISBN')['Diff'].agg(['mean'])
df_mean.head()


# In[160]:


df_mean.shape


# In[161]:


df_mean[df_mean['mean']==0].count()


# ### get the RMSE of baseline

# In[162]:


df_mean['mean'].sum()/len(df_mean)


# !!!! lower than ALS model! we can do better than baseline~

# ## 4. Predict with  our recommendation system 

# In[123]:


li=[276747,276804,276811,276896,276928]


# In[124]:


x_list=[]
a=0
for user in li:
    try:
        ###  Determining what books user has already read and rated so that we can make a list of historical records.
        book_read = df_rate.filter(df_rate.user == user).alias('a').        join(df_book.alias('b'),col('a.ISBN') == col('b.ISBN'),'inner').select('a.user','a.rating','b.ISBN','b.Title')
        ### Determining what books user  has not already read and rated so that we can make new e recommendations
        book_notread = df_rate.filter(df_rate.user == user).alias('a').        join(df_book.alias('b'),col('a.ISBN') == col('b.ISBN'),'right').filter('a.user is null').select('b.ISBN','b.Title')
        #make a list of not reading books for user
        notread_RDD=book_notread.rdd
        # get structure (user, ISBN) pairs for not-reading books
        user_ISBN_notread_RDD=notread_RDD.map(lambda x: (user, x[0]))
        user_recommendations_RDD = als.predictAll(user_ISBN_notread_RDD)
        #get predicting list as a desceding order
        Top= spark.createDataFrame(user_recommendations_RDD, ('user', 'ISBN','pred_rating')).sort('pred_rating',ascending=False)
        # get top5 recommended books
        Top5=sc.parallelize(Top.take(5)).toDF()
        # create SQL to get the final result
        df_book.registerTempTable('book')
        Top5.registerTempTable('Top5')
        if a==0:
            x=sqlContext.sql('Select Top5.user,book.ISBN, Title, URLM from book, Top5 where book.ISBN=Top5.ISBN')
            r=book_read
        else:
            y=sqlContext.sql('Select Top5.user,book.ISBN, Title, URLM from book, Top5 where book.ISBN=Top5.ISBN')
            x=x.union(y)
            b=book_read
            r=r.union(b)
        a+=1
        print(str(user)+' done')
    except:
        print(str(user)+' error')


# In[125]:


#save the prediction result to csv
x.toPandas().to_csv('Prediction.csv',index=0,encoding = 'utf-8')


# In[126]:


#save the reading_history result to csv
r.toPandas().to_csv('Reading_History.csv',index=0,encoding = 'utf-8')

