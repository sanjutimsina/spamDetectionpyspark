#!/usr/bin/env python
# coding: utf-8

# In[57]:


from pyspark.sql import SparkSession

fileName = 'file:///Users/sanjutimsina/Desktop/SMSSpamCollection'
spark = SparkSession.builder.appName('HW4').getOrCreate()
rawData = spark.read.option("header", "false").option("delimiter", "\t").csv(fileName) #load file


# In[58]:


rawData = rawData.toDF("label", "sentences")


# In[59]:


rawData.rdd.map(lambda x: x.label).countByValue()


# In[60]:


#tokenizer to transform sentences to words
from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol="sentences", outputCol="words")


# In[61]:


wordsData = tokenizer.transform(rawData)


# In[62]:


wordsData.show(5)


# In[63]:


#tf
from pyspark.ml.feature import HashingTF

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
featurizedData = hashingTF.transform(wordsData)


# In[64]:


#idf
from pyspark.ml.feature import  IDF

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)


# In[65]:


rescaledData.show(5)


# In[66]:


#stringindexer to transform label to indexedLabel
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
indexedData = indexer.fit(rescaledData).transform(rescaledData)


# In[67]:


#add id column 
from pyspark.sql.functions import monotonically_increasing_id
indexedData = indexedData.withColumn("id", monotonically_increasing_id())


# In[68]:


indexedData.show(5)


# In[69]:


from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(labelCol='indexedLabel', featuresCol='features')


# In[70]:


#evaluator to evaluate data
from pyspark.ml.evaluation import BinaryClassificationEvaluator

binaryEvaluator = BinaryClassificationEvaluator(labelCol='indexedLabel', rawPredictionCol='prediction',metricName='areaUnderROC')


# In[71]:


#generate splits for cross validation
splits = indexedData.randomSplit([0.2,0.2,0.2,0.2,0.2])


# In[72]:


TotalAccuracy = 0

for i in range(5):
   
    testIndex = splits[i].select('id').collect() #get test index for each fold
    rdd = sc.parallelize(testIndex)
    test_rdd = rdd.flatMap(lambda x: x).collect()
    test_Data = indexedData.filter(indexedData.id.isin(test_rdd)) #get test data for each fold
    train_Data = indexedData.filter(~indexedData.id.isin(test_rdd)) #get train data for each model
    model = nb.fit(train_Data)      #fit train data to model
    transformed_data = model.transform(test_Data) # evaluate test data
    accuracy = binaryEvaluator.evaluate(transformed_data) # get accuracy for test data
    print(binaryEvaluator.getMetricName(), 'accuracy:',accuracy)
    TotalAccuracy = TotalAccuracy+accuracy

averageAccuracy = TotalAccuracy/5  # get average accuracy
print(averageAccuracy) 



