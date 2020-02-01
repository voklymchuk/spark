
import os
import pyspark
import sys

from pyspark import SparkConf
from pyspark.streaming import StreamingContext

from pyspark.streaming.kafka import KafkaUtils

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.sql.types import StructType
from pyspark.sql import functions as f

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import OneHotEncoder, StringIndexer

from pyspark.sql import SQLContext

import pandas as pd
import numpy as np
from pyspark.ml.linalg import Vectors, SparseVector, DenseVector


from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel, RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql.utils import AnalysisException, IllegalArgumentException

import re

import time


def sendPartition(df, epoch_id):
    global dataset_df

    if not len(df.head(1)) == 0:
        dataset_df = df.union(dataset_df)

def train(rdd):
    global dataset_df
    global model
    global prev_length
    global evaluate
    global crossval_full

    if dataset_df.count() > prev_length:
        prev_length = dataset_df.count()

        if evaluate == True:
            # Split to train and test
            (trainingData, testData) = dataset_df.randomSplit([0.7, 0.3], seed=0)
        else:
            trainingData = dataset_df

        my_train_df = trainingData

        print("\n\nStarting to fit a model on " + str(my_train_df.count()) +" records")

        # smote is False, so crossvalidating on full_pipeline
        model = crossval_full.fit(my_train_df)
        print("Model fit compleeted\n")
        # parameters of the best model
        #print(model.getEstimatorParamMaps()[np.argmax(model.avgMetrics)])

        if evaluate == True:
            predictions = model.transform(testData)
            evaluator = BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction").setMetricName("areaUnderROC")
            accuracy = evaluator.evaluate(predictions)
            print("Evalluated on "+ str(predictions.count()) +" records")
            print ("Accuracy", accuracy)



def predict(df, epoch_id):
    global model

    if not model == None:
        print("\n\nPredictions:")
        predictions = model.transform(df)['spam', 'message','label', 'prediction', 'probability']
        # print predictions to the console
        predictions.show()
    else:
        print("\n\nModel has not seen training data yet, therefore - no model exists")



if __name__ == '__main__':

    if len(sys.argv)>1:
        train_csv_dir = sys.argv[1]
    else:
        train_csv_dir ="module11_cs1/train_data_csv"

    if len(sys.argv)>2:
        test_csv_dir = sys.argv[2]
    else:
        test_csv_dir = "module11_cs1/test_data_csv"

    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.3 pyspark-shell'
    sc = SparkContext("local[2]", "Batch_Model_on_Stream_Data")
    spark = SparkSession(sc)
    spark.sparkContext.setLogLevel("OFF")

    ############################################################
    # convert to binary label
    indexer = StringIndexer().setInputCol("spam").setOutputCol("label")
    # Extract words
    tokenizer = Tokenizer().setInputCol("message").setOutputCol("words")
    # Remove custom stopwords
    stopwords = StopWordsRemover().getStopWords() + ["-"]
    remover = StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
    # create features
    hashingTF = HashingTF(numFeatures=10, inputCol="filtered", outputCol="features")
    rf = RandomForestClassifier().setFeaturesCol("features").setNumTrees(10)
    #dt = DecisionTreeClassifier()
    lr = LogisticRegression(maxIter=10)
    full_pipeline = Pipeline().setStages([ indexer, tokenizer, remover, hashingTF, lr])
    ############################################################

    paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.15, 0.01]) \
    .build()
    evaluator=BinaryClassificationEvaluator()
    numFolds=2
    crossval_full = CrossValidator(estimator=full_pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=numFolds)  # use 3+ folds in practice
    ############################################################

    # schema for raw csv files
    userSchema = StructType().add("spam", "string").add("message", "string")

    sqlContext = SQLContext(sc)
    # create an empty datframe
    dataset_df = sqlContext.createDataFrame(sc.emptyRDD(), userSchema)
    # or populate initial dataframe from a local csv file
    #dataset_df = sc.textFile("gs://drive3/data/spark/8_cs1_dataset/SMSSpamCollection").map(lambda line: re.split('\t', line)).toDF(["spam", "message"])
    #dataset_df = feature_pipeline.fit(dataset_df).transform(dataset_df)

    model = None

    prev_length = 0
    # whether to split dataset into train and evaluate before training
    evaluate = True
    # duration of training a model on whole batch dataset
    train_duration = 20  # train a model every n seconds
    ############################################################
    # append each batch of trainging stream to dataset_df
    # as part of structured streaming
    df = spark \
        .readStream \
        .option("sep", "\t") \
        .schema(userSchema)  \
        .csv(train_csv_dir) \
        .writeStream.foreachBatch(sendPartition).start()
    # train a new model every n seconds as part of spark streaming
    ssc = StreamingContext(sc, train_duration)
    # generate a helper-stream
    helper_stream = ssc.queueStream([dataset_df.limit(1).rdd], oneAtATime=True, default=dataset_df.limit(1).rdd)
    # train a model
    helper_stream.foreachRDD(train)

    ssc.start()
    # wait for training to finish on first batch
    time.sleep(train_duration*2)
    # print predictions for each batch of testing stream to console
    df_test = spark \
        .readStream \
        .option("sep", "\t") \
        .schema(userSchema)  \
        .csv(test_csv_dir) \
        .writeStream.foreachBatch(predict).start()

    ssc.awaitTermination()  # Wait for the computation to terminate

    ssc.stop()
