

import os
import pyspark
import sys

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming.kafka import KafkaUtils

from pyspark.sql.types import StructType
from pyspark.sql import functions as f
from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF

import json
from pyspark.mllib.linalg import Vectors, SparseVector, DenseVector

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD


def get_hdfs_filepath(file_name, on_cloud=False):
    # path to folder containing this code
    prefix = '/data/spark/11_cs1_dataset_stream/'
    if on_cloud:
        bucket  = os.environ['BUCKET']
        file_path = bucket + prefix + file_name
    else:
        file_path = '/Users/val' + prefix + file_name

    return file_path

def parse_json_line(line):
    label = (json.loads(line))['label']
    feat_vec = (json.loads(line))['features']
    vec = Vectors.sparse(feat_vec['size'], feat_vec['indices'], feat_vec['values'])
    return LabeledPoint(label, vec)




if __name__ == '__main__':
    sc = SparkContext("local[2]", "Streaming_Linear_Regression")
    spark = SparkSession(sc)



    cwd = os.getcwd()
    # screen subfolders in working directory for new csv files
    userSchema = StructType().add("spam", "string").add("message", "string")
    df = spark \
      .readStream \
      .option("sep", "\t") \
      .schema(userSchema)  \
      .csv(cwd+"/"+"module11_cs1/train_data_csv")     # Equivalent to format("csv").load("/path/to/directory")
    df_test = spark \
      .readStream \
      .option("sep", "\t") \
      .schema(userSchema)  \
      .csv(cwd+"/"+"module11_cs1/test_data_csv")     # Equivalent to format("csv").load("/path/to/directory")


    ############################################
    # convert label: spam = 1, ham = 0
    df = df.withColumn("label", f.when(f.col('spam') == "spam", 1).otherwise(0))
    df_test = df_test.withColumn("label", f.when(f.col('spam') == "spam", 1).otherwise(0))

    # Extract words
    tokenizer = Tokenizer().setInputCol("message").setOutputCol("words")
    # Remove custom stopwords
    stopwords = StopWordsRemover().getStopWords() + ["-"]
    remover = StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
    # create features
    hashingTF = HashingTF(numFeatures=10, inputCol="words", outputCol="features")
    pipeline = Pipeline().setStages([tokenizer, remover, hashingTF])

    # transform train and test streams
    featured = pipeline.fit(df).transform(df)
    featured_test = pipeline.fit(df_test).transform(df_test)

    ###########################################
    ssc = StreamingContext(sc, 1)
    # read Dstream from json files in monitored dir for training
    trainingData = ssc.textFileStream(get_hdfs_filepath(file_name="train_stream_json/")).map(parse_json_line)
    trainingData.pprint()

    # read Dstream from json files in monitored dir for prediction
    testData = ssc.textFileStream(get_hdfs_filepath(file_name="test_stream_json/")).map(parse_json_line)
    testData.pprint()

    numFeatures = 10

    # initialize a StreamingLinearRegression model
    model = StreamingLinearRegressionWithSGD()
    model.setInitialWeights([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0])
    # train the model on training Dstream
    model.trainOn(trainingData)

    # print predictions for test Dstream to the console
    model.predictOnValues(testData.map(lambda lp: (lp.label, lp.features))).pprint()

    ssc.start()
    ##########################################
    # save 
    featured['label','features'].writeStream \
        .format("json") \
        .option("path", get_hdfs_filepath(file_name="train_stream_json")) \
        .option("checkpointLocation", get_hdfs_filepath(file_name="chkpnt1"))\
        .start()

    featured_test['label','features'].writeStream \
        .format("json") \
        .option("path", get_hdfs_filepath(file_name="test_stream_json/")) \
        .option("checkpointLocation", get_hdfs_filepath(file_name="chkpnt2"))\
        .start()


    #ssc.checkpoint("/Users/val/Documents/code/spark/m11_to_Upload/myStream")
    # Create a DStream that will connect to hostname:port, like localhost:9999
    #lines=ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    # sudo apt-get install netcat/ncat/nc
    #print(int(sys.argv[2]))

    ssc.awaitTermination()
    ssc.stop()
