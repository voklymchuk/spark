import sys
import os
import requests
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

import matplotlib.pyplot as plt
import numpy as np

import datetime as dt
import dateutil.parser
import pytz

from pyspark.sql.functions import udf, regexp_extract, col, udf, expr, isnan, when, count
from pyspark.sql.types import StringType, IntegerType, TimestampType
from urllib.parse import urlparse
from pyspark.sql import Row


def to_utc_timestamp(string):
    zone_idx = string.find('-') if string.find('-') >= 0 else string.find('+')
    timezone_str = string[zone_idx:]
    date_time_str = string[:zone_idx-1]
    date_time_obj = dt.datetime.strptime(date_time_str, '%d/%b/%Y:%H:%M:%S')
    dt2 = dateutil.parser.parse(str(date_time_obj)+timezone_str)
    return dt2.astimezone(pytz.timezone("UTC"))


def set_pandas_options() -> None:
    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 100
    pd.options.display.max_colwidth = 120
    pd.options.display.width = 140
    # pd.options.display.precision = 2  # set as needed


def get_hdfs_filepath(file_name, on_cloud=True):
    # path to folder containing this code
    prefix = '/data/spark/6_cs2_dataset/'
    if on_cloud:
        bucket  = os.environ['BUCKET']
        file_path = bucket + prefix + file_name
    else:
        file_path = '/Users/val' + prefix + file_name

    return file_path

def get_host(url):
    return urlparse(url).netloc

def main():

    set_pandas_options()
    app_name="Case Study 2"

    conf = SparkConf().setAppName(app_name)
    sc = SparkContext(conf = conf)
    spark = SparkSession(sc)

    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")

    # 1 Load file as a text file in spark
    LOG = get_hdfs_filepath('access.log')
    CLEAN_LOG = get_hdfs_filepath('access.clean.log')
    # read text file
    row = Row("line")
    log_txt_df=sc.textFile(LOG).filter(lambda line: line != '')
    log_txt_df = log_txt_df.map(row).toDF()
    # Convert strings to columns
    udf1 = udf(to_utc_timestamp, TimestampType())
    from pyspark.sql.functions import col
    result = log_txt_df.select(
        regexp_extract(col('line'), r'http:\S+', 0).alias('url'),
        regexp_extract(col('line'), r'\d+\.\d+\.\d+\.\d+\s', 0).alias("ip"),
        regexp_extract(col('line'), r'\d{2}.\w{3}.\d{4}:\d{2}:\d{2}:\d{2}\s.\d{4}', 0).alias("datetime"),
        regexp_extract(col('line'), r"(\sGET\s|\sPOST\s|\sPUT\s|\sHEAD\s)", 0).alias("request"),
        regexp_extract(col('line'), r"\s\S+\sHTTP/\d", 0).alias("path"),
        regexp_extract(col('line'), r'HTTP/\S+\s\d{3}', 0).alias("response"),
        regexp_extract(col('line'), r'\s\d{3}\s\d+\s', 0).alias("bytes")
     ).select(
        'url', 'ip','datetime', udf1('datetime').alias('utc_timestamp'),
        expr("substring(request, 2, length(request)-2)").alias("request"),# same as str[1:-1]
        expr("substring(path, 2, length(path)-8)").alias("path"),# same as str[1:-7]
        expr("substring(response, -3, 3)").alias("response").cast(IntegerType()),# same as str[-3:]
        expr("substring(bytes, 6, length(bytes)-6)").alias("bytes").cast(IntegerType()),# same as str[5:-1]
    )
    df=result
    zz = df.limit(10).toPandas()
    LOGGER.info("\n\n1.\tLoad data into Spark DataFrame\tDone!\n\n{}\n".format(zz))

    # 2 Find out how many 404 HTTP codes are in access logs
    count = df.filter("response = '404'").count()
    LOGGER.info("\n\n2.\tFind out how many 404 HTTP codes are in access logs\tDone!\n\n{}\n\n".format(count))


    # 3 Find out which URLs are broken
    zz = df.where('response!=200').groupBy('url').count().orderBy('count', ascending =0).limit(10).toPandas()
    LOGGER.info("\n\n3.\tFind out which URLs are broken\tDone!\n\n{}\n".format(zz))

    # 4 Verify there are no null columns in the original dataset
    from pyspark.sql.functions import isnan, when, count, col
    zz = df.select([count(when( col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()
    LOGGER.info("\n\n4.\tVerify there are no null columns in the original dataset\tDone!\n\n{}\n".format(zz))

    # 5 Replace null values with constants such as 0
    df = df.na.fill(0)
    zz = df.select([count(when( col(c).isNull(), c)).alias(c) for c in df.columns])
    zz =zz.toPandas()
    LOGGER.info("\n\n5.\tReplace null values with constants such as 0\tDone!\n\n{}\n".format(zz))

    # 6 Parse timestamp to readable date
    zz = df.select('datetime','utc_timestamp').limit(10).toPandas()
    LOGGER.info("\n\n6.\tParse timestamp to readable date\tDone!\n\n{}\n".format(zz))

    # 7 Describe which HTTP status values appear in data and how many
    zz = df.groupBy('response').count().orderBy('count', ascending = False).limit(10).toPandas()
    LOGGER.info("\n\n7.\tDescribe which HTTP status values appear in data and how many\tDone!\n\n{}\n".format(zz))

    # 8 Display as chart the above stat in chart in Zeppelin notebook
    stat = df.groupBy('response').count().orderBy('count', ascending = False)
    pddf = stat.toPandas()
    responses = pddf.iloc[:,0].values.astype(str)
    counts = pddf.iloc[:,1].values
    plt.rcdefaults()
    y_pos = np.arange(len(responses))
    plt.bar(y_pos, counts, align='center', alpha=0.5, log = True)
    plt.xticks(y_pos,responses)
    plt.ylabel('Counts')
    plt.xlabel('Codes')
    plt.title('Log counts per status code')
    plt.show()
    LOGGER.info("\n\n8.\tDisplay as chart the above stat in chart in Zeppelin notebook\tDone!\n\n")

    # 9 How many unique hosts are there in the entire log and their average request
    get_host_udf = udf(get_host, StringType())
    zz = df.select( get_host_udf(df.url).alias('host')).groupBy('host').count().orderBy('count', ascending = 0).limit(10).toPandas()
    LOGGER.info("\n\n9.\tHow many unique hosts are there in the entire log and their average request\tDone!\n\n{}\n".format(zz))

    LOGGER.info("\n\n10.\tCreate a spark-submit application for the same and print the findings in the log\tDone!\n\n")


if __name__ == "__main__":
    main()
