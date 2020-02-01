import sys
import os
import requests
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
import urllib.request
from numpy import array
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import nest_asyncio
nest_asyncio.apply()
import asyncio
from aiohttp import ClientSession
import datetime as dt
import dateutil.parser
import re
from datetime import datetime
from urllib.parse import urlparse
import progressbar

def set_pandas_options() -> None:
    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 100
    pd.options.display.max_colwidth = 120
    pd.options.display.width = 140
    # pd.options.display.precision = 2  # set as needed



def to_timestamp(string):
    timezone_str = string[string.find('+'):]
    date_time_str = string[:string.find('+')-1]
    date_time_obj = dt.datetime.strptime(date_time_str, '%d/%b/%Y:%H:%M:%S')
    dt2 = dateutil.parser.parse(str(date_time_obj)+timezone_str)
    return dt2

class Switcher(object):
    def __init__(self, string):
        self.string = string

    def indirect(self,i):
        method_name='number_'+str(i)
        method=getattr(self,method_name,lambda :'Invalid')
        return method()

    def number_0(self):
        return self.string
    def number_1(self):
        return self.string
    def number_2(self):
        return to_timestamp(self.string)
    def number_3(self):
        return self.string[1:-1]
    def number_4(self):
        return self.string[1:-7]
    def number_5(self):
        return int(self.string[-3:])
    def number_6(self):
        return int(self.string[5:-1])


def my_search(row):
    patterns =[
        r'http:\S+', #url
        r"\d+\.\d+\.\d+\.\d+\s", #ip
        r"\d{2}.\w{3}.\d{4}:\d{2}:\d{2}:\d{2}\s.\d{4}", #datetime
        r"(\sGET\s|\sPOST\s|\sPUT\s|\sHEAD\s)", #request type
        r"\s\S+\sHTTP/\d", #path
        r'HTTP/\S+\s\d{3}', #response code
        r'\s\d{3}\s\d+\s' # no of bytes
    ]
    results=[]
    for idx, pattern in enumerate(patterns):
        r = re.findall(pattern, row)
        if r:
            s=Switcher(r[0])
            results.append(s.indirect(idx))
            del s
        else:
            results.append('None') if idx!=6 and idx!=5 else results.append(0)
    return results

async def fetch(url, session):
    async with session.get(url, timeout = None) as response:
        return await response.read()

async def run(urls):
    tasks = []
    data = []
    async with ClientSession() as session:
        for url in urls:
            url_first = url if (not type(url) == list and not type(url) == tuple) or isinstance(url, str) else url[0]
            task = asyncio.ensure_future(fetch(url_first, session))
            tasks.append(task)
            data.append(url)

        responses = await asyncio.gather(*tasks,return_exceptions=True)
        return zip(responses, data)

def scan(FUTURES):
    loop = asyncio.get_event_loop()

    # create a list of future objects
    future = asyncio.ensure_future(FUTURES)
    responses = loop.run_until_complete(future)

    def to_tuple(row):
        tuple_el = (row,) if isinstance(row, str) else (row)
        return tuple_el


    #copmose a list of codes for each url
    code_list = (
        to_tuple(data)+('ERR',)  if isinstance(response, Exception) else to_tuple(data)+('OK',) for response, data in responses
    )

    return code_list




def get_hdfs_filepath(file_name, on_cloud=True):
    # path to folder containing this code
    bucket  = os.environ['BUCKET']
    
    prefix = '/data/spark/5_cs2_dataset/'
    if on_cloud:
        file_path = bucket + prefix + file_name
    else:
        file_path = '/Users/val' + prefix + file_name

    return file_path


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
    LOG_ = get_hdfs_filepath('access.log')
    CLEAN_LOG = get_hdfs_filepath('access.clean.log')
    # read text file
    text_file = sc.textFile(LOG_).filter(lambda row: row!='')
    split_rdd = text_file.map(lambda row: my_search(row))
    LOGGER.info("\n\n1.\tLoad file as a text file in spark\tDone!\n")

    # 2 Find out how many 404 HTTP codes are in access logs
    count = split_rdd.filter(lambda row: row[5]==404).count()
    LOGGER.info("\n\n2.\tFind out how many 404 HTTP codes are in access logs\tDone!\n\n{}\n\n".format(count))


    # 3 Find out which URLs are broken
    url_count_rdd =  split_rdd.map(lambda row: (row[0], 1)).reduceByKey(lambda x, y: x + y)
    start_time = time.time()

    zz_temp = url_count_rdd.mapPartitions(run)
    zz = zz_temp.mapPartitions(scan).filter(lambda row: row[-1]=='ERR').sortBy(lambda a: -int(a[1])).toDF(['url','count','result']).toPandas()
    LOGGER.info("\n\n3.\tFind out which URLs are broken\tDone!\n\n{}\n{}\n".format(zz,"--- %s seconds ---" % (time.time() - start_time)))

    # 4 Verify there are no null columns in the original dataset
    LOGGER.info("\n\n4.\tVerify there are no null columns in the original dataset\tDone!\n\n")

    # 5 Replace null values with constants such as 0
    LOGGER.info("\n\n5.\tReplace null values with constants such as 0\tDone!\n\n")

    # 6 Parse timestamp to readable date
    date_codes = split_rdd.map(lambda row: (row[2], row[5]))
    dates = date_codes.toDF(['readable_date']).toPandas().iloc[:,0]
    LOGGER.info("\n\n6.\tParse timestamp to readable date\tDone!\n\n{}\n".format(dates.head(10)))

    # 7 Describe which HTTP status values appear in data and how many
    codes = date_codes.map(lambda line: line[1])
    code_counts = codes.map(lambda code: (code,1)).reduceByKey(lambda x, y: x + y).sortBy(lambda a: -a[1])
    df = code_counts.toDF(['code','count']).toPandas()
    LOGGER.info("\n\n7.\tDescribe which HTTP status values appear in data and how many\tDone!\n\n{}\n".format(df))

    # 8 Display as chart the above stat in chart in Zeppelin notebook
    responses = df.iloc[:,0].values.astype(str)
    counts = df.iloc[:,1].values
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
    hosts = url_count_rdd.map(lambda row: (urlparse(row[0]).netloc, row[1]) )
    hosts_counts = hosts.reduceByKey(lambda x, y: x + y).sortBy(lambda a: -a[1])
    df = hosts_counts.toDF(['host','count']).toPandas()
    LOGGER.info("\n\n9.\tHow many unique hosts are there in the entire log and their average request\tDone!\n\n{}\n".format(df))

    LOGGER.info("\n\n10.\tCreate a spark-submit application for the same and print the findings in the log\tDone!\n\n")


if __name__ == "__main__":
    main()
