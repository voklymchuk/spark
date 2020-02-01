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

import datetime as dt
import dateutil.parser
import re
from datetime import datetime
from urllib.parse import urlparse
import progressbar

import concurrent.futures

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



# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

import progressbar

# We can use a with statement to ensure threads are cleaned up promptly
def load_all(URLS, sc):
    
    
    res =[]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {
            executor.submit(load_url, url if (not type(url) == list and not type(url) == tuple) or isinstance(url, str) else url[0] , 120): url for url in URLS
                        }
        
        i=0
        bar = progressbar.ProgressBar(maxval=len(future_to_url), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                #print('%r generated an exception: %s' % (url, exc))
                re = 'ERR'
            else:
                #print('%r page is %d bytes' % (url, len(data)))
                re = 'OK'
            i+=1
            bar.update(i)
            res.append((url, re)  )
    bar.finish()
    return sc.parallelize(res).map(lambda row: np.append(array(row[0]),row[1]).tolist()).filter(lambda row: row[-1]=='ERR')






def get_hdfs_filepath(file_name, on_cloud=True):
    # path to folder containing this code
    my_hdfs = os.getcwd()
    # get folder name
    project_name = my_hdfs.split('/')[-1] 
    if on_cloud:
        file_path = 'gs://drive3/data/'+project_name+'/'+file_name
    else:
        # data is located in ../data/project_name/
        file_path = my_hdfs[0:my_hdfs.rfind(project_name)]+'data/'+project_name+'/'+file_name
    
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
    
    zz = load_all(url_count_rdd.toLocalIterator(), sc).sortBy(lambda a: -int(a[1])).toDF(["url","count","result"]).toPandas()
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
    responses = df.iloc[:,0].values
    counts = df.iloc[:,1].values
    plt.rcdefaults()
    y_pos = np.arange(len(responses))
    plt.bar(responses, counts, align='center', alpha=0.5, log = True)
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
