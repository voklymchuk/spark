import os
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
import datetime as dt
import dateutil.parser
import pytz
import pyspark
import pandas as pd
from pyspark.sql.functions import regexp_extract, col
from pyspark.sql.types import StringType, IntegerType, TimestampType
from pyspark.sql.functions import udf, expr, substring, expr, regexp_replace, count
from pyspark.sql.functions import udf, expr, substring, expr, regexp_replace, count
from pyspark.sql.functions import unix_timestamp, col, max, min
from pyspark.sql.functions import concat_ws, collect_list
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.sql.functions import row_number,lit
from pyspark.sql.window import Window
from pyspark.sql.functions import concat_ws, collect_list
from pyspark.sql.functions import hour, year, month
from pyspark.ml.clustering import KMeans
from gensim import corpora, models


def to_utc_timestamp(string):
    zone_idx = string.find('-') if string.find('-') >= 0 else string.find('+')
    zone_abbr = string[string.find('(')-1:string.find(')')] # not used
    timezone_str = string[zone_idx:string.find('(')-1]
    date_time_str = string[:zone_idx-1]
    date_time_obj = dt.datetime.strptime(date_time_str, '%d %b %Y %H:%M:%S')
    dt2 = dateutil.parser.parse(str(date_time_obj)+timezone_str)
    return dt2.astimezone(pytz.timezone("UTC"))

def get_topic(cluster, transformed):
    #list_of_list_of_tokens
    list_of_list_of_tokens  = [row.filtered for row in transformed.filter(col("prediction")==cluster).select("filtered").collect()]
    dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
    #dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]
    num_topics = 1
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary_LDA, passes=4, alpha=[0.01]*num_topics, eta=[0.01]*len(dictionary_LDA.keys()))
    string = ""
    for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=6):
        string += (str(i)+": "+ topic)
        string += ("\n")
    return string

def set_pandas_options() -> None:
    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 100
    pd.options.display.max_colwidth = 120
    pd.options.display.width = 140
    # pd.options.display.precision = 2  # set as needed

def get_hdfs_filepath(file_name, on_cloud=True):
    # path to folder containing this code
    prefix = '/data/spark/8_cs2_dataset/'
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
    app_name="Case Study 2: Email Analytics"

    conf = SparkConf().setAppName(app_name)
    conf = (conf.setMaster('local[*]')
            .set("spark.driver.host", "localhost")
            .set('spark.executor.memory', '4G')
            .set('spark.driver.memory', '8G')
            .set('spark.driver.maxResultSize', '10G'))
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")

    # 1 Load data into Spark DataFrame
    LOG = get_hdfs_filepath('*/*/*')

    # read text file
    log_txt_df=sc.wholeTextFiles(LOG).filter(lambda line: line != '').toDF()
    # Convert strings to columns
    udf1 = udf(to_utc_timestamp, TimestampType())
    df = log_txt_df
    df = df.select(df._2.alias('line') )
    udf1 = udf(to_utc_timestamp, TimestampType())
    temp = df.select(
        regexp_extract(col('line'), r'Message-ID:\s<.*>',0).alias('Message_ID'),
        regexp_extract(col('line'), r'\d{1,2}\s\w{3}\s\d{4}\s\d{2}:\d{2}:\d{2}\s(\+|\-)\d{4}(.*)', 0).alias("Date"),
        regexp_extract(col('line'), r'From:\s(.*)', 0).alias("From"),
        regexp_extract(col('line'), r"To:\s(.+)((?:\n|\r\n?)((?:(?:\n|\r\n?).+)+)){0,}(\S+@\S+)(?:\n|\r\n?)Subject:\s", 0).alias("To"),
        regexp_extract(col('line'), r"Subject:\s(.+)((?:\n|\r\n?)((?:(?:\n|\r\n?).+)+)){0,}", 1).alias("Subject"),
        regexp_extract(col('line'), r"Cc:\s(.+)((?:\n|\r\n?)((?:(?:\n|\r\n?).+)+)){0,}(?:\n|\r\n?)Mime-Version:\s", 0).alias("Cc"),
        regexp_extract(col('line'), r'Mime-Version:\s(.+)', 1).alias("Mime_Version"),
        regexp_extract(col('line'), r'Content-Type:\s(.*)', 1).alias("Content_Type"),
        regexp_extract(col('line'), r"Content-Transfer-Encoding:\s(.+)", 1).alias("Content_Transfer_Encoding"),
        regexp_extract(col('line'), r"X-From:\s(.*)(?:\n|\r\n?)X-To:\s", 0).alias("X_From"),
        regexp_extract(col('line'), r'X-To:\s(.*)(?:\n|\r\n?)X-cc:\s', 0).alias("X_To"),
        regexp_extract(col('line'), r'X-cc:\s(.*)(?:\n|\r\n?)X-bcc:\s', 0).alias("X_cc"),
        regexp_extract(col('line'), r'X-bcc:\s(.*)(?:\n|\r\n?)X-Folder:\s', 0).alias("X_bcc"),
        regexp_extract(col('line'), r'X-Folder:\s(.*)(?:\n|\r\n?)X-Origin:\s', 0).alias("X_Folder"),
        regexp_extract(col('line'), r"X-Origin:\s(.*)(?:\n|\r\n?)X-FileName:\s", 0).alias("X_Origin"),
        regexp_extract(col('line'), r"X-FileName:\s(.*)", 0).alias("X_FileName"),
        regexp_extract(col('line'), r"X-FileName:\s(.*)((?:\n|\r\n?){1,}(.*)){1,}((?:(?:\n|\r\n?).+)+)", 0).alias("FYI")
    )
    #temp.cache()
    temp1 = temp.select(
        expr("substring(Message_ID, 14, length(Message_ID)-14)").alias("Message_ID"),
        'Date',
        udf1('Date').alias('UTC_timestamp'),
        expr("substring(From, 7, length(From)-6)").alias("From"),
        expr("substring(To, 5, length(To)-15)").alias("To"),
        "Subject",
        expr("substring(Cc, 5, length(Cc)-20)").alias("Cc"),
        "Mime_Version",
        "Content_Type",
        'Content_Transfer_Encoding',
        expr("substring(X_From, 9, length(X_From)-16)").alias("X_From"),
        expr("substring(X_To, 7, length(X_To)-14)").alias("X_To"),
        expr("substring(X_cc, 7, length(X_cc)-15)").alias("X_cc"),
        expr("substring(X_bcc, 8, length(X_bcc)-19)").alias("X_bcc"),
        expr("substring(X_Folder, 11, length(X_Folder)-22)").alias("X_Folder"),
        expr("substring(X_Origin, 11, length(X_Origin)-24)").alias("X_Origin"),
        expr("substring(X_FileName, 13, length(X_FileName)-15)").alias("X_FileName"),
        regexp_replace(col('FYI'), r"(X-FileName:\s(.*)(?:\n|\r\n?){1,})|(-*Original Message-*(.*)((?:\n|\r\n?){1,}(.*)){0,}((?:(?:\n|\r\n?).+)+))", '').alias('FYI')
    )
    #temp1.cache()
    result = temp1.select(
        "Message_ID",
        'Date',
        'UTC_timestamp',
        "From",
        regexp_replace(col('To'), r"\r\n\t", "").alias("To"),
        "Subject",
        regexp_replace(col('Cc'), r"\r\n\t", "").alias("Cc"),
        "Mime_Version",
        "Content_Type",
        'Content_Transfer_Encoding',
        "X_From",
        "X_To",
        "X_cc",
        "X_bcc",
        "X_Folder",
        "X_Origin",
        "X_FileName",
        regexp_replace(col('FYI'), r"(^\s{1,})|(\n{2,})", '').alias('FYI')
    )
    zz = result.limit(5).toPandas()
    LOGGER.info("\n\n1.\tLoad data into Spark DataFrame\tDone!\n\n{}\n".format(zz))

    # 2 Display the top 10 high-frequency users based on weekly numbers of emails sent
    df1 = result
    freq = df1.groupBy('From').agg((count('UTC_timestamp') / ( (max(unix_timestamp(col('UTC_timestamp')))-min(unix_timestamp(col('UTC_timestamp'))))/ 604800)).alias('rate_per_week')).orderBy("rate_per_week",ascending=False)
    zz = freq.limit(10).toPandas()
    LOGGER.info("\n\n2.\tDisplay the top 10 high-frequency users based on weekly numbers of emails sent\tDone!\n\n{}\n".format(zz))

    # 3a Extract top 20 keywords from the subject text for the top 10 high-frequency users
    top = freq.limit(10)
    top_subj = df1.join(top, df1["From"] == top["From"], "inner").select(df1['From'], df1['Subject'])
    top_texts = top_subj.groupBy("From").agg(concat_ws(" ", collect_list("Subject")).alias("texts"))
    top_texts = top_texts.select('texts').agg(concat_ws(" ", collect_list("texts")).alias("subjects"))
    # Extract word
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer().setInputCol("subjects").setOutputCol("words")
    transformed = tokenizer.transform(top_texts)
    # Extend the stop words dictionary by adding your own stop words such as -
    # Remove stopwords
    # custom stopwords
    stopwords = StopWordsRemover().getStopWords() + ["-", "re:", "", "fw"]
    remover = StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
    cleaned = remover.transform(transformed)
    # Extract top 20 keywords by identifying removing the common stop words
    # Generate features
    from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
    cvmodel = CountVectorizer().setInputCol("filtered").setOutputCol("features").fit(cleaned)
    featured = cvmodel.transform(cleaned)
    counts = featured.select('features').collect()
    a = cvmodel.vocabulary
    b = counts[0]['features'].values
    d = {'words':a,'counts':b}
    df = pd.DataFrame(d)
    zz = df.head(20)
    LOGGER.info("\n\n3a.\tExtract top 20 keywords from the subject text for the top 10 high-frequency users\tDone!\n\n{}\n".format(zz))
    # 3b Extract top 20 keywords from the subject text for the non-high frequency users
    w = Window().orderBy(lit('A'))
    bottom = freq.orderBy("rate_per_week",ascending=False).withColumn("row_num", row_number().over(w))
    bottom = bottom.where(col('row_num')>10).select('From','rate_per_week')
    bottom_subj = df1.join(bottom, df1["From"] == bottom["From"], "inner").select(df1["From"], df1["Subject"])
    bottom_texts = bottom_subj.groupBy("From").agg(concat_ws(" ", collect_list("Subject")).alias("texts"))
    bottom_texts = bottom_texts.select('texts').agg(concat_ws(" ", collect_list("texts")).alias("subjects"))
    # Extract word
    tokenizer = Tokenizer().setInputCol("subjects").setOutputCol("words")
    transformed = tokenizer.transform(bottom_texts)
    # Remove stopwords
    # custom stopwords
    stopwords = StopWordsRemover().getStopWords() + ["-", "re:", "fw:", "", "&"]
    remover = StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
    cleaned = remover.transform(transformed)
    # Generate features
    cvmodel = CountVectorizer().setInputCol("filtered").setOutputCol("features").fit(cleaned)
    featured = cvmodel.transform(cleaned)
    counts = featured.select('features').collect()
    a = cvmodel.vocabulary
    b = counts[0]['features'].values
    d = {'words':a,'counts':b}
    df = pd.DataFrame(d)
    zz = df.head(20)
    LOGGER.info("\n\n3b.\tExtract top 20 keywords from the subject text for the non-high frequency users\tDone!\n\n{}\n".format(zz))

    # 6 Introduce a new column label to identify new, replied, and forwarded messages
    df = result
    def to_label(sbj):
        l1 = "RE" if sbj.startswith("RE:") else ("FW" if sbj.startswith("FW:") else 'NEW')
        return l1
    udf2 = udf(to_label, StringType())
    df_with_label = df.withColumn('label', udf2("Subject"))
    zz = df_with_label.limit(5).toPandas()
    LOGGER.info("\n\n6.\tIntroduce a new column label to identify new, replied, and forwarded messages\tDone!\n\n{}\n".format(zz))

    # 7 Get the trend of the over mail activity using the pivot table from spark itself
    pivotDF = df_with_label.groupBy(year("UTC_timestamp").alias('year'), month("UTC_timestamp").alias('month')).pivot("label").count().orderBy("year", "month")
    zz = pivotDF.na.fill(0).toPandas()
    LOGGER.info("\n\n7.\tGet the trend of the over mail activity using the pivot table from spark itself\tDone!\n\n{}\n".format(zz))

    # 8 Use k-means clustering to create 4 clusters from the extracted keywords
    raw = result.select("Message_ID","From", "Subject")
    # Extract word
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer().setInputCol("Subject").setOutputCol("words")
    transformed = tokenizer.transform(raw)
    # Remove stopwords
    # custom stopwords
    stopwords = StopWordsRemover().getStopWords() + ["-", "re:", "fw:", "", "&"]
    remover = StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
    cleaned = remover.transform(transformed)
    cleaned = cleaned.select("Message_ID","words", "filtered")
    # Generate features
    from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
    cvmodel = CountVectorizer().setInputCol("filtered").setOutputCol("features").fit(cleaned)
    featured = cvmodel.transform(cleaned)
    kmeans = KMeans(k=4, seed=1)  # 4 clusters here
    model = kmeans.fit(featured.select('features'))
    transformed = model.transform(featured)
    zz = transformed.limit(5).toPandas()
    LOGGER.info("\n\n8.\tUse k-means clustering to create 4 clusters from the extracted keywords\tDone!\n\n{}\n".format(zz))

    # 9 Use LDA to generate 4 topics from the extracted keywords
    LOGGER.info("\n\n9.\tUse LDA to generate 4 topics from the extracted keywords\tDone!\n\n{}\n{}\n{}\n{}\n".format(get_topic(0, transformed),get_topic(1, transformed),get_topic(2, transformed), get_topic(3, transformed) ))
'''
    # 10 Remove punctuation and cluster text of the emails
    raw = result.select("Message_ID","From", "FYI")
    def lower_clean_str(x):
        punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        lowercased_str = x.lower()
        for ch in punc:
            lowercased_str = lowercased_str.replace(ch, '')
        return lowercased_str
    udf3 = udf (lower_clean_str, StringType())
    raw = raw.select("Message_ID","From", udf3("FYI").alias('FYI'))
    # Extract word
    tokenizer = Tokenizer().setInputCol("FYI").setOutputCol("words")
    transformed = tokenizer.transform(raw)
    # Remove stopwords
    # custom stopwords
    stopwords = StopWordsRemover().getStopWords() + ["-", "re:", "fw:", "", "&"]
    remover = StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
    cleaned = remover.transform(transformed)
    cleaned = cleaned.select("Message_ID","words", "filtered")
    cvmodel = CountVectorizer().setInputCol("filtered").setOutputCol("features").fit(cleaned)
    # Generate features
    cvmodel = CountVectorizer().setInputCol("filtered").setOutputCol("features").fit(cleaned)
    featured = cvmodel.transform(cleaned)
    kmeans = KMeans(k=4, seed=1)  # 4 clusters here
    model = kmeans.fit(featured.select('features'))
    transformed = model.transform(featured)
    zz = transformed.limit(5).toPandas()
    LOGGER.info("\n\n10.\tUse k-means clustering to create 4 clusters from the extracted keywords\tDone!\n\n{}\n".format(zz))
'''

if __name__ == "__main__":
    main()
