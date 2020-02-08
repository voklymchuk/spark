import sys
import os
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession


def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None
    # pd.options.display.precision = 2  # set as needed


def get_hdfs_filepath(file_name, on_cloud=True):
    # path to folder containing this code

    prefix = '/data/spark/5_cs1_dataset/'
    if on_cloud:
        bucket = os.environ['BUCKET']
        file_path = bucket + prefix + file_name
    else:
        file_path = '/Users/val' + prefix + file_name

    return file_path


def main():

    set_pandas_options()

    app_name = "Case Study 1"

    # create Spark context with Spark configuration
    conf = SparkConf().setAppName(app_name)
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # sc.setLogLevel("INFO")
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")

    # 1 load csv into spark
    APP_STORE = get_hdfs_filepath('AppleStore.csv')
    DESCRIPTIONS = get_hdfs_filepath('appleStore_description.csv')

    df_store = spark.read.csv(APP_STORE, multiLine=True, header="true",
                              encoding='utf-8', escape="\"", inferSchema=True)
    df_desc = spark.read.csv(DESCRIPTIONS, multiLine=True, header="true",
                             encoding='utf-8', escape="\"", inferSchema=True)
    LOGGER.info("\n\n1.\tLoad csv into spark\tDone!\n")

    # 2 parse the data as csv
    df_store = df_store.drop('_c0')
    cols = ['id',
            'track_name',
            'size_bytes',
            'currency',
            'price',
            'rating_count_tot',
            'rating_count_ver',
            'user_rating',
            'user_rating_ver',
            'ver',
            'cont_rating',
            'prime_genre',
            'sup_devices_num',
            'screenshots_num',
            'lang_num',
            'vpp_lic']
    # rename columns containing period character
    df_store = df_store.toDF(*cols)
    LOGGER.info("\n\n2.\tParse the data as csv\tDone!\n")

    # 3 convert bytes to MB and GB in a new column
    df_store = df_store.withColumn(
        "MB", df_store['size_bytes']/1024).withColumn("GB", df_store['size_bytes']/1024/1024)
    dfp = df_store.toPandas()
    LOGGER.info("\n\n3.\tConvert bytes to MB and GB in a new column\tDone!\n\n{}".format(
        dfp.head(10).iloc[:, [0, 1, 2, -2, -1]])+"\n")

    # 4 list top 10 trending apps
    dfp = df_store.sort(df_store.rating_count_tot.desc()).limit(10).toPandas()
    LOGGER.info("\n\n4.\tList top 10 trending apps\tDone!\n\n{}".format(
        dfp.head(10).iloc[:, [0, 1, 5]])+"\n")

    # 5 the difference in the average number of screenshots displayed of highest and lowest rating apps
    import pyspark.sql.functions as F
    min_rating, max_rating = df_store.agg(
        F.min(df_store.user_rating), F.max(df_store.user_rating)).collect()[0]
    df1 = df_store
    a = df1.filter(df1.user_rating == max_rating).agg(F.avg(df1.screenshots_num))
    b = df1.filter(df1.user_rating == min_rating).agg(F.avg(df1.screenshots_num))
    diff = a.first()[0] - b.first()[0]
    LOGGER.info("\n\n5.\tThe difference in the average number of screenshots displayed of highest and lowest rating apps\tDone!\n\n{}".format(diff)+"\n")

    # 6 what percentage of high rated apps support multiplelanguages
    perc = df1.filter(df1.lang_num > 1).filter(df1.user_rating == max_rating).count(
    ) * 100 / df1.filter(df1.user_rating == max_rating).count()
    LOGGER.info(
        "\n\n6.\tWhat percentage of high rated apps support multiplelanguages\tDone!\n\n{}".format(perc)+"\n")

    # 7 how does app details contribute to user ratings?
    # get percentiles of ratings
    percentiles = df1.stat.approxQuantile("user_rating", [0.25, 0.50, 0.75], 0.0)
    df_25 = df1.filter(df1.user_rating < percentiles[0])
    df_50 = df1.filter((df1.user_rating >= percentiles[0]) & (df1.user_rating < percentiles[1]))
    df_75 = df1.filter((df1.user_rating >= percentiles[1]) & (df1.user_rating < percentiles[2]))
    df_100 = df1.filter(df1.user_rating >= percentiles[2])
    # compare statistics
    q1 = df_25.agg(F.avg(df_25.lang_num))
    q2 = df_50.agg(F.avg(df_50.lang_num))
    q3 = df_75.agg(F.avg(df_75.lang_num))
    q4 = df_100.agg(F.avg(df_100.lang_num))
    import pandas as pd
    data = [q1.first()[0], q2.first()[0], q3.first()[0], q4.first()[0]]
    dfp = pd.DataFrame(zip([25, 50, 75, 100], data), columns=[
                       'percentile', 'avg_lang_num']).set_index('percentile')
    LOGGER.info("\n\n7.\tHow does app details contribute to user ratings\tDone!\n\n{}".format(
        dfp.head(10))+"\n")

    # 8 compare the statistics of different app groups/genres
    dfp = df1.groupBy("prime_genre").agg(F.avg(df1.lang_num), F.avg(
        df1.screenshots_num), F.avg(df1.rating_count_tot), F.avg(df1.MB)).toPandas()
    # with pd.option_context('display.max_rows', None, 'display.max_columns',5, 'display.width',1000):
    LOGGER.info(
        "\n\n8.\tCompare the statistics of different app groups/genres\tDone!\n\n{}".format(dfp.head(10))+"\n")

    # 9 p length of app description contribute to the ratings?
    df2 = df_desc
    # Create new column for length of description
    df2 = df2.withColumn("desc_len", F.length(df2.app_desc))
    inner_join = df1.join(df2.select([df2.id, df2.app_desc, df2.desc_len]), 'id', 'outer')
    df3 = inner_join
    # Get percentiles of ratings
    percentiles = df3.stat.approxQuantile("rating_count_tot", [0.25, 0.50, 0.75], 0.0)
    # Get the datasets with different percentiles
    df_25 = df3.filter(df3.rating_count_tot < percentiles[0])
    df_50 = df3.filter((df3.rating_count_tot >= percentiles[0]) & (
        df3.rating_count_tot < percentiles[1]))
    df_75 = df3.filter((df3.rating_count_tot >= percentiles[1]) & (
        df3.rating_count_tot < percentiles[2]))
    df_100 = df3.filter(df3.rating_count_tot >= percentiles[2])
    # Compare the statistics
    q1 = df_25.agg(F.avg(df_25.desc_len))
    q2 = df_50.agg(F.avg(df_50.desc_len))
    q3 = df_75.agg(F.avg(df_75.desc_len))
    q4 = df_100.agg(F.avg(df_100.desc_len))
    data = [q1.first()[0], q2.first()[0], q3.first()[0], q4.first()[0]]
    dfp = pd.DataFrame(zip([25, 50, 75, 100], data), columns=[
                       'rating_percentile', 'avg_desc_len']).set_index('rating_percentile')
    LOGGER.info("\n\n9.\tDoes length of app description contribute to the ratings?\tDone!\n\n{}".format(
        dfp.head(10))+"\n")

    LOGGER.info(
        "\n\n10.\tCreate a spark-submit application for the same and print the findings in the log\tDone!\n\n")


if __name__ == "__main__":
    main()
