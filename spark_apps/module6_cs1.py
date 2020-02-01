import sys
import os

from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

from pyspark.sql.functions import isnan, when, count, col, avg

import time
import pandas as pd
from pyspark.sql.functions import *


def set_pandas_options():
    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 100
    pd.options.display.max_colwidth = 120
    pd.options.display.width = 140
    # pd.options.display.precision = 2  # set as needed


def get_hdfs_filepath(file_name, on_cloud=True):
    # path to folder containing this code
    
    prefix = '/data/spark/6_cs1_dataset/'
    if on_cloud:
        bucket  = os.environ['BUCKET']
        file_path = bucket + prefix + file_name
    else:
        file_path = '/Users/val' + prefix + file_name

    return file_path





def main():

    set_pandas_options()
    app_name="Case Study 1"
    
    app_name="Case Study 1"

    conf = SparkConf().setAppName(app_name)
    sc = SparkContext(conf = conf)
    spark = SparkSession(sc)

    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")

    # 1 Load data into Spark DataFrame
    AISLES = get_hdfs_filepath('aisles.csv')
    DEPARTMENTS = get_hdfs_filepath('departments.csv')
    ORDER_PRODUCTS_PRIOR = get_hdfs_filepath('order_products__prior.csv')
    ORDER_PRODUCTS_TRAIN = get_hdfs_filepath('order_products__train.csv')
    ORDERS = get_hdfs_filepath('orders.csv')
    PRODUCTS = get_hdfs_filepath('products.csv')
    
    df_aisles = spark.read.csv(AISLES, multiLine=True, header="true",encoding='ASCII', escape= "\"",inferSchema =True)
    df_departments = spark.read.csv(DEPARTMENTS, multiLine=True, header="true",encoding='ASCII', escape= "\"",inferSchema =True)
    df_order_products_prior = spark.read.csv(ORDER_PRODUCTS_PRIOR, multiLine=True, header="true",encoding='ASCII', escape= "\"",inferSchema =True)
    df_order_products_train = spark.read.csv(ORDER_PRODUCTS_TRAIN, multiLine=True, header="true",encoding='ASCII', escape= "\"",inferSchema =True)
    df_orders = spark.read.csv(ORDERS, multiLine=True, header="true",encoding='ASCII', escape= "\"",inferSchema =True)
    df_products = spark.read.csv(PRODUCTS, multiLine=True, header="true",encoding='ASCII', escape= "\"",inferSchema =True)
    LOGGER.info("\n\n1.\tLoad data into Spark DataFrame\tDone!\n")

    # 2 Merge all the data frames based on the common key and create a single DataFrame
    #Using dataframe joins
    # a: orders + order_products
    orders_train_df = df_orders.join(df_order_products_train, "order_id")
    orders_prior_df = df_orders.join(df_order_products_prior, "order_id")
    # b: products + aisles + departments
    prod_isles_dep_df = df_departments.join(df_products, "department_id").join(df_aisles, "aisle_id")
    # c: orders + order_products + products + aisles + departments
    orders_all_train = prod_isles_dep_df.join(orders_train_df, "product_id")
    orders_all_prior = prod_isles_dep_df.join(orders_prior_df, "product_id")
    # d: train + prior
    orders_all_df = orders_all_train.union(orders_all_prior)
    zz = orders_all_df.limit(20).toPandas()
    LOGGER.info("\n\n2.\tMerge all the data frames based on the common key and create a single DataFrame\tDone!\n\n{}\n\n".format(zz))
    
    
    # 3 Check missing data
    zz = orders_all_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in orders_all_df.columns]).toPandas()
    LOGGER.info("\n\n3.\tCheck missing data\tDone!\n\n{}\n".format(zz))

    # 4 List the most ordered products (top 10)
    # Using spark.sql querries
    df_aisles.createOrReplaceTempView('Aisles')
    df_departments.createOrReplaceTempView('Departments')
    df_order_products_prior.createOrReplaceTempView('Order_products_prior')
    df_order_products_train.createOrReplaceTempView('Order_products_train')
    df_orders.createOrReplaceTempView('Orders')
    df_products.createOrReplaceTempView('Products')
    querry = """
    Select
      m1.order_id,
      m1.user_id,
      m1.eval_set,
      m1.order_number,
      m1.order_dow,
      m1.order_hour_of_day,
      m1.days_since_prior_order,
      m1.product_id,
      m1.add_to_cart_order,
      m1.reordered,
      m2.product_name,
      m2.aisle_id,
      m2.aisle,
      m2.department_id,
      m2.department 
    From
      (
        Select
          o.order_id,
          o.user_id,
          o.eval_set,
          o.order_number,
          o.order_dow,
          o.order_hour_of_day,
          o.days_since_prior_order,
          op.product_id,
          op.add_to_cart_order,
          op.reordered 
        From
          Orders o 
          INNER JOIN
            (
              SELECT
                * 
              FROM
                Order_products_train 
              UNION ALL
              SELECT
                * 
              FROM
                Order_products_prior
            )
            op 
            on o.order_id = op.order_id 
      )
      m1 
      INNER JOIN
        (
          Select
            p.product_id,
            p.product_name,
            p.aisle_id,
            a.aisle,
            p.department_id,
            d.department 
          From
            Departments d 
            INNER JOIN
              Products p 
              on d.department_id = p.department_id 
            INNER JOIN
              Aisles a 
              on p.aisle_id = a.aisle_id 
        )
        m2 
        on m1.product_id = m2.product_id
    """
    orders_all_df = spark.sql(querry)
    orders_all_df.createOrReplaceTempView('Orders_all')
    querry1 = """
    Select
      product_id,
      product_name,
      SUM(add_to_cart_order)
    FROM
      Orders_all 
    GROUP BY
      product_id,
      product_name 
    ORDER BY
      SUM(add_to_cart_order) DESC
    """
    most_ordered_df = spark.sql(querry1).limit(10).toPandas()
    zz = most_ordered_df
    LOGGER.info("\n\n4.\tList the most ordered products (top 10)\tDone!\n\n{}\n".format(zz))
                

    # 5 Do people usually reorder the same previously ordered products?
    querry5 ="""
    Select t1.user_id, t1.count as no_of_reordered_product_ids, t2.count as no_of_new_product_ids, t1.count/(t1.count+t2.count) as percent_reordered
    FROM 
        (
        Select 
          user_id,
          COUNT(product_id) as count
        FROM
          Orders_all
        WHERE
          reordered = 1
        GROUP BY
          user_id
        ) t1
        JOIN
        (
        Select 
          user_id,
          COUNT(product_id) as count
        FROM
          Orders_all
        WHERE
          reordered = 0
        GROUP BY
          user_id
        ) t2
        ON t1.user_id=t2.user_id
    ORDER BY 
     percent_reordered DESC
    """
    reordered5_df = spark.sql(querry5)
    result = reordered5_df.agg(avg('no_of_reordered_product_ids'), avg('no_of_new_product_ids'))
    zz = result.toPandas()
                
    LOGGER.info("\n\n5.\tDo people usually reorder the same previously ordered products?\tDone!\n\n{}\n".format(zz))
    LOGGER.info("\n\n\tSince avg(no_of_reordered_product_ids) is greater than avg(no_of_new_product_ids), we conclude that the answer is YES: people are 30% more likely to reorder same items than buy something new\n\n")
                 
    # 6 List most reordered product
    zz = orders_all_df.groupBy("product_name").agg(
        avg("reordered").alias("percent_reorders"), 
        count("reordered").alias('no_of_orders'),
        sum("add_to_cart_order").alias("untis_sold"),  
        (sum("add_to_cart_order")/countDistinct("order_id")).alias('units_per_order')
    ).sort(
        col("percent_reorders").desc()
    ).limit(20).toPandas()
    LOGGER.info("\n\n6.\tList most reordered products\tDone!\n\n{}\n".format(zz))

    # 7 Most important department and aisle (by number of products)
    # 2: products + aisles + departments
    prod_isles_dep_df = df_departments.join(df_products, "department_id").join(df_aisles, "aisle_id")
    zz = prod_isles_dep_df.groupBy("department").count().sort(col("count").desc()).limit(10).toPandas()
    zz1 = prod_isles_dep_df.groupBy("aisle").count().sort(col("count").desc()).limit(10).toPandas()
    LOGGER.info("\n\n7.\tMost important department and aisle (by number of products)\tDone!\n\n{}\n{}".format(zz, zz1))

    # 8 Get the Top 10 departments
    zz = orders_all_df.groupBy("department").agg(sum("reordered").alias("no_of_orders")).sort(col("no_of_orders").desc()).limit(10).toPandas()
    LOGGER.info("\n\n8.\tGet the Top 10 departments\tDone!\n\n{}\n".format(zz))

    # 9 List top 10 products ordered in the morning (6 AM to 11 AM)
    zz = orders_all_df.where(
    (orders_all_df["order_hour_of_day"]>6) & (orders_all_df["order_hour_of_day"]<11)
).groupBy("product_name").agg(
    sum("reordered").alias("no_of_orders"),sum("add_to_cart_order").alias("no_of_units")
).sort(col("no_of_units").desc()).limit(10).toPandas()
    LOGGER.info("\n\n9.\tList top 10 products ordered in the morning (6 AM to 11 AM)\tDone!\n\n{}\n".format(zz))

    LOGGER.info("\n\n10.\tCreate a spark-submit application for the same and print the findings in the log\tDone!\n\n")


if __name__ == "__main__":
    main()
