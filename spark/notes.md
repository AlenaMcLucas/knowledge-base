# Spark Basics and the RDD Interface


## What's new in Spark 3?

- Support for machine learning is stopping for RDDs, now only for dataframes/sets
- Way faster than Spark 2
- Deprecate Python 2
- Support for GPU processing
- Graph X is moving towards being replaced by SparkGraph, which is a lot more extensible
- ACID support in data lakes is now possible with Delta Lake

## Introduction to Spark

- "A fast and general engine for large-scale data processing"
- Your driver program, or your script, runs on your desktop or your cluster's master node. When your run it, spark knows how to take that script and distribute the work across clusters or CPUs. Spark has its own built-in, default cluster manager, or you can customize it using Yarn. It will split the work and send it to executors. Ideally, you want one executor per CPU core. If one executor goes down, it can recover without actually stopping your entire job.
- People choose Spark because:
	- It runs programs up to 100x faster than Hadoop MapReduce in memory, or 10x faster on disk. (Practically, more like 2-3x faster.)
	- DAG Engine (directed acyclic graph) optimizes workflows
	- Code in Python, Java, or Scala
	- Built around one main concept: the Resilient Distributed Dataset (RDD)
	- It also contains Spark Streaming, Spark SQL, MLLib, and GraphX

## Resilient Distributed Dataset (RDD)

- An RDD is an abstraction for a large dataset built to distribute data processing and handle execution failures
- The Spark Context is:
	- Created by your drive program
	- Responsible for making RDD's resilient and distributed
	- Creates RDDs
	- Note that the Spark shell creates a "sc" object for you
- RDDs can be created, transformed (map, flatMap, filter, distinct, sample, union, intersection, subtract, cartesian), actioned (collect, count, countByValue, take, top, reduce, etc)
- When you call an action, then Spark starts processing. This is called lazy evaluation.

## How-To Notes

- Run your script with `python3 my-script.py` when you want to run it locally without distribution, or `spark-submit my-script.py` when you want to run it locally with distribution.

## Key/Value RDDs

- It's exactly what you think it is haha
- Just map pairs of data into the RDD. For example `totalsByAge = rdd.map(lambda x: (x, 1))`.
- You can have lists as values
- Special functions for key/value RRDs:
	- `reduceByKey`: combine values with the same key using some function `rdd.reduceByKey(lambda x, y: x + y)`
	- `groupByKey`: groups values with the same key
	- `sortByKey`: sort RDD by key values
	- `keys`, `values`: create an RDD of just the keys or just the values
	- SQL-style joins: `join`, `rightOuterJoin`, `leftOuterJoin`, `cogroup`, `subtractByKey`
	- When you want to map (a transformation that doesn't affect the keys), use `mapValues` and `flatMapValues` for better efficiency because it preserves partitions without shuffling the data around
		- These functions will only receive the values, but not the keys (which are still preserved)

## Filtering RDDs

- Filters take a function that return a boolean: `lines.filter(lambda x: x < 100)`
- Filters remove data from your RDDs

## map vs flatMap

- map maintains a 1:1 relationship during transformation, and flatMap can create multiple new entries from each entry


## Additional Notes

- `cache()` saves the RDD so we can refer back to it later
- Discard unneeded data whenever possible so you aren't wasting resources carrying it around.


# SparkSQL, DataFrames, and DataSets


## SparkSQL

- The trend is to us RDDs less and DataFrames more
- Extends RDD to a DataFrame object
- Contains row objects; can run SQL queries; can have a schema (storage efficiency); read and write to JSON, Hive, parquet, csv, etc; communicates with JDBC/ODBC (can look like a database), Tableau
- Things we can do with DataFrames:
	- `dataFrameName.sql('SELECT potato FROM tomato')`
	- `dataFrameName.show()`: shows the first 20 rows
	- `dataFrameName.select('fieldName')`: select that column only
	- `dataFrameName.filter(dataFrameName('fieldName') > 200)`
	- `dataFrameName.groupBy(dataFrameName('fieldName')).mean()`
	- `dataFrameName.rdd().map(mapperFunc)`: turn it into an rdd and apply mapping function to each row of the DataFrame
	- `dataFrameName.sort('fieldName')`
	- `dataFrameName.agg(func.avg('fieldName'))`: to aggregate after groupBy
	- `.alias('friends_avg')`: to give alias name after operation that creates a new column like `agg`
	- `dataFrameName.withColumn('newName', dataFrameName.age + 1)`: creates a new column
- In Python, we use DataFrames more. But in Scala, we use DataSets whenever possible because they can wrap known, explicitly typed data for efficient storage and can be optimized at compile time.
- Can build SQL shell that operates like a console to query a DataFrame
- user-defined functions (udfs): define them and then can use in a query
```
def square(x):
	return x*x

spark.udf.register("square", square, IntegerType())
df = spark.sql("SELECT square('numericField') FROM tableName")
```

## SQL Functions

- `from pyspark.sql import functions as func`
- `func.explode()`: similar to flatMap because explodes columns into rows
- `func.split()`
- `func.lower()`

## Additional Notes

- DataFrames are a better fit for structured data, while RDDs are a better fit for unstructured data
	- You can use both as it fits the problem
- Can refer to column names in three ways:
	- `df.colName` or `df('colName')` or `df.select('colName')`
- When you put a script into production, be sure to remove all debugging steps like `show()` to keep it running as fast as possible
- `master("local[*]")` means 'run locally on all my cpu cores'

# Advanced Spark programs

- Broadcast objects (`sc.broadcast()`) to the executors, such that they're always there whenever needed. Then use `.value()` to get the object back (see script 15).
	- They can be used for map functions, UDFs, dictionaries, or whatever
- Breadth-First-Search algorithm searched through graph and calculates degrees of separation. Select beginning node, it's 0 degrees. Then go to all of its connections and add 1. Repeat until all degrees of separation have been calculated for each node.
- An accumulator allows many executors to increment a shared variable.
