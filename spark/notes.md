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
- Anytime you perform more than one action on a dataframe, you should cache it. Otherwise, Spark might re-evaluate the entire dataframe again.
	- `.cache()` caches it in memory, `.persist()` caches it to disk (recovers better if node fails, but takes longer)
- In this line `spark = SparkSession.builder.appName("MovieSimilarities").master("local[*]").getOrCreate()`, `"local[*]"` indicates to use all CPU cores to run this job. Be careful, in a cluster this could use all CPUs available.

# Running Spark on a Cluster

- AWS Elastic MapReduce (EMR)
	- Sets up a default Spark configuarion for you on top of Hadoop's YARN cluster manager
	- Spark also has a built-in standalone cluster manager and scripts to set up its own EC2-based cluster, but the AWS console is even easier
	- By default, AWS MapReduce uses m3.xlarge instances which can be a bit expensive
	- Remember to shut down your cluster when you're done

## Optimizing for Running on a Cluster: Partitions

- Use `.partitionBy()` on an RDD before running a large operation that benefits from partitioning
	- Those operations include: `.join()`, `.join()`, `.groupWith()`, `.leftOuterJoin()`, `.rightOuterJoin()`, `.groupByKey()`, `.reduceByKey()`, `.combineByKey()`, `.lookup()`
	- Those operations will preserve your partitioning in their result too
- Too few partitions won't take full advantage of your cluster
- Too many results in too much overhead from shuffling data
- You should have at least as many partitions as you have cores, or executors, that fit within your available memory
- `.partitionBy(100)` is usually a reasonable place to start for large operations

## Troubleshooting & Managing Dependencies

- `localhost:4040` will show you DAGs, logs (if in standalone mode, if distributed you can collect them with `yarn logs -applicationID <appID>`), tasks to help troubleshoot when needed
- While your driver script runs, it will log errors like executors failing to issue heartbeats
	- This generally means you're asking too much of each executor, so you may need more of them or they may need more memory
	- Or, use `.partitionBy()` to demand less work form individual executors by using smaller partitions
- Remember your executors aren't necessarily on the same box as your driver script
- Use broadcast variables to shakre data outside of RDDs
- Need a python package that's not pre-loaded on EMR?
	- Set up a step in EMR to run pip for what you need on each worker machine
	- Or use -py-files with spark-submit to add individual libraries that are on the master
	- Try to just avoid using obscure packages you don't need in the first place. Time is money on your cluster, and you're better off not fiddling with it.

# Machine Learning with Spark ML

- ML capabilities:
	- Feature extraction (i.e. tf-idf, etc)
	- Basic statistics
		- Chi-squared test, Pearson or Spearman correlation, min, max, mean, variance
	- Linear regression, logistic regression
	- Support Vector Machines
	- Naive Bayes classifier
	- Decision trees
	- K-Means clustering
	- PCA, SVD
	- Recommendations using Alternating Least Squares
- Previous API was called MLLib and used RDDs and some specialized data structures, but is deprecated in Spark 3
- The newer ML library uses dataframes for everything
- For more depth, read "Advanced Analytics with Spark" from O'Reilly

## Linear Regression with Spark ML

- Train the model and make predictions using tuples taht consist of a label and a vector of features.
- Stochastic Gradient Descent doesn't handle features at different scales well, so you will need to scale it
- You need to call `fitIntercept(true)` to not assume the y-intercept is 0


## Notes

- Putting your faith in a black box is dodgy
- Never blindly trust results when analyzing big data
- Small problems in algorithms become big ones
- Very often, quality of your input data is the real issue
- Might be specific to Decision Trees:
	- You can have multiple input columns in a VectorAssembler
		- `assembler = Vector Assembler().setInputCols(["col1", "col2", ...])`
		- `df = assembler.transform(data).select("labelColumnName", "features")`
	- `.option("header", "true").toption("inferSchema", "true")`
