# RDDs

## Importing

```
from pyspark import SparkConf, SparkContext
```

## Configuration and Context

```
conf = SparkConf().setMaster("local").setAppName("AppName")
sc = SparkContext(conf = conf)
```

## Reading in Data

```
lines = sc.textFile("/path/to/file/file.data")   # .csv, .txt, etc
```

## Transformations

- `rdd.map(func)` - transforms rdd elements with func
- `rdd.mapValues(func)` - pass each value in a key-value pair rdd through a map function without changing the keys
- `rdd.flatMap(func)` - similar to map, where each element can be mapped to 0 or more output elements
- `rdd.reduceByKey(func)` - for key-value rdd, returns another key-value pair rdd where the values for each key are aggregated using func, should take two elements/values and return one
- `rdd.filter(func)` - returns new rdd formed by selecting on elements where func returns true
- `rdd.groupByKey()` - for key-value rdd, returns another key-value pair where value is now iterable to be aggregated
- `rdd.sortByKey([ascending])` - for key-value rdd, returns another key-value pair sorted by keys
- `rdd1.join(rdd2)` - when called on datasets of type (K, V) and (K, W), returns a dataset of (K, (V, W)) pairs with all pairs of elements for each key
- `rdd.sample(withReplacement, fraction, seed)` - sample a fraction of the data, with or without replacement, using a seed
- `rdd1.union(rdd2)`
- `rdd1.intersection(rdd2)`

## Actions

- `rdd.reduce(func)` - aggregate the elements with func, which should take two elements and return one
- `rdd.count()` - counts number of elements
- `rdd.countByValue()` - count unique values, returning (value, count) pairs
- `rdd.countByKey()` - for key-value rdd, count number of times a key occurs, returning (key, count) pairs
- `rdd.first()` - retun first element, similar to `rdd.take(1)`
- `rdd.take(n)` - return array with the first n elements
- `rdd.takeSample(withReplacement, n, [seed])` - return array with random sample of n elements
- `rdd.collect()` - return all elements as an array

- `rdd.saveAsTextFile("file-name")` - writes elements as a Hadoop SequenceFile in the local file system

## Other

- `rdd.cache()`


# DataFrames

## Importing

```
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, LongType
```

## SparkSession, Context, and Reading in Data

```
def mapper(line):
    ...
    return Row(ID=line[0], name=line[1], age=line[2], numFriends=line[3])

# Create a SparkSession
spark = SparkSession.builder.appName("AppName").getOrCreate()

rdd = spark.sparkContext.textFile("/path/to/file/file.data")   # .csv, .txt, etc; creates rdd
rdd_rows = rdd.map(mapper)   # rdd of structured rows

# Infer the schema
df = spark.createDataFrame(rdd_rows).cache()   # dataframe
```

- you can also add `master()` to indicate XXX what nodes the jobs should run on. `"local[*]"` means 'run locally on all my cpu cores'.
```
spark = SparkSession.builder.appName("AppName").master("local[*]").getOrCreate()
```

- tell spark to try to infer the schema
```
df = spark.read.option("header", "true").option("inferSchema", "true").csv("/path/to/file/file.csv")   # dataframe
```

- read each line in as an individual value, with a default column called 'value'
```
df = spark.read.text("/path/to/file/file.txt")
```

- define schema
```
# Define schema
schema = StructType([StructField("stationID", StringType(), True), \
                     StructField("date", IntegerType(), True), \
                     StructField("measure_type", StringType(), True), \
                     StructField("temperature", FloatType(), True)])

# Read the file as dataframe with schema
df = spark.read.option("sep", "\n").schema(schema).csv("/path/to/file/file.csv")
```


## Note

- for `"field"`, it could be either `"age"` or `df.age`


## Basic Operations

- `df.count()` - count non-NA cells
- `df.avg("field")`
- `df.agg(['sum', 'min'])[['A', 'B', 'C']]` - aggregate using one or more operations
- `df.agg(func.round(func.avg("field"), 2).alias("field_avg"))` - aggregate using one or more operations
- `df.min("field")`
- `df.max("field")`


## Options

- `df = spark.read.option([x], [y]).csv("../data/fakefriends-header.csv")`
- `.option("header", "true")`
- `.option("inferSchema", "true")`
- `.option("sep", "|")`
- `.option("charset", "ISO-8859-1")`


## Select

- `df.select("field")`
- `df.select("field", "field" >= 21)`
- `df.select("field1", "field2")`
- `df.select(func.explode(func.split(df.value, "\\W+")))`
- `df.withColumn("field", func.round(func.col("min(field)") * 0.1 * (9.0 / 5.0) + 32.0, 2))`


## Functions

- `func.size()`
- `func.trim()`
- `func.sum()`
- `func.lower("field")`
- `func.col("min(field)")`
- `func.desc("field")`
- `func.sqrt()`
- `func.count()`


## Other XXXs

- `df.cache()`
- `df.printSchema()`
- `df.groupBy("field")`
- `df.sort("field")`
- `df.sort(func.col("field").desc()`
- `df.orderBy("field")`
- `df.orderBy(func.desc("field"))`
- `df.filter("field" >= 21)`
- `df1.join(df2, "join_field")` - can pass how='left', 'right', 'outer', 'inner', default 'left'
- `df.first()`
- `df.show(10)`
- `df.take([0,-1])` - returns elements in given positional indicies along an axis, default 0==rows



## Querying DataFrames

```
# Register the DataFrame as a table
schemaPeople.createOrReplaceTempView("people")   # create temporary view to query it

# SQL can be run over DataFrames that have been registered as a table.
teenagers = spark.sql("SELECT * FROM people WHERE age >= 13 AND age <= 19")   # rdd
query_results = teenagers.collect()
```

## Stop SparkSession

```
spark.stop()
```


# Other Concepts

## Broadcasting

- broadcast dictionary to supply it to all nodes
```
load_movies = spark.sparkContext.broadcast(dictionary)
```

- broadcast user-defined function (udf)
```
func_udf = func.udf(lambda x: load_movies.value[x])

# Use it on a dataframe
df_result = df.withColumn("field", func_udf(func.col("field")))
```

## Accumulators

- declare it: `counter = sc.accumulator(0)`
- increment by 1: `counter.add(1)`
- get current value: `counter.value`

## Partitions

- when the data is too large to compute across the deisignated number of clusters, especially if you're using an intensive operation like a join, you should partition the data
- XXX what does 100 mean?
- `rdd.partitionBy(100)`

## Built-In Algorithms

- Alternating Least Squares (ALS)
```
from pyspark.ml.recommendation import ALS

als = ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userID").setItemCol("movieID") \
    .setRatingCol("rating")

model = als.fit(df_ratings)

df_recommendations = model.recommendForUserSubset(df_users, 10).collect()
```

- Linear Regression with y-intercept term
```
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors

rdd_data = rdd_input.map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))

# Convert this RDD to a DataFrame
colNames = ["label", "features"]
df_data = rdd_data.toDF(colNames)

# Let's split our data into training data and testing data
trainTest = df_data.randomSplit([0.5, 0.5])
trainingDF = trainTest[0]
testDF = trainTest[1]

# Now create our linear regression model
lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Train the model using our training data
model = lir.fit(trainingDF)

# Now see if we can predict values in our test data.
# Generate predictions using our linear regression model for all features in our
# test dataframe:
fullPredictions = model.transform(testDF).cache()

```

- Decision Tree
```
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler().setInputCols(["HouseAge", "DistanceToMRT", \
                               "NumberConvenienceStores"]).setOutputCol("features")

df = assembler.transform(df_data).select("PriceOfUnitArea", "features")

# Let's split our data into training data and testing data
trainTest = df.randomSplit([0.5, 0.5])
trainingDF = trainTest[0]
testDF = trainTest[1]

# Now create our decision tree
dtr = DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("PriceOfUnitArea")

# Train the model using our training data
model = dtr.fit(trainingDF)

# Now see if we can predict values in our test data.
# Generate predictions using our decision tree model for all features in our
# test dataframe:
fullPredictions = model.transform(testDF).cache()
```

## Things I'm not Covering

- `rdd.persist()`
- `createStartingRdd()`
- `df.when()`
- `df.otherwise()`
- `rdd?.zip(rdd?)`
