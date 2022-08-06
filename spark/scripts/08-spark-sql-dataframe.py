from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# tells spark to try to infer schema
# returns dataframe with a schema
people = spark.read.option("header", "true").option("inferSchema", "true") \
    .csv("../data/fakefriends-header.csv")

print("Here is our inferred schema:")
people.printSchema()

print("Let's display the name column:")
people.select("name").show()

print("Filter out anyone over 21:")
people.filter(people.age < 21).show()   # easier syntax using schema

print("Group by age")
people.groupBy("age").count().show()

print("Make everyone 10 years older:")
people.select(people.name, people.age + 10).show()   # creates a new column of age + 10 years

spark.stop()
