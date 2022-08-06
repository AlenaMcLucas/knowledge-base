import re
from pyspark import SparkConf, SparkContext

def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())   # '\W+' keeps words only

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("../data/book.txt")
words = input.flatMap(normalizeWords)

# countByValue() the hard way so it remains an RDD we can sort after
wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

# flip the (word, count) pairs, then sort
wordCountsSorted = wordCounts.map(lambda pair: (pair[1], pair[0])).sortByKey()
results = wordCountsSorted.collect()

for result in results:
    count, word = result[0], result[1]
    cleanWord = word.encode('ascii', 'ignore')
    if (cleanWord):
        print(cleanWord.decode() + " " + str(count))
