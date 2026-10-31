"""
Task 2: Data Analytic
Trains an ALS collaborative-filtering recommender on the Philadelphia
restaurant-review matrix and generates top-N recommendations.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


OUTPUT_DIR = "output"  # output for the ratings 
TOP_N = 10 

spark = (
    SparkSession.builder
    .appName("YelpALS")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

ratings = spark.read.parquet(f"{OUTPUT_DIR}/phl_ratings")
restaurants = spark.read.parquet(f"{OUTPUT_DIR}/phl_restaurants")

# ---------------------------------------------------------------------------
# 1. ALS requires integer ids; map the hash-string ids with StringIndexer
# ---------------------------------------------------------------------------
user_idx = StringIndexer(inputCol="user_id", outputCol="userIdx",
                         handleInvalid="skip").fit(ratings)
biz_idx  = StringIndexer(inputCol="business_id", outputCol="bizIdx",
                         handleInvalid="skip").fit(ratings)

df = biz_idx.transform(user_idx.transform(ratings))
df = df.select(
    F.col("userIdx").cast("int"),
    F.col("bizIdx").cast("int"),
    F.col("stars").cast("float").alias("rating"),
)

# ---------------------------------------------------------------------------
# 2. Train / test split + ALS
# ---------------------------------------------------------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)

als = ALS(
    userCol="userIdx",
    itemCol="bizIdx",
    ratingCol="rating",
    rank=20,                    # latent factors 
    regParam=0.1,               # regularization 
    maxIter=10,
    nonnegative=True,
    coldStartStrategy="drop",   # avoid NaN predictions in evaluation
    seed=42,
)
model = als.fit(train)

# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
preds = model.transform(test)
rmse = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
).evaluate(preds)
print(f"\nTest RMSE: {rmse:.4f}")

# ---------------------------------------------------------------------------
# 4. Top-N recommendations, mapped back to readable restaurant names
# ---------------------------------------------------------------------------
recs = model.recommendForAllUsers(TOP_N)

# bizIdx -> business_id  (invert the StringIndexer label mapping)
biz_labels = spark.createDataFrame(
    [(i, lab) for i, lab in enumerate(biz_idx.labels)],
    ["bizIdx", "business_id"],
)
biz_named = biz_labels.join(
    restaurants.select("business_id", "name", "categories"), on="business_id"
)

# explode the recommendation arrays into rows and attach names
exploded = (
    recs.select("userIdx", F.explode("recommendations").alias("rec"))
        .select("userIdx",
                F.col("rec.bizIdx").alias("bizIdx"),
                F.col("rec.rating").alias("predicted"))
        .join(biz_named, on="bizIdx")
)

print("\n--- Sample recommendations ---")
(exploded
 .orderBy("userIdx", F.col("predicted").desc())
 .select("userIdx", "name", "categories", "predicted")
 .show(30, truncate=40))

spark.stop()