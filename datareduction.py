"""
Task 1: Data Engineering
Reduces the raw Yelp dataset to a Philadelphia restaurant-review matrix
suitable for collaborative filtering, and produces descriptive statistics.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Data set paths
BUSINESS_PATH = "data/yelp_dataset/yelp_academic_dataset_business.json"
REVIEW_PATH   = "data/yelp_dataset/yelp_academic_dataset_review.json"
OUTPUT_DIR    = "output"

CITY = "Philadelphia"
MIN_USER_REVIEWS = 5
MIN_BIZ_REVIEWS  = 5


spark = (
    SparkSession.builder
    .appName("YelpDataEngineering")
    .config("spark.driver.memory", "8g")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# 1. Businesses -> Philadelphia restaurants
# ---------------------------------------------------------------------------
business = spark.read.json(BUSINESS_PATH)

restaurants = (
    business
    .filter(F.col("categories").isNotNull())
    .filter(F.col("categories").rlike("(?i)Restaurants|Food"))
    .filter(F.col("city") == CITY)
    .select("business_id", "name", "city", "stars", "review_count", "categories")
)
restaurants.cache()
print(f"[businesses] Philadelphia restaurants kept: {restaurants.count()}")

# ---------------------------------------------------------------------------
# 2. Reviews -> keep only reviews of those restaurants
# ---------------------------------------------------------------------------
reviews = spark.read.json(REVIEW_PATH).select(
    "user_id", "business_id", "stars", "text"
)
phl = reviews.join(restaurants.select("business_id"), on="business_id", how="inner")

# ---------------------------------------------------------------------------
# 3. Descriptive stats that need text — computed ONCE, before dropping text
# ---------------------------------------------------------------------------
print("--- Review text length (chars) ---")
phl.select(F.length("text").alias("len")).describe("len").show()

# Drop text now: the recommender never uses it. Everything downstream is tiny.
phl_slim = phl.select("user_id", "business_id", "stars").cache()
n_reviews_raw = phl_slim.count()
print(f"[reviews] restaurant reviews before sparsity filtering: {n_reviews_raw}")

print("\n--- Rating distribution ---")
phl_slim.groupBy("stars").count().orderBy("stars").show()

print("--- Reviews per user (pre-filter) ---")
phl_slim.groupBy("user_id").count().describe("count").show()

print("--- Reviews per business (pre-filter) ---")
phl_slim.groupBy("business_id").count().describe("count").show()

# ---------------------------------------------------------------------------
# 4. Sparsity filtering via window counts (single pass, no iterative joins).
#    A few passes are still needed because removing users can drop a business
#    below threshold; but each pass is a cheap window count, not a re-join.
# ---------------------------------------------------------------------------
def sparsity_pass(df):
    uw = Window.partitionBy("user_id")
    bw = Window.partitionBy("business_id")
    return (
        df.withColumn("u_cnt", F.count("*").over(uw))
          .withColumn("b_cnt", F.count("*").over(bw))
          .filter((F.col("u_cnt") >= MIN_USER_REVIEWS) &
                  (F.col("b_cnt") >= MIN_BIZ_REVIEWS))
          .drop("u_cnt", "b_cnt")
    )

df = phl_slim
prev = -1
while True:
    df = sparsity_pass(df).cache()
    cur = df.count()
    if cur == prev:
        break
    prev = cur

ratings = df.select("user_id", "business_id", "stars")
ratings.cache()

n_ratings = ratings.count()
n_users   = ratings.select("user_id").distinct().count()
n_biz     = ratings.select("business_id").distinct().count()
density   = n_ratings / (n_users * n_biz) if n_users and n_biz else 0

print("\n--- FINAL intermediate dataset ---")
print(f"ratings : {n_ratings}")
print(f"users   : {n_users}")
print(f"biz     : {n_biz}")
print(f"matrix density : {density:.6f}  ({density*100:.4f}% filled)")
print(f"reduced from {n_reviews_raw} -> {n_ratings} restaurant reviews")

# ---------------------------------------------------------------------------
# 5. Persist for Task 2
# ---------------------------------------------------------------------------
ratings.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/phl_ratings")
restaurants.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/phl_restaurants")
print(f"\nWrote intermediate data to {OUTPUT_DIR}")

spark.stop()