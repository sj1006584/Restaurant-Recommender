# Philadelphia Restaurant Recommender (Yelp Open Dataset)

A Spark-based collaborative-filtering recommender that turns the multi-gigabyte
Yelp Open Dataset into personalized restaurant recommendations for the
Philadelphia metropolitan area. The pipeline reduces the raw data to a
user business rating matrix (Task 1) and factorizes it with Spark MLlib's
Alternating Least Squares (ALS) algorithm to produce top-N recommendations
(Task 2).

---

## Project Structure

```
project/
├── data/                       # raw Yelp JSON files (see Data Acquisition)
├── output/                     # generated intermediate data (created by Task 1)
│   ├── phl_ratings/            #   reduced rating matrix (Parquet)
│   └── phl_restaurants/        #   filtered restaurant metadata (Parquet)
├── datareduction.py            # Task 1: data engineering / reduction
├── dataanalysis.py             # Task 2: ALS model + recommendations
└── README.md
```

---

## Requirements

- Python 3.8+
- Apache Spark / PySpark (`pip install pyspark`)
- Java 8 or 11 (required by Spark)
- ~5 GB free disk space for the raw dataset, plus scratch space for Spark

Verify the install:

```bash
python -c "import pyspark; print(pyspark.__version__)"
```

---

## Data Acquisition

This project uses the **Yelp Open Dataset**, which is not redistributed in this
repository and must be downloaded directly from Yelp.

1. **Download.** Go to <https://www.yelp.com/dataset>, accept the dataset license,
   and download the archive. The download is a single compressed file.

2. **Unpack (two stages).** The archive is "double-packaged": the outer download
   contains a single TAR file, which in turn contains the JSON files.

   On Linux / macOS:
   ```bash
   tar -xvf yelp_dataset.tar
   ```
   If `tar` reports it is not a tar file, it may still be gzipped:
   ```bash
   tar -xzvf yelp_dataset.tar
   ```

   On Windows: rename the inner file so it ends in `.tar`, then extract it.

3. **Confirm the files.** After the second extraction you should have:
   ```
   yelp_academic_dataset_business.json
   yelp_academic_dataset_review.json
   yelp_academic_dataset_user.json
   yelp_academic_dataset_checkin.json
   yelp_academic_dataset_tip.json
   ```
   Only `business.json` and `review.json` are used by this project.

   > **Note:** These are newline-delimited JSON files (one JSON object per line),
   > not a single JSON array. Spark reads this format natively. Do **not** open
   > `review.json` in a text editor or load it with `json.load()`  it is several
   > gigabytes.

4. **Place the files.** Put both required files in a `data/` subfolder so the
   default paths in the scripts resolve:
   ```
   project/data/yelp_academic_dataset_business.json
   project/data/yelp_academic_dataset_review.json
   ```
   If you store them elsewhere, edit `BUSINESS_PATH` and `REVIEW_PATH` at the top
   of `datareduction.py` accordingly.

---

## Running the Project

Run the two stages in order from inside the `project/` folder. Task 1 writes the
reduced dataset to `output/`, which Task 2 reads.

### Task 1 — Data Engineering

```bash
spark-submit datareduction.py
```

This filters the data to Philadelphia restaurants, joins reviews, applies
sparsity filtering, prints descriptive statistics, and writes the reduced
rating matrix to `output/phl_ratings/` and `output/phl_restaurants/`.

A successful run ends with:
```
--- FINAL intermediate dataset ---
ratings : ...
Wrote intermediate data to output
```

### Task 2 — Data Analytic

```bash
spark-submit dataanalysis.py
```

This trains the ALS model on the reduced matrix, reports the test-set RMSE, and
prints sample top-N restaurant recommendations with readable names.

---

## Configuration

Adjustable parameters at the top of `datareduction.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `CITY` | `Philadelphia` | Metropolitan area to scope to |
| `MIN_USER_REVIEWS` | `5` | Drop users with fewer than this many reviews |
| `MIN_BIZ_REVIEWS` | `5` | Drop businesses with fewer than this many reviews |

Raising the `MIN_*` thresholds yields a smaller, denser matrix (more active
users and businesses); lowering them keeps more of the long tail.

---

## Troubleshooting

**`No space left on device`** — Spark spills shuffle data to a temp directory.
On some systems (e.g. WSL) `/tmp` is a small RAM-backed filesystem. Point Spark's
scratch directory at a disk with free space:
```bash
spark-submit --conf spark.local.dir=$HOME/spark-tmp datareduction.py
```
(create the folder first with `mkdir -p $HOME/spark-tmp`).

**`OutOfMemoryError` / heartbeat timeouts** — Increase the Spark driver heap:
```bash
spark-submit --driver-memory 8g datareduction.py
```
Leave headroom for the OS; do not allocate all available RAM.


**Zero restaurants returned** — Some Yelp releases ship the business
`categories` field as an array rather than a comma-separated string. Check with
`business.printSchema()`; if it is an array, replace the `rlike` category filter
with an `array_contains`-based filter.

---

## Notes
- I ran this in a WSL with Ubuntu as the distro 
- Intermediate data is stored as Parquet so Task 2 can read it without
  re-parsing the multi-gigabyte JSON on every run.
- The Yelp dataset is subject to Yelp's dataset license; review the terms before
  use or redistribution.
