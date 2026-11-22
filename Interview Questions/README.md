<div align="center">

# 🐼 Pandas Interview Questions for AI/ML Engineers

### 50 questions · Easy → Advanced · Answers with runnable code

![Questions](https://img.shields.io/badge/Questions-50-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x%20%7C%203.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Focus](https://img.shields.io/badge/Focus-AI%2FML%20Engineering-orange?style=for-the-badge)

</div>

---

## 📚 Overview

| Level | Questions | Topics |
|:------|:---------:|:-------|
| 🟢 **Easy** | Q1 – Q15 | Series/DataFrame, indexing, missing values, groupby, merge, I/O |
| 🟡 **Intermediate** | Q16 – Q35 | Feature engineering, time series, reshaping, memory, leakage-safe patterns |
| 🔴 **Advanced** | Q36 – Q50 | Internals, Copy-on-Write, Arrow, point-in-time joins, pipelines, scale |

<details>
<summary><b>📑 Table of Contents</b></summary>

**🟢 Easy**
[Q1](#q1) · [Q2](#q2) · [Q3](#q3) · [Q4](#q4) · [Q5](#q5) · [Q6](#q6) · [Q7](#q7) · [Q8](#q8) · [Q9](#q9) · [Q10](#q10) · [Q11](#q11) · [Q12](#q12) · [Q13](#q13) · [Q14](#q14) · [Q15](#q15)

**🟡 Intermediate**
[Q16](#q16) · [Q17](#q17) · [Q18](#q18) · [Q19](#q19) · [Q20](#q20) · [Q21](#q21) · [Q22](#q22) · [Q23](#q23) · [Q24](#q24) · [Q25](#q25) · [Q26](#q26) · [Q27](#q27) · [Q28](#q28) · [Q29](#q29) · [Q30](#q30) · [Q31](#q31) · [Q32](#q32) · [Q33](#q33) · [Q34](#q34) · [Q35](#q35)

**🔴 Advanced**
[Q36](#q36) · [Q37](#q37) · [Q38](#q38) · [Q39](#q39) · [Q40](#q40) · [Q41](#q41) · [Q42](#q42) · [Q43](#q43) · [Q44](#q44) · [Q45](#q45) · [Q46](#q46) · [Q47](#q47) · [Q48](#q48) · [Q49](#q49) · [Q50](#q50)

</details>

```python
import numpy as np
import pandas as pd
```

---

# 🟢 Easy

![Level](https://img.shields.io/badge/Level-Easy-brightgreen?style=flat-square)

### Q1
**What is the difference between a `Series` and a `DataFrame`?**

<details>
<summary>💡 Answer</summary>

- A **Series** is a 1-D labeled array with a single dtype (one column).
- A **DataFrame** is a 2-D labeled table: a dict-like collection of Series sharing the same index, each column with its own dtype.

```python
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
df = pd.DataFrame({"age": [25, 32], "city": ["Pune", "Delhi"]})
type(df["age"])      # pandas.core.series.Series
type(df[["age"]])    # pandas.core.frame.DataFrame  (double brackets keep 2-D)
```

**ML relevance:** a DataFrame is typically your feature matrix `X`; a Series is your target `y`.
</details>

### Q2
**How do you load data, and which `read_csv` parameters matter most in practice?**

<details>
<summary>💡 Answer</summary>

```python
df = pd.read_csv(
    "data.csv",
    usecols=["id", "age", "income", "signup_date"],  # load only what you need
    dtype={"id": "int32", "income": "float32"},      # control memory
    parse_dates=["signup_date"],                     # datetime at load time
    na_values=["?", "NA", "-"],                      # custom missing markers
    nrows=10_000,                                    # quick sampling while exploring
)
```

Other loaders: `read_parquet`, `read_json`, `read_excel`, `read_sql`.
</details>

### Q3
**How do you quickly inspect a new dataset?**

<details>
<summary>💡 Answer</summary>

```python
df.head()                 # first rows
df.shape                  # (rows, cols)
df.info()                 # dtypes, non-null counts, memory
df.describe(include="all")# summary stats
df.isna().sum()           # missing per column
df.nunique()              # cardinality per column
df.dtypes
```

**Tip:** always check `info()` first. An `object` column that should be numeric usually signals dirty data.
</details>

### Q4
**Explain `loc` vs `iloc`.**

<details>
<summary>💡 Answer</summary>

| | `loc` | `iloc` |
|---|---|---|
| Based on | **labels** | **integer positions** |
| Slice end | **inclusive** | **exclusive** |
| Boolean mask | ✅ | ✅ (array only) |

```python
df.loc[0:2, "age"]      # rows labeled 0,1,2 (inclusive)
df.iloc[0:2, 0]         # rows at positions 0,1 (exclusive)
df.loc[df["age"] > 30, ["age", "city"]]
```
</details>

### Q5
**How do you filter rows by multiple conditions?**

<details>
<summary>💡 Answer</summary>

Use `&`, `|`, `~` with **parentheses** (operator precedence), not `and`/`or`.

```python
df[(df["age"] > 25) & (df["city"] == "Pune")]
df[df["city"].isin(["Pune", "Delhi"])]
df[~df["income"].isna()]
df[df["age"].between(20, 40)]
```

Using `and` raises `ValueError: truth value of a Series is ambiguous`.
</details>

### Q6
**How do you handle missing values?**

<details>
<summary>💡 Answer</summary>

```python
df.isna().mean()                         # fraction missing per column
df.dropna(subset=["target"])             # drop rows missing the label
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna("Unknown")
df["income"] = df["income"].ffill()      # time-ordered data only
```

**ML caveat:** compute fill statistics on the **training set only** and reuse them on validation/test, otherwise you leak information (see Q20).
</details>

### Q7
**How do you find and remove duplicates?**

<details>
<summary>💡 Answer</summary>

```python
df.duplicated().sum()
df.duplicated(subset=["user_id", "date"], keep=False)  # show all copies
df = df.drop_duplicates(subset=["user_id", "date"], keep="last")
```

**ML relevance:** duplicates across train/test splits inflate evaluation metrics, so deduplicate **before** splitting.
</details>

### Q8
**How do you rename columns, drop columns, and change dtypes?**

<details>
<summary>💡 Answer</summary>

```python
df = df.rename(columns={"old": "new"})
df = df.drop(columns=["unused"])
df["age"] = df["age"].astype("int32")
df["price"] = pd.to_numeric(df["price"], errors="coerce")   # bad values -> NaN
df.columns = df.columns.str.lower().str.replace(" ", "_")
```
</details>

### Q9
**How do you sort data and count category frequencies?**

<details>
<summary>💡 Answer</summary>

```python
df.sort_values(["city", "age"], ascending=[True, False])
df["city"].value_counts()
df["city"].value_counts(normalize=True)       # proportions
df["label"].value_counts(dropna=False)        # include NaN: class balance check
```
</details>

### Q10
**What's the difference between `apply`, `map`, and `DataFrame.map` (formerly `applymap`)?**

<details>
<summary>💡 Answer</summary>

| Method | Works on | Operates on |
|---|---|---|
| `Series.map` | Series | each element (also accepts dict/Series for lookup) |
| `Series.apply` | Series | each element via a function |
| `DataFrame.apply` | DataFrame | each **column** (`axis=0`) or **row** (`axis=1`) |
| `DataFrame.map` | DataFrame | each element (renamed from `applymap` in 2.1) |

```python
df["grade"].map({"A": 4, "B": 3, "C": 2})
df["income"].apply(np.log1p)         # but prefer np.log1p(df["income"])
df[["a", "b"]].map(lambda x: x * 2)
```

Prefer vectorized operations over any of these when possible (see Q27).
</details>

### Q11
**How does `groupby` work?**

<details>
<summary>💡 Answer</summary>

It follows **split → apply → combine**.

```python
df.groupby("city")["income"].mean()
df.groupby("city").agg(avg_income=("income", "mean"),
                       n_users=("user_id", "nunique"))
df.groupby(["city", "gender"])["income"].median()
```
</details>

### Q12
**Compare `concat`, `merge`, and `join`.**

<details>
<summary>💡 Answer</summary>

| Function | Purpose |
|---|---|
| `pd.concat` | **Stack** frames along an axis (rows or columns) |
| `pd.merge` | **SQL-style join** on column(s) |
| `df.join` | Join on **index** (convenience wrapper over merge) |

```python
pd.concat([df_a, df_b], axis=0, ignore_index=True)
pd.merge(users, orders, on="user_id", how="left")
users.set_index("user_id").join(orders.set_index("user_id"))
```
</details>

### Q13
**What does the index do? When do you use `set_index` and `reset_index`?**

<details>
<summary>💡 Answer</summary>

The index is the row label, enabling fast lookups, automatic **alignment** in arithmetic, and joins.

```python
df = df.set_index("user_id")      # column -> index
df = df.reset_index()             # index -> column
df.reset_index(drop=True)         # discard old index (common after filtering/shuffling)
```

**Gotcha:** operations between Series align on index labels, not positions. Mismatched indices produce `NaN`.
</details>

### Q14
**How do you get unique values and their counts?**

<details>
<summary>💡 Answer</summary>

```python
df["city"].unique()          # array of unique values
df["city"].nunique()         # count (excludes NaN by default)
df["city"].nunique(dropna=False)
df["city"].value_counts()    # frequency table
```

Use `nunique()` to spot **high-cardinality** features that need special encoding (Q22, Q46).
</details>

### Q15
**How do you save data, and why is Parquet usually preferred over CSV for ML?**

<details>
<summary>💡 Answer</summary>

```python
df.to_csv("out.csv", index=False)
df.to_parquet("out.parquet", index=False)   # needs pyarrow or fastparquet
```

| | CSV | Parquet |
|---|---|---|
| Preserves dtypes | ❌ | ✅ |
| Compression | ❌ | ✅ (snappy/zstd) |
| Columnar reads | ❌ | ✅ (`columns=[...]`) |
| Speed on large data | slow | fast |
</details>

---

# 🟡 Intermediate

![Level](https://img.shields.io/badge/Level-Intermediate-yellow?style=flat-square)

### Q16
**Explain merge types and how to prevent silent row explosions.**

<details>
<summary>💡 Answer</summary>

`how` can be `inner`, `left`, `right`, `outer`, or `cross`.

A many-to-many merge silently multiplies rows. Guard against it with `validate` and inspect with `indicator`:

```python
out = pd.merge(
    orders, users,
    on="user_id",
    how="left",
    validate="many_to_one",   # raises MergeError if users has duplicate user_id
    indicator=True,           # adds _merge column: left_only / both
)
out["_merge"].value_counts()
```

**Interview tip:** always assert row counts after a join in a training pipeline.
</details>

### Q17
**What's the difference between `agg`, `transform`, and `apply` in a groupby?**

<details>
<summary>💡 Answer</summary>

| Method | Output shape |
|---|---|
| `agg` | one row **per group** |
| `transform` | **same length** as the input (broadcast back) |
| `apply` | flexible, but slowest |

```python
# group-wise imputation
df["income"] = df["income"].fillna(df.groupby("city")["income"].transform("median"))

# group-relative feature
df["income_vs_city"] = df["income"] / df.groupby("city")["income"].transform("mean")
```
</details>

### Q18
**When do you use `pivot_table` vs `groupby`, and what is `crosstab`?**

<details>
<summary>💡 Answer</summary>

`pivot_table` is a `groupby` + reshape in one call. It produces a wide matrix.

```python
pd.pivot_table(df, index="city", columns="gender",
               values="income", aggfunc="mean", fill_value=0)

pd.crosstab(df["city"], df["label"], normalize="index")  # class share per city
```

Use `crosstab` for quick categorical vs. target analysis.
</details>

### Q19
**Explain `melt`, `stack`, and `unstack`. When do you go wide-to-long?**

<details>
<summary>💡 Answer</summary>

```python
long = df.melt(id_vars=["id"], value_vars=["jan", "feb", "mar"],
               var_name="month", value_name="sales")
wide = long.pivot(index="id", columns="month", values="sales")
```

- `melt` : wide → long (needed for seaborn/plotly, tidy data)
- `pivot` / `unstack` : long → wide (feature matrix per entity)
- `stack` : columns → inner index level
</details>

### Q20
**How do you avoid data leakage during imputation and scaling in pandas?**

<details>
<summary>💡 Answer</summary>

**Split first, then compute statistics on train only.**

```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)

median = train["income"].median()            # learned from train only
train["income"] = train["income"].fillna(median)
test["income"]  = test["income"].fillna(median)   # reuse train statistic
```

Better still, put imputers/scalers in an sklearn `Pipeline` so `fit` only ever sees training folds (Q43).
</details>

### Q21
**How do you detect outliers with pandas?**

<details>
<summary>💡 Answer</summary>

**IQR method**

```python
q1, q3 = df["income"].quantile([0.25, 0.75])
iqr = q3 - q1
mask = df["income"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
clean = df[mask]
```

**Z-score:** `(x - mean) / std`, assuming roughly normal data.

**Winsorizing:** `df["income"].clip(lower=q_low, upper=q_high)` keeps rows but caps extremes.

Domain knowledge matters. Fraud and anomaly tasks treat outliers as the signal.
</details>

### Q22
**How do you encode categorical variables, and what goes wrong with `get_dummies` in production?**

<details>
<summary>💡 Answer</summary>

```python
X = pd.get_dummies(df, columns=["city"], drop_first=True)
```

**Problem:** train and inference data may contain different categories, producing mismatched columns.

```python
train_cols = X_train.columns
X_new = pd.get_dummies(new_df).reindex(columns=train_cols, fill_value=0)
```

More robust: sklearn `OneHotEncoder(handle_unknown="ignore")`, or the pandas `category` dtype with a fixed category list.
</details>

### Q23
**How do you work with datetime data and extract features?**

<details>
<summary>💡 Answer</summary>

```python
df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
df["hour"]       = df["ts"].dt.hour
df["dayofweek"]  = df["ts"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6])

# cyclical encoding for models that can't learn wrap-around
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df.set_index("ts").resample("W")["sales"].sum()   # weekly aggregation
```
</details>

### Q24
**How do you build lag and rolling features without leaking the future?**

<details>
<summary>💡 Answer</summary>

Always `shift` **before** rolling so the current row's target is excluded.

```python
df = df.sort_values(["store_id", "date"])
g = df.groupby("store_id")["sales"]

df["lag_1"]       = g.shift(1)
df["lag_7"]       = g.shift(7)
df["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
df["ewm_7"]       = g.transform(lambda s: s.shift(1).ewm(span=7).mean())
```
</details>

### Q25
**What is `SettingWithCopyWarning` and how do you fix it?**

<details>
<summary>💡 Answer</summary>

It appears when you assign into something that **may be a copy** of another frame (chained indexing).

```python
# ❌ ambiguous
df[df["age"] > 30]["flag"] = 1

# ✅ explicit
df.loc[df["age"] > 30, "flag"] = 1

# ✅ when working on a subset
sub = df[df["age"] > 30].copy()
sub["flag"] = 1
```

Under Copy-on-Write (Q37), chained assignment simply never modifies the original.
</details>

### Q26
**What is the difference between a view and a copy?**

<details>
<summary>💡 Answer</summary>

- A **view** shares memory with the original. Modifying it may change the parent.
- A **copy** has its own memory.

Whether an operation returns a view or a copy historically depended on internal memory layout, so it was hard to predict. This unpredictability is why Copy-on-Write was introduced: **every derived object behaves as a copy**, and the data is physically copied lazily only on write.
</details>

### Q27
**Why is vectorization faster than `apply` or `iterrows`? Show the progression.**

<details>
<summary>💡 Answer</summary>

Vectorized ops run in compiled C/NumPy loops over contiguous arrays. `apply(axis=1)` and `iterrows` call Python code per row and build a Series object each time.

```python
# 🔴 slowest
for _, row in df.iterrows():
    ...

# 🟡 slow
df["bmi"] = df.apply(lambda r: r["w"] / r["h"] ** 2, axis=1)

# 🟢 fast
df["bmi"] = df["w"] / df["h"] ** 2

# 🟢 conditional logic
df["tier"] = np.where(df["income"] > 1e5, "high", "low")
df["band"] = np.select([df.age < 18, df.age < 60], ["minor", "adult"], "senior")
```

Typically **one to two orders of magnitude** faster. Benchmark on your own data.
</details>

### Q28
**How do you reduce a DataFrame's memory footprint?**

<details>
<summary>💡 Answer</summary>

```python
df.memory_usage(deep=True).sum() / 1e6          # MB

df["age"]    = pd.to_numeric(df["age"], downcast="integer")
df["price"]  = df["price"].astype("float32")
df["city"]   = df["city"].astype("category")    # big win for low-cardinality strings
```

Also: load only needed columns, drop unused columns early, and store in Parquet.
</details>

### Q29
**How do you process a file that doesn't fit in memory?**

<details>
<summary>💡 Answer</summary>

```python
chunks = pd.read_csv("big.csv", chunksize=500_000, usecols=["user", "amount"])
partials = [c.groupby("user")["amount"].sum() for c in chunks]
total = pd.concat(partials).groupby(level=0).sum()
```

Alternatives: Parquet with column pruning, **Dask**, **Polars** (lazy), **DuckDB**, or **PySpark** (Q42).
</details>

### Q30
**How do you clean text columns with the `.str` accessor?**

<details>
<summary>💡 Answer</summary>

```python
s = df["name"]
s.str.strip().str.lower()
s.str.replace(r"[^a-z0-9 ]", "", regex=True)
s.str.contains("pro", case=False, na=False)
s.str.extract(r"(\d{4})-(\d{2})")             # regex groups -> columns
s.str.split(",", expand=True)
```

`na=False` avoids `NaN` poisoning boolean masks.
</details>

### Q31
**What are `query()` and `eval()`? When are they useful?**

<details>
<summary>💡 Answer</summary>

```python
threshold = 30
df.query("age > @threshold and city in ['Pune', 'Delhi']")
df.eval("bmi = weight / height ** 2", inplace=True)
```

They improve readability and can reduce temporary arrays on large frames. For small data the speed difference is negligible.
</details>

### Q32
**What is a `MultiIndex` and when would you use one?**

<details>
<summary>💡 Answer</summary>

A hierarchical index with multiple levels, produced naturally by multi-key `groupby`.

```python
g = df.groupby(["city", "gender"])["income"].mean()
g.loc["Pune"]                  # all genders in Pune
g.xs("F", level="gender")      # cross-section
g.reset_index()                # flatten back to columns
```

Flatten before feeding to ML libraries, which expect plain column names.
</details>

### Q33
**How should you split time-ordered data for ML?**

<details>
<summary>💡 Answer</summary>

Never shuffle. Train on the past and evaluate on the future.

```python
df = df.sort_values("date")
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]

# or sklearn
from sklearn.model_selection import TimeSeriesSplit
for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(df):
    ...
```

Random splits on temporal data leak future patterns and overestimate performance.
</details>

### Q34
**Explain `pd.cut` vs `pd.qcut`.**

<details>
<summary>💡 Answer</summary>

| | `cut` | `qcut` |
|---|---|---|
| Bins | fixed-width / custom edges | equal-**frequency** quantiles |
| Bin sizes | uneven counts | roughly equal counts |

```python
pd.cut(df["age"], bins=[0, 18, 35, 60, 120], labels=["teen", "young", "mid", "senior"])
pd.qcut(df["income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
```

Learn bin edges from train and apply them to test via `pd.cut(test, bins=edges)`.
</details>

### Q35
**How does pandas represent missing values? Compare `np.nan`, `None`, and `pd.NA`.**

<details>
<summary>💡 Answer</summary>

- `np.nan` is a float, so an integer column with NaN becomes `float64`. `NaN != NaN`.
- `None` is a Python object, usually converted to NaN in numeric columns.
- `pd.NA` is the **nullable** missing marker used with `Int64`, `string`, `boolean`, and Arrow dtypes, and it propagates through logical ops.

```python
s = pd.Series([1, 2, None], dtype="Int64")   # stays integer
s.isna()
df.convert_dtypes()                          # infer nullable dtypes
```

Always test with `isna()`, never `== np.nan`.
</details>

---

# 🔴 Advanced

![Level](https://img.shields.io/badge/Level-Advanced-red?style=flat-square)

### Q36
**How does pandas store data internally, and why does that affect performance?**

<details>
<summary>💡 Answer</summary>

A DataFrame is a **column-oriented** collection of arrays managed by a block manager that groups same-dtype columns into 2-D NumPy blocks.

Consequences:
- Column operations are fast (contiguous memory); row-wise iteration is slow.
- Mixed-dtype row access creates object arrays and copies.
- `object` dtype stores pointers to Python objects, which is memory hungry and slow. Use `category`, `string[pyarrow]`, or numeric dtypes.
- Adding columns one at a time can fragment blocks (`PerformanceWarning`). Build columns in a dict and use `pd.concat(axis=1)` once.
</details>

### Q37
**Explain Copy-on-Write (CoW) and how it changes pandas code.**

<details>
<summary>💡 Answer</summary>

With CoW, any object derived from another behaves like an independent copy. Data is shared lazily and copied only when one of them is modified.

```python
pd.options.mode.copy_on_write = True      # opt-in in pandas 2.x; default in 3.0

df2 = df["a"]
df2.iloc[0] = 99      # df is NOT modified
```

**Impact:**
- Chained assignment (`df["a"][0] = 1`, `df["a"].fillna(0, inplace=True)`) no longer updates `df`. Use `df.loc[...]` or `df["a"] = df["a"].fillna(0)`.
- `SettingWithCopyWarning` disappears.
- Fewer defensive `.copy()` calls, lower memory use, more predictable behavior.
</details>

### Q38
**What is the PyArrow backend and when would you use it?**

<details>
<summary>💡 Answer</summary>

Pandas can back columns with **Apache Arrow** arrays instead of NumPy.

```python
df = pd.read_parquet("data.parquet", dtype_backend="pyarrow")
df["name"] = df["name"].astype("string[pyarrow]")
```

**Benefits:** native nullable types, much faster and smaller string columns, zero-copy interop with Parquet, Polars, DuckDB, and Spark.

**Trade-offs:** some operations are not yet optimized for Arrow dtypes, and a few libraries still expect NumPy arrays, so convert with `.to_numpy()` before model input.
</details>

### Q39
**Implement leakage-safe target encoding for a high-cardinality feature.**

<details>
<summary>💡 Answer</summary>

Naive group-mean encoding leaks the label into its own feature. Use **out-of-fold** encoding with smoothing.

```python
from sklearn.model_selection import KFold

def target_encode_oof(df, col, target, n_splits=5, smooth=20, seed=42):
    prior = df[target].mean()
    out = pd.Series(np.nan, index=df.index)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(df):
        stats = df.iloc[tr].groupby(col)[target].agg(["mean", "count"])
        enc = (stats["count"] * stats["mean"] + smooth * prior) / (stats["count"] + smooth)
        out.iloc[va] = df.iloc[va][col].map(enc).to_numpy()
    return out.fillna(prior)

df["city_te"] = target_encode_oof(df, "city", "y")
```

At inference, use encodings computed from the **full training set**. sklearn's `TargetEncoder` implements the same cross-fitting idea.
</details>

### Q40
**What is `merge_asof` and why does it matter for point-in-time correctness?**

<details>
<summary>💡 Answer</summary>

It joins each left row to the **nearest previous** right row by a sorted key. This is the standard way to attach "latest known" features without peeking into the future.

```python
feat = pd.merge_asof(
    events.sort_values("ts"),
    prices.sort_values("ts"),
    on="ts",
    by="symbol",
    direction="backward",
    tolerance=pd.Timedelta("1D"),
)
```

Using a plain merge on dates can attach feature values that were not yet available at prediction time, a classic cause of unrealistic backtests.
</details>

### Q41
**Build per-entity rolling statistics efficiently and correctly.**

<details>
<summary>💡 Answer</summary>

```python
df = df.sort_values(["user_id", "ts"])

# time-based window, per user, excluding the current row
roll = (
    df.set_index("ts")
      .groupby("user_id")["amount"]
      .rolling("7D", closed="left")
      .agg(["mean", "sum", "count"])
      .reset_index()
)
```

`closed="left"` excludes the current observation, preventing target/feature leakage. Prefer built-in rolling aggregations over `transform(lambda ...)` for speed.
</details>

### Q42
**Pandas vs Polars vs Dask vs PySpark vs DuckDB: how do you choose?**

<details>
<summary>💡 Answer</summary>

| Tool | Best for |
|---|---|
| **pandas** | Data that fits in RAM, rich ecosystem, EDA, sklearn interop |
| **Polars** | Fast single-machine, lazy evaluation, multi-threaded, larger-than-RAM streaming |
| **DuckDB** | SQL-first analytics on Parquet/CSV with very low overhead |
| **Dask** | Pandas-like API across cores or a cluster |
| **PySpark** | Truly distributed, TB-scale pipelines |

Rule of thumb: prototype in pandas, scale with the tool that matches your data size and team skills. Interop through Arrow makes switching cheap.
</details>

### Q43
**How do you integrate pandas with scikit-learn pipelines while keeping column names?**

<details>
<summary>💡 Answer</summary>

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

num = ["age", "income"]
cat = ["city", "gender"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
]).set_output(transform="pandas")        # returns a DataFrame

model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42))])
model.fit(X_train, y_train)
```

This guarantees fit-on-train-only semantics, reproducible preprocessing, and one serializable artifact.
</details>

### Q44
**What is method chaining and how does `pipe` help write clean, testable pipelines?**

<details>
<summary>💡 Answer</summary>

```python
def add_features(d):
    return d.assign(bmi=d["weight"] / d["height"] ** 2,
                    log_income=np.log1p(d["income"]))

def clip_outliers(d, col, q=0.99):
    return d.assign(**{col: d[col].clip(upper=d[col].quantile(q))})

clean = (
    raw
    .rename(columns=str.lower)
    .drop_duplicates()
    .query("age >= 18")
    .pipe(add_features)
    .pipe(clip_outliers, col="income")
    .reset_index(drop=True)
)
```

Benefits: no mutation of intermediate state, each step is a unit-testable function, and the flow reads top to bottom.
</details>

### Q45
**How do you make large joins faster and more memory-safe?**

<details>
<summary>💡 Answer</summary>

- Ensure **key dtypes match** (a mismatched `int64` vs `object` key is slow and may silently fail to match).
- Convert repeated string keys to `category`, or map them to integer IDs.
- **Select only needed columns** from the right table before merging.
- Deduplicate the right side and use `validate="many_to_one"`.
- Join on a sorted, unique **index** when possible.
- Filter rows **before** the join, not after.

```python
right = right[["user_id", "segment"]].drop_duplicates("user_id")
out = left.merge(right, on="user_id", how="left", validate="many_to_one")
```
</details>

### Q46
**How do you handle high-cardinality categorical features and sparse data?**

<details>
<summary>💡 Answer</summary>

- **Frequency encoding:** `df["city"].map(df["city"].value_counts(normalize=True))`
- **Rare-category grouping:** collapse categories below a threshold into `"other"`.
- **Target encoding** (Q39) or **hashing** for very large vocabularies.
- **Sparse storage** for wide one-hot matrices:

```python
X = pd.get_dummies(df["city"], sparse=True)
X.sparse.density
```

Avoid dense one-hot on columns with thousands of levels. It explodes memory and hurts tree and linear models alike.
</details>

### Q47
**A pandas script is slow. How do you profile and fix it?**

<details>
<summary>💡 Answer</summary>

**Measure first**

```python
%timeit df["a"].sum()
%prun -l 10 run_pipeline(df)          # function-level profile
df.memory_usage(deep=True)
```

Tools: `line_profiler`, `memory_profiler`, `tracemalloc`.

**Common culprits and fixes**

| Symptom | Fix |
|---|---|
| `iterrows` / `apply(axis=1)` | vectorize, `np.where`, `np.select` |
| Many `object` columns | `category` or Arrow strings |
| Repeated `concat` in a loop | collect in a list, concat once |
| Repeated groupby on the same key | compute once, reuse the groupby object |
| Huge intermediate frames | chain with `query`/`eval`, drop columns early |
| Heavy custom function | Numba, or move to Polars/DuckDB |
</details>

### Q48
**How do you validate data quality in a production ML pipeline?**

<details>
<summary>💡 Answer</summary>

Fail fast on schema and distribution problems before they reach the model.

```python
import pandera.pandas as pa

schema = pa.DataFrameSchema({
    "age":    pa.Column(int,   pa.Check.between(0, 120)),
    "income": pa.Column(float, pa.Check.ge(0), nullable=True),
    "city":   pa.Column(str,   pa.Check.isin(KNOWN_CITIES), nullable=False),
})
validated = schema.validate(df, lazy=True)     # collect all errors at once
```

Also: assert row counts after joins, check null rates against training baselines, and monitor **feature drift** (PSI/KS tests). Tools such as Great Expectations and Evidently cover this at scale.
</details>

### Q49
**What are the trade-offs between CSV, Parquet, Feather, and pickle for persisting DataFrames?**

<details>
<summary>💡 Answer</summary>

| Format | Dtypes kept | Speed | Notes |
|---|:---:|:---:|---|
| CSV | ❌ | slow | human-readable, universal |
| **Parquet** | ✅ | fast | compressed, columnar, **best default** |
| Feather | ✅ | very fast | great for short-lived intermediates |
| Pickle | ✅ | fast | ⚠️ **unsafe to load untrusted files** and fragile across versions |

For reproducibility, version the data (DVC, lakeFS, or dated paths) and store the schema alongside the artifact.
</details>

### Q50
**Scenario: design a feature pipeline for a 50M-row transaction dataset on a 16 GB machine.**

<details>
<summary>💡 Answer</summary>

1. **Ingest efficiently:** convert raw CSV to partitioned Parquet once; read only required columns with explicit compact dtypes (`float32`, `category`, Arrow strings).
2. **Process in partitions:** iterate by date or user-hash shard, or use Polars (lazy) / DuckDB / Dask for the heavy aggregations.
3. **Compute aggregates once:** per-user and per-merchant features via `groupby().agg()`, then **join back** instead of per-row `apply`.
4. **Prevent leakage:** time-based split, `shift` before rolling, `merge_asof` for point-in-time features, target encoding out-of-fold.
5. **Validate:** schema checks, row-count assertions after joins, null-rate comparisons with training baselines.
6. **Persist and version:** write feature tables to Parquet with a schema and run ID; keep the preprocessing in a fitted sklearn `Pipeline` for reproducibility at inference.
7. **Verify:** profile memory at each stage and unit-test feature functions on small fixtures.

**What interviewers look for:** a clear plan covering memory, correctness (no leakage), and reproducibility, not just speed tricks.
</details>

---

<div align="center">

### ⭐ If this helped your prep, star the repo and share it

![Made for](https://img.shields.io/badge/Made%20for-AI%2FML%20Interviews-purple?style=flat-square)

</div>
