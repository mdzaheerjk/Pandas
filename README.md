# Pandas Complete Reference Notes
## Data Wrangling, Cleaning & Machine Learning Prep

---

## What is Pandas?

Pandas is the primary Python library for tabular data manipulation. It provides:
- **Series** — a labelled 1-D array
- **DataFrame** — a labelled 2-D table (the workhorse)
- Reading/writing dozens of file formats
- Powerful grouping, reshaping, merging, time-series, and statistical tools

```bash
pip install pandas
```

```python
import pandas as pd
import numpy as np
```

---

## Core Data Structures

### Series

```python
s = pd.Series([10, 20, 30])              # index auto-assigned 0,1,2
s = pd.Series([10, 20, 30], index=["a","b","c"])
s = pd.Series({"a": 10, "b": 20})        # from dict

s["a"]           # 10
s[["a","c"]]     # fancy indexing
s[s > 15]        # boolean mask
s.values         # numpy array
s.index          # Index object
s.dtype
s.name = "score"
```

### DataFrame

```python
df = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol"],
    "age":   [25, 30, 35],
    "score": [90.5, 85.0, 92.3]
})

df = pd.DataFrame([[1,2],[3,4]], columns=["A","B"], index=["r1","r2"])
df = pd.DataFrame(records_list)          # list of dicts
```

### Key attributes

```python
df.shape          # (rows, cols)
df.dtypes         # dtype per column
df.columns        # column labels
df.index          # row labels
df.values         # underlying numpy array
df.info()         # dtypes + non-null counts + memory
df.describe()     # stats for numeric columns
df.describe(include="all")   # includes object/category cols
df.head(5)        # first 5 rows
df.tail(5)        # last 5 rows
df.sample(5)      # 5 random rows
df.size           # total elements
df.ndim           # always 2
df.memory_usage(deep=True)
```

---

## Reading & Writing Data

### Reading

```python
# CSV
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv",
    sep=";",                    # delimiter
    header=0,                   # row to use as column names
    names=["a","b","c"],        # override column names
    index_col="id",             # set column as index
    usecols=["a","b"],          # load only these columns
    dtype={"age": np.int32},    # force dtypes
    nrows=1000,                 # read first N rows
    skiprows=[1,2],             # skip row indices
    na_values=["NA","?","-"],   # treat as NaN
    parse_dates=["date"],       # parse column as datetime
    chunksize=10000             # iterator over chunks (big files)
)

# Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_excel("data.xlsx", sheet_name=0)         # by index
all_sheets = pd.read_excel("data.xlsx", sheet_name=None)  # dict of dfs

# JSON
df = pd.read_json("data.json")
df = pd.read_json("data.json", orient="records")

# SQL
import sqlalchemy
engine = sqlalchemy.create_engine("sqlite:///db.sqlite")
df = pd.read_sql("SELECT * FROM table", engine)
df = pd.read_sql_table("table_name", engine)
df = pd.read_sql_query("SELECT * FROM t WHERE id > 10", engine)

# Parquet (fast columnar format)
df = pd.read_parquet("data.parquet")

# Others
df = pd.read_feather("data.feather")
df = pd.read_clipboard()
df = pd.read_html("https://example.com/table")[0]   # scrape tables
```

### Writing

```python
df.to_csv("out.csv", index=False)
df.to_csv("out.csv", sep="\t", float_format="%.2f")
df.to_excel("out.xlsx", sheet_name="Results", index=False)
df.to_json("out.json", orient="records", indent=2)
df.to_parquet("out.parquet", compression="snappy")
df.to_sql("table_name", engine, if_exists="replace", index=False)
df.to_markdown()      # returns markdown string
df.to_string()        # plain text
df.to_dict()          # dict of dicts
df.to_dict("records") # list of dicts
df.to_numpy()         # numpy array
```

---

## Indexing & Selection

### `.loc` — label-based

```python
df.loc[0]                      # row with label 0
df.loc[0, "name"]              # single value
df.loc[0:2, "name":"score"]    # inclusive slice by label
df.loc[[0,2], ["name","age"]]  # fancy selection
df.loc[df["age"] > 28]         # boolean mask
df.loc[df["age"] > 28, "name"] # filtered, single column
```

### `.iloc` — integer position-based

```python
df.iloc[0]                     # first row
df.iloc[0, 1]                  # row 0, col 1
df.iloc[0:3, 0:2]              # exclusive slice by position
df.iloc[[0,2], [0,1]]          # list of positions
df.iloc[-1]                    # last row
```

### Column access

```python
df["name"]                     # Series (preferred)
df[["name","age"]]             # DataFrame (list of cols)
df.name                        # attribute access (avoid: breaks on spaces, clashes)
```

### Boolean indexing

```python
df[df["age"] > 25]
df[(df["age"] > 25) & (df["score"] > 85)]   # & | ~ (not and/or)
df[df["name"].isin(["Alice","Bob"])]
df[~df["name"].isin(["Carol"])]
df[df["score"].between(85, 92)]
df[df["name"].str.startswith("A")]
```

### `.at` and `.iat` — fast scalar access

```python
df.at[0, "name"]               # by label — faster than .loc for single value
df.iat[0, 1]                   # by position — faster than .iloc for single value
df.at[0, "name"] = "Alicia"    # fast scalar assignment
```

### Setting values

```python
df.loc[df["score"] > 90, "grade"] = "A"
df.iloc[0, 1] = 26
df["age"] = df["age"] + 1           # update whole column
df.loc[:, "score"] = df["score"] * 1.1
```

---

## DataFrame Operations

### Adding & removing columns

```python
df["bonus"] = df["score"] * 0.1            # new column
df["full_name"] = df["first"] + " " + df["last"]
df.insert(1, "rank", [1,2,3])              # insert at position
df = df.assign(tax=df["score"]*0.2, net=lambda x: x["score"]-x["tax"])

df.drop("bonus", axis=1)                   # drop column (returns copy)
df.drop(["bonus","rank"], axis=1, inplace=True)
df.pop("bonus")                            # drop & return column
del df["bonus"]                            # in-place delete
```

### Adding & removing rows

```python
new_row = pd.DataFrame([{"name":"Dave","age":28,"score":88}])
df = pd.concat([df, new_row], ignore_index=True)

df.drop(0)                                 # drop row with label 0
df.drop([0,2])                             # drop multiple rows
df = df.reset_index(drop=True)             # reset after drops
```

### Renaming

```python
df.rename(columns={"name":"full_name", "score":"points"})
df.rename(index={0:"first_row"})
df.columns = ["A","B","C"]                 # rename all at once
df.add_prefix("col_")                      # "col_name", "col_age", ...
df.add_suffix("_2024")
```

### Sorting

```python
df.sort_values("age")
df.sort_values("age", ascending=False)
df.sort_values(["age","score"], ascending=[True,False])
df.sort_index()                            # sort by row index
df.sort_index(axis=1)                      # sort columns alphabetically
df.nlargest(3, "score")                    # top 3 by score
df.nsmallest(3, "age")                     # bottom 3 by age
```

### Copying

```python
df2 = df.copy()       # deep copy — independent
df2 = df.copy(deep=False)   # shallow — shares data
```

---

## Missing Data (Data Cleaning — Core Topic)

### Detecting

```python
df.isna()                          # boolean DataFrame — True where NaN
df.isnull()                        # alias for isna()
df.notna()                         # inverse
df.isna().sum()                    # count missing per column
df.isna().sum() / len(df) * 100    # % missing per column
df.isna().any()                    # True if any missing per column
df.isna().all()                    # True if all missing per column
df.isna().sum().sum()              # total missing in whole df

# Visual overview
print(df.info())                   # shows non-null counts
df.isna().mean().sort_values(ascending=False)  # sorted % missing
```

### Dropping

```python
df.dropna()                        # drop rows with ANY NaN
df.dropna(how="all")               # drop rows where ALL are NaN
df.dropna(axis=1)                  # drop columns with any NaN
df.dropna(subset=["age","score"])  # only consider these cols for dropping
df.dropna(thresh=3)                # keep rows with at least 3 non-NaN values
```

### Filling

```python
df.fillna(0)                       # fill all NaN with 0
df.fillna({"age": 0, "score": df["score"].mean()})   # per-column fill

# Forward fill (carry last valid value forward)
df.fillna(method="ffill")          # deprecated in 2.x → use:
df.ffill()
df["col"].ffill()

# Backward fill
df.bfill()

# Fill with stats
df["age"].fillna(df["age"].mean())
df["age"].fillna(df["age"].median())
df["city"].fillna(df["city"].mode()[0])   # most frequent value

# Interpolation
df["price"].interpolate()                 # linear by default
df["price"].interpolate(method="time")    # for time-indexed data
df["price"].interpolate(method="polynomial", order=2)
```

### Replacing

```python
df.replace(-999, np.nan)                  # replace sentinel with NaN
df.replace(["-","?","NA"], np.nan)        # multiple values → NaN
df.replace({"gender": {"M":"Male","F":"Female"}})  # per-column dict
df["score"].replace({0: np.nan})
```

---

## Data Types & Conversion

```python
df.dtypes                              # dtype per column
df["age"].dtype

# Casting
df["age"] = df["age"].astype(int)
df["score"] = df["score"].astype(float)
df["flag"] = df["flag"].astype(bool)
df["name"] = df["name"].astype("string")   # pandas StringDtype (nullable)
df["cat"] = df["cat"].astype("category")   # memory-efficient for low cardinality

# Numeric coercion
pd.to_numeric(df["col"])                   # raises on non-numeric
pd.to_numeric(df["col"], errors="coerce")  # non-numeric → NaN
pd.to_numeric(df["col"], errors="ignore")  # leave non-numeric as-is
df["col"].apply(pd.to_numeric, errors="coerce")

# Datetime
pd.to_datetime(df["date"])
pd.to_datetime(df["date"], format="%Y-%m-%d")
pd.to_datetime(df["date"], errors="coerce")    # bad dates → NaT
pd.to_datetime(df[["year","month","day"]])      # from components

# Infer better dtypes automatically
df = df.convert_dtypes()     # converts to best nullable dtypes
df = df.infer_objects()      # tries to infer object columns
```

---

## Duplicates

```python
df.duplicated()                            # bool Series — True for duplicates
df.duplicated(subset=["name","age"])       # only these cols define duplicate
df.duplicated(keep="first")               # default — mark all but first
df.duplicated(keep="last")                # mark all but last
df.duplicated(keep=False)                 # mark ALL duplicates

df.duplicated().sum()                     # count duplicate rows

df.drop_duplicates()                      # remove duplicates (keep first)
df.drop_duplicates(subset=["email"])      # deduplicate on one column
df.drop_duplicates(keep="last")
df.drop_duplicates(inplace=True)
```

---

## String Operations (`.str` accessor)

```python
s = df["name"]

# Case
s.str.lower();  s.str.upper();  s.str.title();  s.str.capitalize()
s.str.swapcase()

# Strip whitespace
s.str.strip()           # both ends
s.str.lstrip()
s.str.rstrip()
s.str.strip("$")        # strip specific chars

# Pad / align
s.str.pad(10, side="left", fillchar="0")
s.str.zfill(5)          # zero-pad to width 5
s.str.center(10)
s.str.ljust(10);  s.str.rjust(10)

# Search / match
s.str.contains("ali", case=False, na=False)   # returns bool Series
s.str.startswith("A")
s.str.endswith("e")
s.str.match(r"^A\w+")   # regex match at start
s.str.fullmatch(r"\d+") # entire string must match
s.str.count("a")         # count occurrences
s.str.find("li")         # index of first match

# Extract / split
s.str.split(",")                    # list per cell
s.str.split(",", expand=True)       # into separate columns
s.str.split(",", n=1, expand=True)  # max 1 split
s.str.get(0)                        # first element after split/list
s.str.extract(r"(\d+)")             # first capture group → column
s.str.extractall(r"(\d+)")          # all matches → multi-index rows
s.str.findall(r"\d+")               # all matches → list per cell

# Replace
s.str.replace("old", "new")
s.str.replace(r"\s+", " ", regex=True)   # regex replace
s.str.removeprefix("Dr. ")         # Python 3.9+ style (pandas 1.4+)
s.str.removesuffix(" PhD")

# Slice
s.str[0:3]              # first 3 chars
s.str[-4:]              # last 4 chars

# Concatenate
s.str.cat(sep=", ")     # join all strings in series
s.str.cat(other_s, sep="-")   # element-wise join with another series

# Test
s.str.isdigit()
s.str.isalpha()
s.str.isalnum()
s.str.isnumeric()
s.str.isspace()
s.str.isupper();  s.str.islower()

# Length
s.str.len()

# Encode / decode
s.str.encode("utf-8")
s.str.decode("utf-8")
```

---

## Datetime Operations (`.dt` accessor)

```python
df["date"] = pd.to_datetime(df["date"])

# Components
df["date"].dt.year;  df["date"].dt.month;  df["date"].dt.day
df["date"].dt.hour;  df["date"].dt.minute;  df["date"].dt.second
df["date"].dt.microsecond;  df["date"].dt.nanosecond
df["date"].dt.dayofweek      # 0=Monday … 6=Sunday
df["date"].dt.day_of_week    # alias
df["date"].dt.day_name()     # "Monday", "Tuesday", ...
df["date"].dt.month_name()   # "January", ...
df["date"].dt.dayofyear      # 1–366
df["date"].dt.weekofyear     # ISO week number (deprecated → isocalendar)
df["date"].dt.isocalendar()  # returns df with year/week/day
df["date"].dt.quarter        # 1–4
df["date"].dt.is_month_start;  df["date"].dt.is_month_end
df["date"].dt.is_year_start;   df["date"].dt.is_year_end
df["date"].dt.is_leap_year

# Rounding
df["date"].dt.floor("H")     # floor to hour
df["date"].dt.ceil("min")    # ceil to minute
df["date"].dt.round("D")     # round to day

# Conversion
df["date"].dt.date           # Python date objects
df["date"].dt.time           # Python time objects
df["date"].dt.timestamp()    # POSIX timestamps
df["date"].dt.to_period("M") # PeriodIndex (monthly)

# Timezone
df["date"].dt.tz_localize("UTC")
df["date"].dt.tz_convert("US/Eastern")

# Timedeltas
df["date2"] - df["date1"]              # Series of Timedelta
(df["date2"] - df["date1"]).dt.days
(df["date2"] - df["date1"]).dt.total_seconds()

# Date ranges
pd.date_range("2024-01-01", periods=12, freq="M")
pd.date_range("2024-01-01", "2024-12-31", freq="D")
pd.bdate_range("2024-01-01", periods=5)   # business days
pd.period_range("2024-01", periods=12, freq="M")
```

### Resample (time-series aggregation)

```python
df.set_index("date", inplace=True)

df.resample("M").mean()       # monthly mean
df.resample("W").sum()        # weekly sum
df.resample("Q").agg({"revenue":"sum","users":"max"})
df.resample("D").ffill()      # fill gaps by day
df.resample("H").interpolate()
```

---

## GroupBy

```python
g = df.groupby("category")           # GroupBy object
g = df.groupby(["cat","sub"])        # multi-level
g = df.groupby("cat", sort=False)    # preserve original order
g = df.groupby("cat", dropna=False)  # include NaN groups

# Single aggregation
g["score"].mean()
g["score"].sum()
g["score"].count()
g["score"].min()
g["score"].max()
g["score"].std()
g["score"].median()
g["score"].nunique()
g["score"].first()
g["score"].last()

# Multiple aggregations at once
g.agg({"score": "mean", "age": "max"})
g["score"].agg(["mean","std","count"])
g.agg(
    avg_score=("score","mean"),
    max_age=("age","max"),
    count=("name","count")
)

# Transform — returns same shape as input (for adding back to df)
df["score_zscore"] = g["score"].transform(lambda x: (x - x.mean()) / x.std())
df["group_mean"]   = g["score"].transform("mean")
df["rank_in_group"] = g["score"].transform("rank")

# Filter — keep groups meeting a condition
df.groupby("cat").filter(lambda g: len(g) >= 5)     # groups with ≥5 rows
df.groupby("cat").filter(lambda g: g["score"].mean() > 80)

# Apply — most flexible (slow for large data)
df.groupby("cat").apply(lambda g: g.nlargest(2, "score"))

# Iterate over groups
for name, group_df in df.groupby("category"):
    print(name, group_df.shape)

# Named aggregations (pandas 0.25+)
result = df.groupby("cat").agg(
    revenue_sum   = ("revenue", "sum"),
    revenue_mean  = ("revenue", "mean"),
    user_count    = ("user_id", "nunique"),
    first_seen    = ("date", "min"),
    last_seen     = ("date", "max")
)
```

---

## Merging & Joining

### merge (SQL-style joins)

```python
# Inner join (default)
pd.merge(left, right, on="key")
pd.merge(left, right, on=["key1","key2"])          # composite key
pd.merge(left, right, left_on="l_id", right_on="r_id")

# Join types
pd.merge(left, right, on="key", how="left")        # left join
pd.merge(left, right, on="key", how="right")       # right join
pd.merge(left, right, on="key", how="outer")       # full outer join
pd.merge(left, right, on="key", how="inner")       # inner join
pd.merge(left, right, on="key", how="cross")       # cartesian product

# Merge options
pd.merge(left, right, on="key", suffixes=("_x","_y"))  # for overlapping cols
pd.merge(left, right, on="key", validate="one_to_many")  # sanity check
pd.merge(left, right, on="key", indicator=True)    # adds _merge column

# Merge on index
pd.merge(left, right, left_index=True, right_index=True)
pd.merge(left, right, left_index=True, right_on="key")
```

### join (index-based)

```python
df1.join(df2)                              # join on index
df1.join(df2, how="left")
df1.join(df2, on="key")                    # df1 col → df2 index
df1.join([df2, df3])                       # join multiple at once
```

### concat (stacking)

```python
pd.concat([df1, df2])                      # stack rows (axis=0)
pd.concat([df1, df2], ignore_index=True)   # reset index
pd.concat([df1, df2], axis=1)             # stack columns
pd.concat([df1, df2], keys=["a","b"])     # hierarchical index
pd.concat([df1, df2], join="inner")       # only common columns
pd.concat([df1, df2], join="outer")       # all columns (default)
```

---

## Pivot Tables & Reshaping

### pivot_table

```python
df.pivot_table(
    values="revenue",
    index="region",
    columns="quarter",
    aggfunc="sum",            # "mean","count","min","max" or function
    fill_value=0,
    margins=True,             # add row/col totals
    margins_name="Total"
)

# Multiple values and aggfuncs
df.pivot_table(values=["rev","cost"], index="region", aggfunc={"rev":"sum","cost":"mean"})
```

### pivot (no aggregation — must be unique)

```python
df.pivot(index="date", columns="product", values="sales")
```

### melt (wide → long)

```python
pd.melt(df, id_vars=["name"], value_vars=["q1","q2","q3","q4"],
        var_name="quarter", value_name="revenue")
df.melt(id_vars=["id","name"])  # all other columns become rows
```

### stack / unstack

```python
df.stack()           # columns → innermost row index
df.unstack()         # innermost row index → columns
df.unstack(level=0)  # specify which level
```

### crosstab

```python
pd.crosstab(df["gender"], df["grade"])
pd.crosstab(df["gender"], df["grade"], normalize="index")  # row percentages
pd.crosstab(df["gender"], df["grade"], values=df["score"], aggfunc="mean")
```

---

## Apply, Map & Applymap

```python
# Apply along axis (row or column)
df.apply(np.sum, axis=0)                     # column sums
df.apply(np.sum, axis=1)                     # row sums
df.apply(lambda row: row["a"] + row["b"], axis=1)
df["score"].apply(lambda x: "pass" if x >= 50 else "fail")

# Map (element-wise on Series, pandas 2.1+)
df["grade"].map({"A":4,"B":3,"C":2})         # dict lookup
df["score"].map(lambda x: round(x, 1))

# map vs apply on Series
s.map(func)      # element-wise, NaN-aware, dict/Series supported
s.apply(func)    # element-wise, can return Series (expands)

# map on DataFrame (element-wise on whole df — renamed in 2.1)
df.map(np.sqrt)                              # pandas 2.1+
df.applymap(np.sqrt)                         # pandas < 2.1 (deprecated)

# pipe — method chaining
df.pipe(func1).pipe(func2, arg=val)
```

---

## Window Functions

### Rolling

```python
df["score"].rolling(window=3).mean()        # 3-period moving average
df["score"].rolling(window=3).sum()
df["score"].rolling(window=3).std()
df["score"].rolling(window=3, min_periods=1).mean()  # allow smaller windows at edges
df["score"].rolling("7D").mean()            # time-based window (needs DatetimeIndex)
df.rolling(3).agg({"score":"mean","age":"max"})
```

### Expanding

```python
df["score"].expanding().mean()              # cumulative mean from start
df["score"].expanding(min_periods=5).sum()  # start after 5 rows
```

### Exponentially Weighted (EWM)

```python
df["score"].ewm(span=10).mean()             # exponential moving average
df["score"].ewm(halflife=5).mean()
df["score"].ewm(com=0.5).std()
df["score"].ewm(alpha=0.3).mean()           # smoothing factor directly
```

---

## Categorical Data

```python
df["grade"] = df["grade"].astype("category")
df["size"] = pd.Categorical(["S","M","L","M","S"], categories=["S","M","L"], ordered=True)

df["grade"].cat.categories          # Index(['A','B','C'])
df["grade"].cat.codes               # integer codes (memory efficient)
df["grade"].cat.ordered
df["grade"].cat.add_categories(["F"])
df["grade"].cat.remove_categories(["F"])
df["grade"].cat.rename_categories({"A":"Excellent"})
df["grade"].cat.reorder_categories(["C","B","A"])
df["grade"].cat.set_categories(new_cats, ordered=True)
df["grade"].cat.remove_unused_categories()
pd.cut(df["age"], bins=[0,18,35,60,100], labels=["child","young","adult","senior"])
pd.qcut(df["score"], q=4, labels=["Q1","Q2","Q3","Q4"])   # equal-frequency bins
```

---

## Index Operations

```python
df.index                               # RangeIndex, Int64Index, etc.
df.set_index("name")                   # set column as index
df.set_index(["region","date"])        # MultiIndex
df.reset_index()                       # move index back to columns
df.reset_index(drop=True)             # discard the index

# MultiIndex
df.index.get_level_values(0)           # outer level values
df.index.get_level_values("region")
df.loc[("West", "2024")]               # select with tuple
df.xs("West", level="region")         # cross-section
df.swaplevel(0, 1)                     # swap MultiIndex levels
df.sort_index(level=0)

# Index alignment (automatic on arithmetic)
s1 = pd.Series([1,2,3], index=["a","b","c"])
s2 = pd.Series([10,20], index=["b","c"])
s1 + s2       # a=NaN, b=22, c=23  — aligns on index

# Useful index methods
df.index.is_unique
df.index.is_monotonic_increasing
df.reindex(["a","b","c","d"])          # reindex, NaN for missing
df.reindex(new_idx, fill_value=0)
df.reindex_like(other_df)             # match other df's index
```

---

## Data Cleaning — Complete Workflow

### Inspect

```python
df.info()
df.describe(include="all")
df.isna().sum()
df.duplicated().sum()
df.dtypes
df.nunique()               # unique values per column
df["col"].value_counts()   # frequency of each value
df["col"].value_counts(normalize=True)   # proportions
df["col"].value_counts(dropna=False)     # include NaN count
```

### Standardise column names

```python
df.columns = df.columns.str.lower().str.strip().str.replace(" ","_")
df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]","", regex=True)
```

### Handle missing values (strategy depends on domain)

```python
# 1. Drop columns with > 50% missing
df = df.loc[:, df.isna().mean() < 0.5]

# 2. Drop rows with any missing in key columns
df = df.dropna(subset=["target","id"])

# 3. Impute numeric: mean / median
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
df[num_cols] = imp.fit_transform(df[num_cols])

# 4. Impute categorical: most frequent
imp_cat = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

# 5. KNN imputation (better but slower)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])
```

### Fix data types

```python
df["age"]    = pd.to_numeric(df["age"], errors="coerce")
df["date"]   = pd.to_datetime(df["date"], errors="coerce")
df["price"]  = df["price"].str.replace("[$,]","", regex=True).astype(float)
df["phone"]  = df["phone"].str.replace(r"\D","", regex=True)  # digits only
```

### Remove outliers

```python
# IQR method
Q1 = df["score"].quantile(0.25)
Q3 = df["score"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["score"] >= Q1 - 1.5*IQR) & (df["score"] <= Q3 + 1.5*IQR)]

# Z-score method
from scipy import stats
df = df[(np.abs(stats.zscore(df[num_cols])) < 3).all(axis=1)]

# Cap (winsorise) instead of removing
lower = df["score"].quantile(0.01)
upper = df["score"].quantile(0.99)
df["score"] = df["score"].clip(lower, upper)
```

### Validate ranges and values

```python
assert df["age"].between(0, 120).all(), "Invalid ages"
assert df["score"].between(0, 100).all()
assert df["email"].str.contains("@").all()
invalid_mask = ~df["status"].isin(["active","inactive","pending"])
df.loc[invalid_mask, "status"] = np.nan   # nullify invalid categories
```

---

## Feature Engineering for Machine Learning

### Encoding categorical variables

```python
# Label encoding (ordinal)
df["grade_code"] = df["grade"].map({"A":3,"B":2,"C":1,"F":0})

# One-hot encoding
pd.get_dummies(df, columns=["city","category"], drop_first=True)
pd.get_dummies(df, columns=["city"], prefix="city", dtype=int)

# Target encoding (mean of target per category)
target_mean = df.groupby("city")["price"].transform("mean")
df["city_encoded"] = target_mean

# Frequency encoding
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)

# sklearn OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
enc = OrdinalEncoder()
df[cat_cols] = enc.fit_transform(df[cat_cols])

# sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop="first")
ohe_array = ohe.fit_transform(df[["city"]])
```

### Scaling / Normalisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler: zero mean, unit variance — good for linear models, PCA
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# MinMaxScaler: scales to [0,1] — good for neural nets
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# RobustScaler: uses median and IQR — good when outliers are present
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Log transform for skewed features
df["revenue_log"] = np.log1p(df["revenue"])   # log(1+x) handles zeros

# Box-Cox / Yeo-Johnson (makes features more Gaussian)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")    # handles negatives
df[num_cols] = pt.fit_transform(df[num_cols])
```

### Creating new features

```python
# Arithmetic
df["bmi"]       = df["weight"] / (df["height"] / 100) ** 2
df["age_score"] = df["age"] * df["score"]

# Binning (continuous → categorical)
df["age_group"] = pd.cut(df["age"], bins=[0,18,35,60,100],
                          labels=["teen","young","mid","senior"])
df["score_q"]   = pd.qcut(df["score"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])

# Date features
df["year"]      = df["date"].dt.year
df["month"]     = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"]= df["date"].dt.dayofweek >= 5
df["days_since"]= (pd.Timestamp.today() - df["date"]).dt.days
df["quarter"]   = df["date"].dt.quarter

# Text features
df["name_length"]  = df["name"].str.len()
df["word_count"]   = df["description"].str.split().str.len()
df["has_email"]    = df["contact"].str.contains("@", na=False)
df["digit_count"]  = df["phone"].str.count(r"\d")

# Lag and rolling features (for time series)
df = df.sort_values("date")
df["sales_lag1"]  = df["sales"].shift(1)
df["sales_lag7"]  = df["sales"].shift(7)
df["sales_roll3"] = df["sales"].rolling(3).mean()
df["sales_diff"]  = df["sales"].diff()
df["sales_pct"]   = df["sales"].pct_change()

# Interaction features
df["age_x_score"] = df["age"] * df["score"]
df["age_div_score"] = df["age"] / (df["score"] + 1e-9)   # avoid div-by-zero
```

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Split DataFrame
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42,
                                                      stratify=y)  # for classification

# Temporal split (no shuffling for time series)
split_date = "2024-01-01"
train = df[df["date"] < split_date]
test  = df[df["date"] >= split_date]
```

### Preparing final feature matrix

```python
# Select feature subsets
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
bool_cols= df.select_dtypes(include=["bool"]).columns.tolist()

# Remove columns leaking the target or not useful
drop_cols = ["id","name","date","target"]
X = df.drop(columns=drop_cols)

# Check for infinite values (will crash most models)
np.isinf(df[num_cols]).sum()
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

# Final check before model training
assert X.isna().sum().sum() == 0, "Still has NaN!"
assert np.isinf(X.values).sum() == 0, "Still has Inf!"
print(X.shape)
```

---

## sklearn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Full pipeline with model
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ("prep",  preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)

# Extract feature names after transformation
ohe_features = model["prep"].named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols)
all_features = num_cols + list(ohe_features)
```

---

## Performance & Memory Optimisation

```python
# Check memory usage
df.memory_usage(deep=True).sum() / 1e6   # MB

# Downcast numerics
df[int_cols]   = df[int_cols].apply(pd.to_numeric, downcast="integer")
df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

# Use category for low-cardinality strings (often 5–10× savings)
for col in cat_cols:
    if df[col].nunique() / len(df) < 0.1:   # < 10% unique
        df[col] = df[col].astype("category")

# Read in chunks for large files
chunks = []
for chunk in pd.read_csv("big.csv", chunksize=100_000):
    processed = chunk[chunk["value"] > 0]   # filter early
    chunks.append(processed)
df = pd.concat(chunks, ignore_index=True)

# Use parquet instead of CSV for speed + type preservation
df.to_parquet("data.parquet")
df = pd.read_parquet("data.parquet", columns=["a","b"])   # column pruning

# Copy-on-Write (pandas 2.0+) — avoid SettingWithCopyWarning
pd.options.mode.copy_on_write = True

# eval and query for large DataFrames (uses numexpr, avoids temporaries)
result = df.query("age > 30 and score > 80")
df["z"] = df.eval("(score - score.mean()) / score.std()")
```

---

## Useful Configuration

```python
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_colwidth", 100)
pd.set_option("mode.chained_assignment", None)    # silence SettingWithCopy

# Context manager (temporary)
with pd.option_context("display.max_rows", 20):
    print(df)

# Reset to defaults
pd.reset_option("all")

# Useful global settings for data science
pd.options.display.float_format = "{:,.3f}".format
pd.options.mode.copy_on_write = True     # pandas 2.0+ best practice
```

---

## Quick Reference Card

```
I/O             read_csv / read_excel / read_json / read_parquet / read_sql
                to_csv / to_excel / to_json / to_parquet / to_sql

Select          df["col"]  df[["a","b"]]
                .loc[label, col]  .iloc[pos, col]  .at[lbl,col]  .iat[pos,col]
                .query("age > 30")  boolean mask

Clean           isna/notna  dropna  fillna  ffill/bfill  interpolate
                replace  drop_duplicates  duplicated  astype  to_numeric  to_datetime
                str.strip/lower/replace/contains/extract  clip (outliers)

Engineer        pd.cut / pd.qcut  get_dummies  shift / diff / pct_change
                rolling / ewm / expanding  dt.year/month/dayofweek

GroupBy         groupby → agg / transform / filter / apply
                Named agg: agg(alias=("col","func"))

Merge           pd.merge(how="left/right/inner/outer/cross")
                pd.concat(axis=0/1)  df.join()

Reshape         pivot_table  pivot  melt  stack  unstack  crosstab

Scale/Encode    StandardScaler  MinMaxScaler  RobustScaler  PowerTransformer
                LabelEncoder  OrdinalEncoder  OneHotEncoder  pd.get_dummies

ML Prep         train_test_split(stratify=y)  ColumnTransformer  Pipeline
                SimpleImputer  KNNImputer  select_dtypes  replace inf/nan

Optimise        astype("category")  downcast  read_parquet  chunksize
                query / eval  copy_on_write  memory_usage(deep=True)
```

---

*Pandas version this covers: 1.5 – 2.2. Docs: https://pandas.pydata.org/docs/*
