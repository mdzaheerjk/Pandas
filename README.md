# Pandas — Complete Notes for ML / DL / Gen AI / Agentic AI Engineers
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

A labelled 1-D array — like a single spreadsheet column where every value has a named index.

```python
s = pd.Series([10, 20, 30])                        # index auto-assigned 0,1,2
s = pd.Series([10, 20, 30], index=["a","b","c"])
s = pd.Series({"a": 10, "b": 20})                  # from dict

s["a"]           # → 10
s[["a","c"]]     # fancy indexing → Series with a=10, c=30
s[s > 15]        # boolean mask → b=20, c=30
s.values         # numpy array
s.index          # Index object
s.dtype
s.name = "score"
```

**Sample output:**

```
a    10
b    20
c    30
dtype: int64
```

---

### DataFrame

A labelled 2-D table — rows and columns, where each column is a Series.

```python
df = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol"],
    "age":   [25, 30, 35],
    "score": [90.5, 85.0, 92.3]
})

df = pd.DataFrame([[1,2],[3,4]], columns=["A","B"], index=["r1","r2"])
df = pd.DataFrame(records_list)          # list of dicts
```

**Sample output:**

```
    name  age  score
0  Alice   25   90.5
1    Bob   30   85.0
2  Carol   35   92.3
```

---

### Key Attributes

```python
df.shape          # (3, 3) — rows × cols
df.dtypes         # dtype per column
df.columns        # Index(['name', 'age', 'score'])
df.index          # RangeIndex(start=0, stop=3, step=1)
df.values         # underlying numpy array
df.info()         # dtypes + non-null counts + memory
df.describe()     # stats for numeric columns
df.describe(include="all")   # includes object/category cols
df.head(5)        # first 5 rows
df.tail(5)        # last 5 rows
df.sample(5)      # 5 random rows
df.size           # total elements (rows × cols)
df.ndim           # always 2
df.memory_usage(deep=True)
```

**Sample output of `df.describe()`:**

```
             age      score
count   3.000000   3.000000
mean   30.000000  89.266667
std     5.000000   3.793976
min    25.000000  85.000000
25%    27.500000  87.750000
50%    30.000000  90.500000
75%    32.500000  91.400000
max    35.000000  92.300000
```

**Sample output of `df.info()`:**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   name    3 non-null      object
 1   age     3 non-null      int64
 2   score   3 non-null      float64
dtypes: float64(1), int64(1), object(1)
memory usage: 200.0+ bytes
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
df = pd.read_excel("data.xlsx", sheet_name=0)            # by index
all_sheets = pd.read_excel("data.xlsx", sheet_name=None) # dict of dfs

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
df.to_markdown()        # returns markdown string
df.to_string()          # plain text
df.to_dict()            # dict of dicts
df.to_dict("records")   # list of dicts
df.to_numpy()           # numpy array
```

---

## Indexing & Selection

### `.loc` — label-based

Selects rows and columns **by their label names**. Slices are **inclusive** on both ends.

```python
df.loc[0]                      # row with label 0
df.loc[0, "name"]              # single value → "Alice"
df.loc[0:2, "name":"score"]    # inclusive slice by label
df.loc[[0,2], ["name","age"]]  # fancy selection
df.loc[df["age"] > 28]         # boolean mask
df.loc[df["age"] > 28, "name"] # filtered, single column
```

**Sample output of `df.loc[df["age"] > 28]`:**

```
    name  age  score
1    Bob   30   85.0
2  Carol   35   92.3
```

---

### `.iloc` — integer position-based

Selects rows and columns **by integer position** (like list indexing). Slices are **exclusive** at the end.

```python
df.iloc[0]                     # first row
df.iloc[0, 1]                  # row 0, col 1 → 25
df.iloc[0:3, 0:2]              # exclusive slice by position
df.iloc[[0,2], [0,1]]          # list of positions
df.iloc[-1]                    # last row
```

**Sample output of `df.iloc[0:2, 0:2]`:**

```
    name  age
0  Alice   25
1    Bob   30
```

---

### Boolean Indexing

```python
df[df["age"] > 25]
df[(df["age"] > 25) & (df["score"] > 85)]   # & | ~ (not and/or/not)
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

### Setting Values

```python
df.loc[df["score"] > 90, "grade"] = "A"
df.iloc[0, 1] = 26
df["age"] = df["age"] + 1           # update whole column
df.loc[:, "score"] = df["score"] * 1.1
```

---

## DataFrame Operations

### Adding & Removing Columns

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

### Adding & Removing Rows

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

**Sample output of `df.sort_values("age", ascending=False)`:**

```
    name  age  score
2  Carol   35   92.3
1    Bob   30   85.0
0  Alice   25   90.5
```

---

## Missing Data

### Definition

`NaN` (Not a Number) marks missing values. `NaT` is used for missing datetimes. Always inspect missing data before modelling.

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

print(df.info())                   # shows non-null counts
df.isna().mean().sort_values(ascending=False)  # sorted % missing
```

**Sample output of `df.isna().sum()`:**

```
name     0
age      1
score    2
dtype: int64
```

### Dropping

```python
df.dropna()                        # drop rows with ANY NaN
df.dropna(how="all")               # drop rows where ALL are NaN
df.dropna(axis=1)                  # drop columns with any NaN
df.dropna(subset=["age","score"])  # only consider these cols
df.dropna(thresh=3)                # keep rows with at least 3 non-NaN values
```

### Filling

```python
df.fillna(0)                       # fill all NaN with 0
df.fillna({"age": 0, "score": df["score"].mean()})   # per-column fill

# Forward fill — carry last valid value forward
df.ffill()
df["col"].ffill()

# Backward fill — use next valid value
df.bfill()

# Fill with statistics
df["age"].fillna(df["age"].mean())
df["age"].fillna(df["age"].median())
df["city"].fillna(df["city"].mode()[0])   # most frequent value

# Interpolation
df["price"].interpolate()                           # linear (default)
df["price"].interpolate(method="time")              # time-indexed data
df["price"].interpolate(method="polynomial", order=2)
```

**Sample output — before and after `fillna(mean)`:**

```
Before:         After fillna(87.75):
score           score
90.5      →     90.5
NaN       →     87.75
NaN       →     87.75
92.3      →     92.3
```

### Replacing

```python
df.replace(-999, np.nan)                            # replace sentinel with NaN
df.replace(["-","?","NA"], np.nan)                  # multiple values → NaN
df.replace({"gender": {"M":"Male","F":"Female"}})   # per-column dict
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

# Datetime
pd.to_datetime(df["date"])
pd.to_datetime(df["date"], format="%Y-%m-%d")
pd.to_datetime(df["date"], errors="coerce")         # bad dates → NaT
pd.to_datetime(df[["year","month","day"]])           # from components

# Infer better dtypes automatically
df = df.convert_dtypes()     # converts to best nullable dtypes
df = df.infer_objects()      # tries to infer object columns
```

---

## Duplicates

A **duplicate** is a row that appears more than once. Duplicates inflate counts, distort statistics, and cause data leakage in ML.

```python
df.duplicated()                            # bool Series — True for duplicates
df.duplicated(subset=["name","age"])       # only these cols define a duplicate
df.duplicated(keep="first")               # mark all but first occurrence
df.duplicated(keep="last")                # mark all but last
df.duplicated(keep=False)                 # mark ALL duplicates

df.duplicated().sum()                     # count duplicate rows

df.drop_duplicates()                      # remove duplicates (keep first)
df.drop_duplicates(subset=["email"])      # deduplicate on one column
df.drop_duplicates(keep="last")
df.drop_duplicates(inplace=True)
```

**Sample output — `df.duplicated(keep=False)`:**

```
    email    name    duplicated
0  a@x.com  Alice   False
1  b@x.com  Bob     False
2  a@x.com  Alice2  True
```

---

## String Operations (`.str` accessor)

The `.str` accessor lets you apply string methods element-wise to a Series without writing a loop.

```python
s = df["name"]

# Case
s.str.lower()         # "alice", "bob"
s.str.upper()         # "ALICE", "BOB"
s.str.title()         # "Alice", "Bob"

# Strip whitespace
s.str.strip()         # both ends
s.str.lstrip()
s.str.rstrip()
s.str.strip("$")      # strip specific chars

# Search / match
s.str.contains("ali", case=False, na=False)   # bool Series
s.str.startswith("A")
s.str.endswith("e")
s.str.match(r"^A\w+")
s.str.count("a")      # count occurrences

# Extract / split
s.str.split(",")                    # list per cell
s.str.split(",", expand=True)       # into separate columns
s.str.extract(r"(\d+)")             # first capture group → column
s.str.findall(r"\d+")               # all matches → list per cell

# Replace
s.str.replace("old", "new")
s.str.replace(r"\s+", " ", regex=True)

# Slice
s.str[0:3]            # first 3 chars
s.str[-4:]            # last 4 chars

# Test
s.str.isdigit()
s.str.isalpha()
s.str.len()           # length of each string
```

**Sample output of `df["name"].str.upper()`:**

```
0    ALICE
1      BOB
2    CAROL
dtype: object
```

---

## Datetime Operations (`.dt` accessor)

The `.dt` accessor unlocks date/time components and arithmetic on a datetime Series.

```python
df["date"] = pd.to_datetime(df["date"])

# Components
df["date"].dt.year
df["date"].dt.month
df["date"].dt.day
df["date"].dt.hour
df["date"].dt.dayofweek      # 0=Monday … 6=Sunday
df["date"].dt.day_name()     # "Monday", "Tuesday", ...
df["date"].dt.month_name()   # "January", ...
df["date"].dt.quarter        # 1–4
df["date"].dt.is_month_start
df["date"].dt.is_leap_year

# Rounding
df["date"].dt.floor("H")     # floor to hour
df["date"].dt.ceil("min")    # ceil to minute
df["date"].dt.round("D")     # round to day

# Timezone
df["date"].dt.tz_localize("UTC")
df["date"].dt.tz_convert("US/Eastern")

# Timedeltas
df["date2"] - df["date1"]                   # Series of Timedelta
(df["date2"] - df["date1"]).dt.days
(df["date2"] - df["date1"]).dt.total_seconds()

# Date ranges
pd.date_range("2024-01-01", periods=12, freq="M")
pd.date_range("2024-01-01", "2024-12-31", freq="D")
pd.bdate_range("2024-01-01", periods=5)     # business days
```

**Sample output of `.dt` components:**

```
   date        year  month  day  dayofweek  day_name
0  2024-03-15  2024      3   15          4   Friday
1  2024-07-04  2024      7    4          3 Thursday
2  2024-12-25  2024     12   25          2Wednesday
```

### Resample (time-series aggregation)

```python
df.set_index("date", inplace=True)

df.resample("M").mean()    # monthly mean
df.resample("W").sum()     # weekly sum
df.resample("Q").agg({"revenue":"sum","users":"max"})
df.resample("D").ffill()   # fill gaps by day
df.resample("H").interpolate()
```

---

## GroupBy

GroupBy follows a **split → apply → combine** pattern: rows are split into groups, a function is applied to each group, and the results are combined back.

```python
g = df.groupby("category")           # GroupBy object
g = df.groupby(["cat","sub"])        # multi-level
g = df.groupby("cat", sort=False)    # preserve original order
g = df.groupby("cat", dropna=False)  # include NaN groups
```

### Aggregation

```python
g["score"].mean()
g["score"].sum()
g["score"].count()
g["score"].min()
g["score"].max()
g["score"].std()
g["score"].nunique()

# Multiple aggregations
g.agg({"score": "mean", "age": "max"})
g["score"].agg(["mean","std","count"])

# Named aggregations (pandas 0.25+)
g.agg(
    avg_score = ("score", "mean"),
    max_age   = ("age",   "max"),
    count     = ("name",  "count")
)
```

**Sample output:**

```
          avg_score  max_age  count
category
A              91.4       35      3
B              85.0       30      2
C              72.5       28      1
```

### Transform

Returns a Series the **same shape as the input** — useful for adding computed columns back to the original DataFrame.

```python
df["score_zscore"] = g["score"].transform(lambda x: (x - x.mean()) / x.std())
df["group_mean"]   = g["score"].transform("mean")
df["rank_in_group"] = g["score"].transform("rank")
```

**Sample output:**

```
    name  grade  score  group_mean
0  Alice      A   92.3        91.4
1    Bob      B   85.0        85.0
2  Carol      A   90.5        91.4
```

### Filter & Apply

```python
# Filter — keep groups meeting a condition
df.groupby("cat").filter(lambda g: len(g) >= 5)
df.groupby("cat").filter(lambda g: g["score"].mean() > 80)

# Apply — most flexible (slow for large data)
df.groupby("cat").apply(lambda g: g.nlargest(2, "score"))

# Iterate
for name, group_df in df.groupby("category"):
    print(name, group_df.shape)
```

---

## Merging & Joining

### `pd.merge` — SQL-style joins

```python
# Inner join (default) — only matching rows from both
pd.merge(left, right, on="key")
pd.merge(left, right, on=["key1","key2"])         # composite key
pd.merge(left, right, left_on="l_id", right_on="r_id")

# Join types
pd.merge(left, right, on="key", how="left")       # all left + matching right
pd.merge(left, right, on="key", how="right")      # all right + matching left
pd.merge(left, right, on="key", how="outer")      # all rows from both
pd.merge(left, right, on="key", how="inner")      # only matching rows
pd.merge(left, right, on="key", how="cross")      # cartesian product

# Options
pd.merge(left, right, on="key", suffixes=("_x","_y"))
pd.merge(left, right, on="key", validate="one_to_many")
pd.merge(left, right, on="key", indicator=True)   # adds _merge column
```

**Sample — inner vs left join:**

```
Left df:          Right df:
id  name          id  score
1   Alice         1   90
2   Bob           3   85
3   Carol         4   78

inner join (id):  left join (id):
id  name  score   id  name  score
1   Alice  90     1   Alice  90.0
3   Carol  85     2   Bob    NaN
                  3   Carol  85.0
```

### `pd.concat` — stacking

```python
pd.concat([df1, df2])                     # stack rows (axis=0)
pd.concat([df1, df2], ignore_index=True)  # reset index
pd.concat([df1, df2], axis=1)            # stack columns
pd.concat([df1, df2], keys=["a","b"])    # hierarchical index
pd.concat([df1, df2], join="inner")      # only common columns
pd.concat([df1, df2], join="outer")      # all columns (default)
```

### `df.join` — index-based

```python
df1.join(df2)                  # join on index
df1.join(df2, how="left")
df1.join(df2, on="key")        # df1 col → df2 index
df1.join([df2, df3])           # join multiple at once
```

---

## Pivot Tables & Reshaping

### `pivot_table`

Like an Excel pivot table — groups data, aggregates values, and spreads one column's values across new columns.

```python
df.pivot_table(
    values="revenue",
    index="region",
    columns="quarter",
    aggfunc="sum",
    fill_value=0,
    margins=True,           # add row/col totals
    margins_name="Total"
)
```

**Sample output:**

```
quarter    Q1     Q2     Q3     Q4  Total
region
East     1200   1500   1800   2000   6500
West      900   1100   1300   1600   4900
Total    2100   2600   3100   3600  11400
```

### `melt` — wide → long

Unpivots a DataFrame from wide format (one column per time period) to long format (one row per observation).

```python
pd.melt(df, id_vars=["name"], value_vars=["q1","q2","q3","q4"],
        var_name="quarter", value_name="revenue")
```

**Sample output:**

```
Before (wide):              After melt (long):
name  q1   q2              name  quarter  revenue
Alice 100  120             Alice      q1      100
Bob   200  180             Alice      q2      120
                           Bob        q1      200
                           Bob        q2      180
```

### `stack` / `unstack`

```python
df.stack()           # columns → innermost row index (wide → long)
df.unstack()         # innermost row index → columns (long → wide)
df.unstack(level=0)  # specify which level
```

### `crosstab`

```python
pd.crosstab(df["gender"], df["grade"])
pd.crosstab(df["gender"], df["grade"], normalize="index")  # row %
pd.crosstab(df["gender"], df["grade"], values=df["score"], aggfunc="mean")
```

---

## Apply, Map & Pipe

```python
# apply — along an axis (row or column)
df.apply(np.sum, axis=0)                     # column sums
df.apply(np.sum, axis=1)                     # row sums
df.apply(lambda row: row["a"] + row["b"], axis=1)
df["score"].apply(lambda x: "pass" if x >= 50 else "fail")

# map — element-wise on Series, dict lookup supported
df["grade"].map({"A":4,"B":3,"C":2})        # dict lookup
df["score"].map(lambda x: round(x, 1))

# map on DataFrame (element-wise, pandas 2.1+)
df.map(np.sqrt)                             # pandas 2.1+
df.applymap(np.sqrt)                        # pandas < 2.1 (deprecated)

# pipe — method chaining
df.pipe(func1).pipe(func2, arg=val)
```

**Sample output of `.apply` for pass/fail:**

```
0    pass
1    pass
2    fail
3    pass
dtype: object
```

---

## Window Functions

### Rolling — moving window

Computes a statistic over a **sliding window** of N rows. Essential for smoothing noisy time series.

```python
df["score"].rolling(window=3).mean()        # 3-period moving average
df["score"].rolling(window=3).sum()
df["score"].rolling(window=3).std()
df["score"].rolling(window=3, min_periods=1).mean()  # handle edges
df["score"].rolling("7D").mean()            # time-based window
```

**Sample output:**

```
day  sales  ma3 (rolling mean)
1    100    NaN
2    120    NaN
3     90    103.3
4    150    120.0
5    130    123.3
```

### Expanding

Cumulative statistic from the start of the series — window grows with each row.

```python
df["score"].expanding().mean()              # cumulative mean
df["score"].expanding(min_periods=5).sum()
```

### Exponentially Weighted (EWM)

Gives **more weight to recent values** — useful for trends where older data matters less.

```python
df["score"].ewm(span=10).mean()    # exponential moving average
df["score"].ewm(halflife=5).mean()
df["score"].ewm(alpha=0.3).mean()  # smoothing factor directly
```

### Lag Features — shift / diff / pct_change

```python
df["sales_lag1"]  = df["sales"].shift(1)      # value from 1 row ago
df["sales_lag7"]  = df["sales"].shift(7)      # value from 7 rows ago
df["sales_diff"]  = df["sales"].diff()        # change from previous row
df["sales_pct"]   = df["sales"].pct_change()  # % change from previous row
```

**Sample output:**

```
day  sales  lag1  change  pct %
1    100    NaN   NaN     NaN
2    120    100    20    20.0
3     90    120   -30   -25.0
4    150     90    60    66.7
```

---

## Categorical Data

**Categorical** dtype stores data as integer codes internally — memory-efficient for columns with few unique values (e.g. grade, status, city).

```python
df["grade"] = df["grade"].astype("category")
df["size"] = pd.Categorical(["S","M","L","M","S"],
                             categories=["S","M","L"], ordered=True)

df["grade"].cat.categories              # Index(['A','B','C'])
df["grade"].cat.codes                   # integer codes
df["grade"].cat.ordered                 # True/False
df["grade"].cat.add_categories(["F"])
df["grade"].cat.remove_categories(["F"])
df["grade"].cat.rename_categories({"A":"Excellent"})
df["grade"].cat.reorder_categories(["C","B","A"])
df["grade"].cat.remove_unused_categories()

# Binning
pd.cut(df["age"], bins=[0,18,35,60,100],
       labels=["child","young","adult","senior"])

pd.qcut(df["score"], q=4,
        labels=["Q1","Q2","Q3","Q4"])   # equal-frequency bins
```

**Sample output of `pd.cut`:**

```
0    young     (age=28)
1    adult     (age=45)
2    senior    (age=70)
3    child     (age=14)
dtype: category
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
df.index.get_level_values(0)
df.loc[("West", "2024")]               # select with tuple
df.xs("West", level="region")         # cross-section
df.swaplevel(0, 1)
df.sort_index(level=0)

# Reindex
df.reindex(["a","b","c","d"])          # NaN for missing labels
df.reindex(new_idx, fill_value=0)

# Useful checks
df.index.is_unique
df.index.is_monotonic_increasing
```

---

## Data Cleaning — Complete Workflow

### Step 1 — Inspect

```python
df.info()
df.describe(include="all")
df.isna().sum()
df.duplicated().sum()
df.dtypes
df.nunique()
df["col"].value_counts()
df["col"].value_counts(normalize=True)
df["col"].value_counts(dropna=False)
```

### Step 2 — Standardise column names

```python
df.columns = df.columns.str.lower().str.strip().str.replace(" ","_")
df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]","", regex=True)
```

### Step 3 — Handle missing values

```python
# Drop columns with > 50% missing
df = df.loc[:, df.isna().mean() < 0.5]

# Drop rows missing in key columns
df = df.dropna(subset=["target","id"])

# Impute numeric — median
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
df[num_cols] = imp.fit_transform(df[num_cols])

# Impute categorical — most frequent
imp_cat = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

# KNN imputation (better but slower)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])
```

### Step 4 — Fix data types

```python
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
df["date"]  = pd.to_datetime(df["date"], errors="coerce")
df["price"] = df["price"].str.replace("[$,]","", regex=True).astype(float)
df["phone"] = df["phone"].str.replace(r"\D","", regex=True)  # digits only
```

### Step 5 — Remove outliers

```python
# IQR method — flag values outside 1.5× the interquartile range
Q1 = df["score"].quantile(0.25)
Q3 = df["score"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["score"] >= Q1 - 1.5*IQR) & (df["score"] <= Q3 + 1.5*IQR)]

# Z-score method — flag values more than 3 std devs from mean
from scipy import stats
df = df[(np.abs(stats.zscore(df[num_cols])) < 3).all(axis=1)]

# Cap (winsorise) — clip instead of removing
lower = df["score"].quantile(0.01)
upper = df["score"].quantile(0.99)
df["score"] = df["score"].clip(lower, upper)
```

### Step 6 — Validate

```python
assert df["age"].between(0, 120).all(), "Invalid ages"
assert df["score"].between(0, 100).all()
assert df["email"].str.contains("@").all()
invalid_mask = ~df["status"].isin(["active","inactive","pending"])
df.loc[invalid_mask, "status"] = np.nan
```

---

## Feature Engineering for Machine Learning

### Encoding Categorical Variables

```python
# Label encoding (ordinal relationship exists)
df["grade_code"] = df["grade"].map({"A":3,"B":2,"C":1,"F":0})

# One-hot encoding (no ordinal relationship)
pd.get_dummies(df, columns=["city","category"], drop_first=True)
pd.get_dummies(df, columns=["city"], prefix="city", dtype=int)

# Target encoding — mean of target per category
target_mean = df.groupby("city")["price"].transform("mean")
df["city_encoded"] = target_mean

# Frequency encoding
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)
```

**Sample output of `get_dummies`:**

```
Before:               After (drop_first=True):
city                  score  city_London  city_NYC
NYC        →          90     0            1
London                85     1            0
Berlin                92     0            0
```

### Scaling / Normalisation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler — zero mean, unit variance (good for linear models, PCA)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# MinMaxScaler — scales to [0,1] (good for neural nets)
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# RobustScaler — uses median and IQR (good when outliers are present)
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Log transform — reduces right skew
df["revenue_log"] = np.log1p(df["revenue"])   # log(1+x) handles zeros

# Box-Cox / Yeo-Johnson — makes features more Gaussian
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")    # handles negatives
df[num_cols] = pt.fit_transform(df[num_cols])
```

### Creating New Features

```python
# Arithmetic
df["bmi"]       = df["weight"] / (df["height"] / 100) ** 2
df["age_score"] = df["age"] * df["score"]

# Binning
df["age_group"] = pd.cut(df["age"], bins=[0,18,35,60,100],
                          labels=["teen","young","mid","senior"])
df["score_q"]   = pd.qcut(df["score"], q=5,
                           labels=["Q1","Q2","Q3","Q4","Q5"])

# Date features
df["year"]       = df["date"].dt.year
df["month"]      = df["date"].dt.month
df["dayofweek"]  = df["date"].dt.dayofweek
df["is_weekend"] = df["date"].dt.dayofweek >= 5
df["days_since"] = (pd.Timestamp.today() - df["date"]).dt.days
df["quarter"]    = df["date"].dt.quarter

# Text features
df["name_length"] = df["name"].str.len()
df["word_count"]  = df["description"].str.split().str.len()
df["has_email"]   = df["contact"].str.contains("@", na=False)

# Lag and rolling (time series)
df = df.sort_values("date")
df["sales_lag1"]  = df["sales"].shift(1)
df["sales_lag7"]  = df["sales"].shift(7)
df["sales_roll3"] = df["sales"].rolling(3).mean()
df["sales_diff"]  = df["sales"].diff()
df["sales_pct"]   = df["sales"].pct_change()

# Interaction features
df["age_x_score"]   = df["age"] * df["score"]
df["age_div_score"] = df["age"] / (df["score"] + 1e-9)   # avoid div-by-zero
```

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop("target", axis=1)
y = df["target"]

# Random split (classification — use stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Temporal split (time series — never shuffle)
split_date = "2024-01-01"
train = df[df["date"] < split_date]
test  = df[df["date"] >= split_date]
```

### Preparing the Final Feature Matrix

```python
# Select by dtype
num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols  = df.select_dtypes(include=["object","category"]).columns.tolist()
bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

# Drop non-feature columns
drop_cols = ["id","name","date","target"]
X = df.drop(columns=drop_cols)

# Handle infinite values (crash most models)
np.isinf(df[num_cols]).sum()
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

# Final assertions
assert X.isna().sum().sum() == 0, "Still has NaN!"
assert np.isinf(X.values).sum() == 0, "Still has Inf!"
print(X.shape)
```

---

## sklearn Pipeline Integration

A **Pipeline** chains preprocessing and modelling steps into one object — preventing data leakage and simplifying cross-validation.

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
preds = model.predict(X_test)
score = model.score(X_test, y_test)

# Extract feature names after OHE transformation
ohe_features  = model["prep"].named_transformers_["cat"]["ohe"] \
                              .get_feature_names_out(cat_cols)
all_features  = num_cols + list(ohe_features)
```

---

## Performance & Memory Optimisation

```python
# Check memory
df.memory_usage(deep=True).sum() / 1e6   # MB

# Downcast numerics — use smaller dtypes
df[int_cols]   = df[int_cols].apply(pd.to_numeric, downcast="integer")
df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

# Category for low-cardinality strings (5–10× savings)
for col in cat_cols:
    if df[col].nunique() / len(df) < 0.1:   # < 10% unique values
        df[col] = df[col].astype("category")

# Read large files in chunks
chunks = []
for chunk in pd.read_csv("big.csv", chunksize=100_000):
    processed = chunk[chunk["value"] > 0]   # filter early
    chunks.append(processed)
df = pd.concat(chunks, ignore_index=True)

# Parquet — faster and type-preserving vs CSV
df.to_parquet("data.parquet")
df = pd.read_parquet("data.parquet", columns=["a","b"])  # column pruning

# Copy-on-Write (pandas 2.0+) — avoids SettingWithCopyWarning
pd.options.mode.copy_on_write = True

# eval and query — avoids temporary arrays (faster on large dfs)
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
pd.set_option("mode.chained_assignment", None)  # silence SettingWithCopy

# Context manager — temporary change
with pd.option_context("display.max_rows", 20):
    print(df)

# Reset to defaults
pd.reset_option("all")

# Recommended global settings
pd.options.display.float_format = "{:,.3f}".format
pd.options.mode.copy_on_write = True     # pandas 2.0+ best practice
```

---

## Quick Reference Card

| Category | Key functions |
|---|---|
| **I/O** | `read_csv` · `read_excel` · `read_json` · `read_parquet` · `read_sql` · `to_csv` · `to_excel` · `to_parquet` |
| **Select** | `df["col"]` · `df[["a","b"]]` · `.loc[label,col]` · `.iloc[pos,col]` · `.at` · `.iat` · `.query()` · boolean mask |
| **Inspect** | `.info()` · `.describe()` · `.dtypes` · `.shape` · `.nunique()` · `.value_counts()` |
| **Clean** | `isna` · `dropna` · `fillna` · `ffill` · `bfill` · `interpolate` · `replace` · `drop_duplicates` · `clip` |
| **Types** | `astype` · `to_numeric` · `to_datetime` · `convert_dtypes` |
| **Strings** | `.str.strip` · `.str.lower` · `.str.replace` · `.str.contains` · `.str.extract` · `.str.split` |
| **Datetime** | `.dt.year` · `.dt.month` · `.dt.dayofweek` · `.dt.day_name()` · `resample` · `date_range` |
| **GroupBy** | `groupby → agg / transform / filter / apply` · named agg: `agg(alias=("col","func"))` |
| **Merge** | `pd.merge(how=inner/left/right/outer/cross)` · `pd.concat(axis=0/1)` · `df.join()` |
| **Reshape** | `pivot_table` · `pivot` · `melt` · `stack` · `unstack` · `crosstab` |
| **Engineer** | `pd.cut` · `pd.qcut` · `get_dummies` · `shift` · `diff` · `pct_change` · `rolling` · `ewm` |
| **Scale** | `StandardScaler` · `MinMaxScaler` · `RobustScaler` · `PowerTransformer` · `np.log1p` |
| **Encode** | `LabelEncoder` · `OrdinalEncoder` · `OneHotEncoder` · `pd.get_dummies` |
| **ML Prep** | `train_test_split(stratify=y)` · `ColumnTransformer` · `Pipeline` · `SimpleImputer` · `KNNImputer` |
| **Optimise** | `astype("category")` · downcast · `read_parquet` · `chunksize` · `query` · `eval` · `copy_on_write` |

---

*Covers pandas 1.5–2.2. Docs: https://pandas.pydata.org/docs/*
