# Python Data Science
### Data Analysis | Visualization | Preprocessing & Feature Engineering

**Author:** Kevin Victor
**Scope:** Consolidated reference across three laboratory modules
**Status:** Educational & Applied

---

## Module Overview

This module covers three progressive domains of the Python data science workflow, from raw data understanding through visual communication to machine learning–ready preprocessing.

| Module | Focus Area |
|---|---|
| **Module 4** | Data Analysis — Descriptive Statistics, EDA, Missing Values, Memory Optimization |
| **Module 5** | Data Visualization — Matplotlib & Seaborn Charts, Outlier Detection, Correlation Analysis |
| **Module 6** | Data Preprocessing — Imputation, Encoding, Outlier Treatment, Scaling, Chunked Processing, Validation |

The unifying principle: raw data must be *understood*, *visualized*, and *cleaned* before it can produce reliable analytical or predictive outputs.

---

## Module 4 — Data Analysis with Pandas & NumPy

### Core Concepts

**Descriptive Statistics (Mean, Median, Mode)** are the three measures of central tendency. Mean is sensitive to outliers; median is robust to them; mode is the only measure applicable to categorical data. Comparing all three reveals distribution shape — symmetric, right-skewed, or left-skewed — which informs how data should be treated downstream.

**`describe()`** generates an eight-number statistical profile for every numerical column in a single call: count, mean, standard deviation, min, Q1, median, Q3, and max. The IQR (Q3 − Q1) and the mean-to-median gap are immediately readable from this output, making it the standard starting point for any exploratory data analysis.

**Univariate Analysis** examines each variable in isolation — its central tendency, spread, distribution shape, and range — before studying inter-variable relationships. Histograms complement point statistics by making distribution shape, multimodality, and tail behavior visible.

**Missing Value Detection** uses `df.isna()` to produce a boolean mask over the DataFrame. Column-wise summation yields missing counts per feature; `df[df.isna().any(axis=1)]` identifies incomplete records. Overall completeness is computed as `(total_cells − missing_cells) / total_cells × 100` — a single quality metric trackable over time.

**Categorical Frequency Analysis** applies `value_counts()` and `mode()` to string-valued columns where arithmetic statistics are undefined. Weighted random generation via `np.random.choice(..., p=probabilities)` produces realistic frequency distributions for demonstration.

**DataFrame Memory Optimization** reduces memory consumption by converting default 64-bit dtypes to smaller equivalents: `float64 → float32`, `int64 → int32`, and `object → category`. The `category` dtype stores each unique string once and encodes occurrences as integers — particularly impactful for low-cardinality string columns. `df.memory_usage(deep=True)` measures true memory usage including object-dtype string content. Reductions of 50–75% are common at scale.

**Grouped Aggregation and Correlation** extend analysis to inter-variable relationships. `df.groupby().agg()` produces per-group summary statistics; `df.corr()` computes Pearson pairwise correlation across all numerical columns simultaneously.

### Industrial Use Cases

| Domain | Pattern Applied |
|---|---|
| Meteorology / Climate Analytics | Mean, median, mode on time-series weather data — mirrors automated report generation in weather APIs |
| Semiconductor Manufacturing (SPC) | `describe()` on fab KPIs — mirrors statistical process control in MES systems |
| Materials Science / Research | Univariate analysis with reproducible `np.random.seed()` — standard research data validation |
| Retail / Business Intelligence | `groupby()` aggregation by category — core operation of BI dashboards |
| Healthcare / Data Quality | Missing value completeness tracking — mirrors regulatory data quality assurance |
| ML Engineering / MLOps | dtype optimization on million-row datasets — standard practice before GPU training |

---

## Module 5 — Data Visualization with Matplotlib & Seaborn

### Core Concepts

**Bar Charts** compare discrete categorical values by encoding magnitude as bar height. Gridlines on the value axis (`axis="y"`) aid comparison; uniform zero baseline is required. Seaborn's `barplot()` accepts DataFrames directly with named column mappings and perceptually uniform palettes (`viridis`).

**Scatter Plots** reveal relationships between two continuous variables. Point annotations identify individual observations when the dataset is small enough to avoid label overlap. The absence of a pattern in a scatter plot is itself a finding — it rules out a linear or monotonic relationship.

**Line Graphs** convey change over ordered sequences (typically time). Multi-variable line charts require feature scaling (`MinMaxScaler`) when variables differ in units or magnitude. The `LineCollection` + `LinearSegmentedColormap` technique encodes temporal progression as a color gradient along the line itself — dual-encoding time as both position and color.

**Count Plots** summarize the frequency distribution of binary outcomes across categorical groups. `sns.countplot()` counts occurrences automatically from a long-format DataFrame. `DataFrame.melt()` reshapes wide boolean outlier flags into the long format required by Seaborn.

**Box Plots** display the five-number summary (min, Q1, median, Q3, max) with outliers as isolated markers beyond the 1.5×IQR fences. A logarithmic scale (`plt.xscale("log")`) is essential when data spans multiple orders of magnitude, preventing compression of the main distribution.

**Correlation Heatmaps** render `df.corr()` as a color grid using `sns.heatmap()`. The `coolwarm` diverging colormap is standard — it centers at zero (no correlation), with warm colors for positive and cool colors for negative relationships. `annot=True` adds numerical values for precision alongside color encoding.

**Outlier Detection in Visualizations** uses two methods: Z-score (`scipy.stats.zscore()`, threshold = 3 standard deviations) for approximately normal data, and IQR (`Q1 − 1.5×IQR`, `Q3 + 1.5×IQR`) for distribution-agnostic detection. Both can be visualized through count plots and box plots respectively.

### Industrial Use Cases

| Domain | Pattern Applied |
|---|---|
| Aviation Operations | Bar charts of route durations — mirrors route efficiency reporting in airline analytics |
| Digital / Social Media Analytics | Annotated scatter plots of engagement vs. demographics — mirrors brand tracking studies |
| Climate Science | Gradient line graphs with `LineCollection` — mirrors IPCC multi-variable trend visualizations |
| Industrial IoT / Acoustic Monitoring | Count plots of Z-score outlier flags — mirrors anomaly classification in condition monitoring |
| FinTech / Fraud Detection | Box plots with log scale + IQR outlier bounds — standard in financial transaction monitoring |
| Product Analytics / UX Research | Correlation heatmaps of feature scores vs. sales — mirrors feature prioritization in product analytics |
| Sales / Business Intelligence | Bar + pie chart combinations for regional distribution — standard in BI dashboards |

---

## Module 6 — Data Preprocessing & Feature Engineering

### Core Concepts

**Missing Value Imputation** fills absent entries before model training. Mean imputation is appropriate for normally distributed numerical columns; median is preferred for skewed distributions or when outliers are present; mode is the only valid strategy for categorical columns. All implementations operate on `df.copy()` to preserve the original data. Advanced method: `sklearn.impute.IterativeImputer` (MICE) estimates missing values from all other features jointly.

**Categorical Encoding** converts string labels to numbers for model compatibility. Label encoding assigns a sorted integer to each unique value — appropriate only for genuinely ordinal categories or classification targets, as it introduces a spurious numeric ordering. One-hot encoding (`OneHotEncoder`) creates one binary column per unique category, eliminating ordinal implication. `handle_unknown="ignore"` prevents errors on unseen categories at inference time; `get_feature_names_out()` generates self-documenting column names.

**Outlier Detection and Treatment** applies two statistical methods. The IQR method (Q1 − 1.5×IQR, Q3 + 1.5×IQR) is non-parametric and robust to non-normal distributions. The Z-score method (threshold = 3) assumes approximate normality. Treatment options are: clipping (constrains values to a percentile boundary, preserving all rows) and median replacement (substitutes physically impossible values with a central estimate). The choice between them is domain-dependent.

**Feature Scaling** normalizes numerical ranges before training distance-based or gradient-based models (linear regression, SVM, neural networks). StandardScaler (Z-score normalization: mean = 0, std = 1) is preferred for normally distributed features. MinMaxScaler (range = [0, 1]) is appropriate for bounded uniform distributions or algorithms requiring inputs in a fixed range. Scalers must be fitted exclusively on training data to prevent data leakage.

**Chunked Processing** reads large files incrementally using `pd.read_csv(file_path, chunksize=N)`, holding only N rows in memory at a time. Global statistics are accumulated across chunks: running sum and count for mean; `min()`/`max()` comparisons for bounds; dictionary accumulators for grouped sums. `chunk.memory_usage(deep=True)` quantifies the per-chunk memory footprint directly.

**Data Validation Pipelines** verify correctness against domain-defined rules before model training. Range checks (`voltage: 100–300V`, `frequency: 47–53 Hz`) apply physical or regulatory constraints — distinct from statistical outlier detection, which is data-relative. The correct cleaning sequence is: impute missing values → remove duplicates → apply range corrections → treat statistical outliers. Sequencing matters: later steps depend on the completeness and plausibility of data from prior steps.

**Feature Vector Parsing** addresses lists stored as strings in CSV cells. `ast.literal_eval()` safely parses the string back to a Python list; `pd.DataFrame(col.tolist(), columns=[...])` expands list values into separate numerical columns efficiently. A `try-except` wrapper with a zero-vector fallback handles malformed records without aborting the pipeline.

### Industrial Use Cases

| Domain | Pattern Applied |
|---|---|
| Network Operations / ISP | Mean/median/mode imputation on routing logs — mirrors pre-processing for traffic analytics models |
| Industrial Robotics | One-hot encoding + feature vector parsing + scaling — mirrors multi-modal sensor fusion pipelines |
| Autonomous Vehicles / ADAS | IQR + Z-score outlier detection and treatment — mirrors safety-critical sensor data validation |
| Smart Home / Energy Management | StandardScaler + MinMaxScaler before regression — standard in building energy prediction models |
| Power Grid Operations | Chunked aggregation on 1M-row grid sensor logs — mirrors big data processing in SCADA systems |
| Grid Operations / ML Pre-Training | Full validation pipeline with domain-range rules — mirrors data contracts in production ML platforms |

---

## Future Industry-Grade Extensions

The following upgrade paths apply across all three modules and reflect standard engineering concerns when moving from working demonstrations to production-scale data systems.

**Storage and Loading Infrastructure:** Replace CSV files with Apache Parquet (columnar, compressed, supports selective column loading). Connect to cloud data warehouses (BigQuery, Snowflake, Redshift) via `pd.read_sql()`. Version datasets with DVC alongside code to ensure analytical reproducibility.

**Advanced Statistical Analysis:** Extend beyond mean/median/mode with SciPy normality tests (Shapiro-Wilk), non-parametric tests (Mann-Whitney U), confidence intervals, and bootstrap resampling. Use `ydata-profiling` for automated EDA report generation in a single function call.

**Advanced Imputation:** Replace single-value imputation with `IterativeImputer` (MICE) for context-aware imputation, or `KNNImputer` for neighbor-based estimation. Add binary missing-indicator columns to preserve the predictive signal that a value was absent. For time-series data, use forward-fill, backward-fill, or linear interpolation.

**High-Cardinality Encoding:** For features with many unique categories, use target encoding, frequency encoding, or neural network embedding layers in place of one-hot encoding, which creates dimensionality problems at scale.

**Scikit-learn Pipeline API:** Chain all preprocessing steps (imputation, encoding, scaling) and the model into a single `Pipeline` object using `ColumnTransformer`. This guarantees correct train/test isolation — all fitted transformers are fit on training data only — and enables the entire pipeline to be serialized with `joblib.dump()` for production deployment.

**Interactive Visualization:** Replace static Matplotlib/Seaborn figures with Plotly Express for hover, zoom, and filter capabilities. Wrap in Streamlit or Dash for interactive browser-based dashboards with filter widgets and drill-down charts. Use Bokeh for linked brushing in high-performance large-dataset contexts.

**Advanced Chart Types:** Add violin plots (distribution shape + box plot in one), faceted grids (`seaborn.FacetGrid`) for multi-group comparisons, geographic choropleth maps for regional data, and animated time-series for progressive trend communication.

**Automated Reporting and Scheduling:** Schedule analytical pipelines with Apache Airflow. Generate PDF/HTML reports automatically with `matplotlib.savefig()` + `reportlab`. Integrate monitoring dashboards with Grafana for real-time refresh from InfluxDB or Prometheus.

**Distributed Processing at Scale:** For datasets exceeding available RAM, migrate chunked Pandas processing to Dask (minimal code changes, parallel execution) or Apache Spark (PySpark, for terabyte–petabyte scale, with native `filter()`, `map()`, and `groupBy().agg()` equivalents).

**Formal Data Validation:** Replace custom range-check dictionaries with Great Expectations or Pandera schema definitions. Formalize data quality requirements as data contracts — machine-enforceable agreements between data producers and consumers, with automated alerting on violations.

**Colormap Accessibility:** Use perceptually uniform colormaps (`viridis`, `cividis`) rather than rainbow or jet. Evaluate designs against color blindness simulations. Respect domain color conventions (red = loss in finance; red = alert in monitoring).

---

## Concept-to-Production Mapping

| Demonstrated Concept | Production Equivalent |
|---|---|
| `df.describe()` on KPI columns | Statistical process control charts in MES / SCADA |
| `df.isna().sum()` completeness reporting | Data quality dashboard in production ingestion pipeline |
| `astype("float32")` / `astype("category")` | Pre-GPU dtype optimization in ML training infrastructure |
| `df.groupby().agg()` | SQL `GROUP BY` in BI warehouse; Spark `groupBy().agg()` |
| `df.corr()` heatmap | Feature selection and collinearity analysis in ML pipelines |
| `sns.countplot()` on outlier flags | Anomaly classification dashboard in condition monitoring |
| Box plot + IQR outlier detection | Transaction monitoring in FinTech fraud detection systems |
| `LineCollection` gradient line | Multi-variable temporal trend visualization in scientific publications |
| Median imputation + `df.copy()` | `sklearn.SimpleImputer(strategy="median")` in Scikit-learn pipelines |
| `LabelEncoder` / `OneHotEncoder` | `ColumnTransformer` encoding stage in production ML pipelines |
| IQR clipping / median replacement | Sensor data sanitization in ADAS and robotics preprocessing |
| `StandardScaler` / `MinMaxScaler` | Scaling stage in Scikit-learn `Pipeline`; feature normalization before neural network training |
| `pd.read_csv(chunksize=N)` | PySpark partitioned DataFrame; Dask chunked processing |
| Range validation dictionary | Great Expectations / Pandera schema enforcement; data contracts |

---

## Summary

These three modules collectively cover the foundational data science workflow in Python — from understanding what a dataset contains, to communicating patterns visually, to transforming raw data into a machine learning–ready form. The tools demonstrated — Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn preprocessing — are not educational alternatives to production tools. They are the production tools, used at every scale from startup analytics to enterprise ML infrastructure. The operations remain conceptually identical whether applied to 28 rows of weather data or 100 million rows in a distributed warehouse; the underlying engineering thinking does not change with scale.

---

