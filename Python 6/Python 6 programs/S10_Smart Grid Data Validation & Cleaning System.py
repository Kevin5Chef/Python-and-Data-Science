import pandas as pd
import numpy as np
import os
from scipy import stats


print("SY-5, Kevin Victor, Roll No.-30")

ORIGINAL_CSV = "smart_grid_logs.csv"
CORRUPTED_CSV = "smart_grid_logs_corrupted.csv"
CLEANED_CSV = "smart_grid_logs_cleaned.csv"

# ---------------------------------------------------------
# LOAD ORIGINAL DATA
# ---------------------------------------------------------

def load_original():
    if not os.path.exists(ORIGINAL_CSV):
        print("Original dataset not found!")
        return None
    return pd.read_csv(ORIGINAL_CSV, parse_dates=["timestamp"])

# ---------------------------------------------------------
# INJECT REAL-WORLD ERRORS (~7%)
# ---------------------------------------------------------

def inject_data_issues(df, corruption_rate=0.07):
    df = df.copy()
    n = len(df)
    num_corrupt = int(n * corruption_rate)

    # Remove existing CLEANED file
    if os.path.exists(CLEANED_CSV):
        os.remove(CLEANED_CSV)

    print("\nInjecting artificial real-world issues...")
    print(f"Rows to corrupt (~{corruption_rate*100:.1f}%): {num_corrupt}")

    # 1) Inject NaNs
    for col in ["voltage", "current", "power", "frequency", "load"]:
        indices = np.random.choice(n, num_corrupt, replace=False)
        df.loc[indices, col] = np.nan

    # 2) Inject Outliers
    indices = np.random.choice(n, num_corrupt, replace=False)
    df.loc[indices, "voltage"] = np.random.choice([50, 500, 800], size=num_corrupt)

    indices = np.random.choice(n, num_corrupt, replace=False)
    df.loc[indices, "frequency"] = np.random.choice([30, 70], size=num_corrupt)

    indices = np.random.choice(n, num_corrupt, replace=False)
    df.loc[indices, "load"] = -np.random.uniform(10, 50, num_corrupt)

    # 3) Duplicate rows
    duplicate_sample = df.sample(frac=0.03)
    df = pd.concat([df, duplicate_sample], ignore_index=True)

    df.to_csv(CORRUPTED_CSV, index=False)
    print("Corrupted data created.")
    return df

# ---------------------------------------------------------
# DETECTION FUNCTIONS
# ---------------------------------------------------------

def check_missing_values(df):
    print("\n--- Missing Value Check ---")
    print(df.isna().sum())

def check_duplicates(df):
    print("\n--- Duplicate Row Check ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")

def check_ranges(df):
    print("\n--- Range Sanity Check ---")
    rules = {
        "voltage": (100, 300),
        "current": (0, 150),
        "power": (0, 50000),
        "load": (0, 150),
        "frequency": (47, 53)
    }
    for col, (low, high) in rules.items():
        below = df[df[col] < low].shape[0]
        above = df[df[col] > high].shape[0]
        print(f"{col}: below {low}: {below}, above {high}: {above}")

def detect_outliers_iqr(df):
    print("\n--- IQR Outlier Detection ---")
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["load"])
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    for col in numeric.columns:
        count = ((numeric[col] < lower[col]) | (numeric[col] > upper[col])).sum()
        print(f"{col}: {count} outliers")

def detect_outliers_zscore(df, thresh=3):
    print("\n--- Z-score Outlier Detection ---")
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["load"])
    z_scores = np.abs(stats.zscore(numeric, nan_policy="omit"))
    outlier_counts = (z_scores > thresh).sum(axis=0)
    for col, count in zip(numeric.columns, outlier_counts):
        print(f"{col}: {count} extreme values")

# ---------------------------------------------------------
# CLEANING / CORRECTION FUNCTIONS
# ---------------------------------------------------------

def clean_missing_values(df):
    df_clean = df.copy()
    # Impute numeric features with median (robust)
    for col in ["voltage", "current", "power", "frequency", "load"]:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
    return df_clean

def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)

def correct_ranges(df):
    df_clean = df.copy()
    # Clip to realistic grid bounds
    df_clean["voltage"] = df_clean["voltage"].clip(100, 300)
    df_clean["current"] = df_clean["current"].clip(0, 150)
    df_clean["power"] = df_clean["power"].clip(0, 50000)
    df_clean["load"] = df_clean["load"].clip(0, 150)
    df_clean["frequency"] = df_clean["frequency"].clip(47, 53)
    return df_clean

def treat_outliers_median(df):
    df_clean = df.copy()
    numeric = df_clean.select_dtypes(include=[np.number]).columns
    # Replace extreme values with median
    for col in numeric:
        med = df_clean[col].median()
        # 5th–95th percentile for realistic bounds
        lower = df_clean[col].quantile(0.05)
        upper = df_clean[col].quantile(0.95)
        df_clean[col] = np.where(
            (df_clean[col] < lower) | (df_clean[col] > upper),
            med,
            df_clean[col]
        )
    return df_clean

def full_cleaning_pipeline(df):
    print("\nRunning full data cleaning pipeline...")

    df1 = clean_missing_values(df)
    print("✔ Missing values handled (median imputation)")

    df2 = remove_duplicates(df1)
    print("✔ Duplicate rows removed")

    df3 = correct_ranges(df2)
    print("✔ Values clipped to realistic ranges")

    df4 = treat_outliers_median(df3)
    print("✔ Outliers corrected via median-based trimming")

    df4.to_csv(CLEANED_CSV, index=False)
    print(f"\nCleaned dataset stored in {CLEANED_CSV}")

    return df4

# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def main_menu():
    df_original = load_original()
    if df_original is None:
        return

    df_corrupted = None
    df_cleaned = None

    while True:
        print("\n" + "-"*75)
        print("Smart Grid Data Validation & Cleaning System")
        print("-"*75)
        print("1. Inject artificial data defects (~7%)")
        print("2. Show missing values")
        print("3. Show duplicate rows")
        print("4. Range checks (realistic grid bounds)")
        print("5. Outlier detection (IQR)")
        print("6. Outlier detection (Z-score)")
        print("7. Run full cleaning pipeline")
        print("8. Show cleaned sample")
        print("9. Exit")

        choice = input("\nEnter your choice (1-9): ")

        if choice == "1":
            df_corrupted = inject_data_issues(df_original)

        elif choice in ["2", "3", "4", "5", "6"]:
            if df_corrupted is None:
                print("Please inject data defects first (option 1).")
                continue
            if choice == "2":
                check_missing_values(df_corrupted)
            elif choice == "3":
                check_duplicates(df_corrupted)
            elif choice == "4":
                check_ranges(df_corrupted)
            elif choice == "5":
                detect_outliers_iqr(df_corrupted)
            elif choice == "6":
                detect_outliers_zscore(df_corrupted)

        elif choice == "7":
            if df_corrupted is None:
                print("Please corrupt data first (option 1).")
            else:
                df_cleaned = full_cleaning_pipeline(df_corrupted)

        elif choice == "8":
            if df_cleaned is None:
                print("Dataset not cleaned yet — run option 7 first.")
            else:
                print("\n==== Cleaned Dataset Sample ====\n")
                print(df_cleaned.head(15).to_string(index=False))

        elif choice == "9":
            print("Exiting system.")
            break

        else:
            print("Invalid choice. Try again.")

# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()