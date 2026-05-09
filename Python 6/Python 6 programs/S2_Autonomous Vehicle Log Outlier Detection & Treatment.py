import pandas as pd
import numpy as np
from scipy import stats

print("SY-5, Kevin Victor, Roll No.-30")

CSV_FILENAME = "autonomous_vehicle_logs.csv"
CSV_CLEANED = "autonomous_vehicle_cleaned.csv"

# ---------------------------------------------------------
# GENERATE SIMULATED ADAS LOG DATA
# ---------------------------------------------------------

def generate_av_logs(num=2000):
    np.random.seed(42)

    timestamps = pd.date_range("2026-01-01", periods=num, freq="S")

    # Normal operating ranges (approx)
    speed = np.random.normal(60, 10, num)        # km/h
    front_dist = np.random.normal(50, 15, num)   # meters
    side_dist = np.random.normal(25, 8, num)     # meters
    lidar = np.random.normal(48, 12, num)
    radar = np.random.normal(0, 5, num)

    steering = np.random.normal(0, 5, num)       # degrees
    brake = np.random.uniform(0, 80, num)        # psi
    throttle = np.random.uniform(0, 100, num)    # %

    df = pd.DataFrame({
        "timestamp": timestamps,
        "speed_kmh": speed,
        "front_cam_distance": front_dist,
        "side_cam_distance": side_dist,
        "lidar_mean": lidar,
        "radar_rel_speed": radar,
        "steering_angle": steering,
        "brake_pressure": brake,
        "throttle_position": throttle
    })

    # Introduce extreme outliers
    out_idx = np.random.choice(df.index, size=int(0.03 * num), replace=False)
    df.loc[out_idx, "speed_kmh"] = df["speed_kmh"].mean() + 6 * df["speed_kmh"].std()
    df.loc[out_idx, "lidar_mean"] = df["lidar_mean"].mean() - 6 * df["lidar_mean"].std()
    df.loc[out_idx, "steering_angle"] = 1000  # impossible angle
    df.loc[out_idx, "brake_pressure"] = 200  # beyond physical
    return df


# ---------------------------------------------------------
# OUTLIER DETECTION & TREATMENT
# ---------------------------------------------------------

def detect_outliers_iqr(df):
    print("\nIQR Outlier Results:")
    numeric = df.select_dtypes(include=[np.number])
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((numeric < lower) | (numeric > upper)).sum()
    print(outliers)


def detect_outliers_zscore(df, thresh=3):
    numeric = df.select_dtypes(include=[np.number])
    z_scores = np.abs(stats.zscore(numeric))
    outlier_flags = (z_scores > thresh).sum(axis=0)
    print("\nZ-score Outlier Counts (|z|>3):")
    print(outlier_flags)


def treat_outliers_clip(df):
    df_t = df.copy()
    numeric = df_t.select_dtypes(include=[np.number])

    low = numeric.quantile(0.01)
    high = numeric.quantile(0.99)

    for col in numeric.columns:
        df_t[col] = df_t[col].clip(lower=low[col], upper=high[col])

    df_t.to_csv(CSV_CLEANED, index=False)
    print("Outliers treated by clipping & saved to:", CSV_CLEANED)
    return df_t


def treat_outliers_median(df):
    df_t = df.copy()
    numeric = df_t.select_dtypes(include=[np.number])
    med = numeric.median()
    for col in numeric.columns:
        df_t[col] = np.where(
            (df_t[col] < numeric[col].quantile(0.05)) |
            (df_t[col] > numeric[col].quantile(0.95)),
            med[col],
            df_t[col]
        )
    df_t.to_csv(CSV_CLEANED, index=False)
    print("Outliers treated by median replacement & saved to:", CSV_CLEANED)
    return df_t


# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def main_menu():
    df = generate_av_logs()
    df.to_csv(CSV_FILENAME, index=False)

    while True:
        print("\n" + "="*70)
        print("Autonomous Vehicle Log Outlier Detection & Treatment")
        print("="*70)
        print("1. Show Raw Sample")
        print("2. Detect Outliers (IQR Method)")
        print("3. Detect Outliers (Z-score Method)")
        print("4. Treat Outliers by Clipping")
        print("5. Treat Outliers by Median Replacement")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == "1":
            print("\nRaw Sample Data (first 10 rows):")
            print(df.head(10).to_string(index=False))

        elif choice == "2":
            detect_outliers_iqr(df)

        elif choice == "3":
            detect_outliers_zscore(df)

        elif choice == "4":
            treat_outliers_clip(df)

        elif choice == "5":
            treat_outliers_median(df)

        elif choice == "6":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Try again.")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    main_menu()