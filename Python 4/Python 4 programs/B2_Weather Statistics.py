import pandas as pd
import numpy as np
from scipy import stats
import random
print("SY-5, Kevin Victor, Roll No.-30")
# Simulate weather data for Pune in Feb 2025
def generate_pune_february_data():
    days = list(range(1, 29))  # Feb 1 - Feb 28

    data = {
        "Day": days,
        "MaxTemp": [round(random.uniform(28.0, 34.0),1) for _ in days],
        "MinTemp": [round(random.uniform(11.0, 20.0),1) for _ in days],
        "Humidity": [round(random.uniform(30, 60),1) for _ in days],
        "WindSpeed": [round(random.uniform(5, 22),1) for _ in days],
        "CloudCover": [random.choice([0,10,25,40,55,70]) for _ in days]
    }
    return pd.DataFrame(data)

# Function to compute mean, median, mode
def compute_statistics(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats_report = []

    for col in numeric_cols:
        col_data = df[col]
        mean_val = col_data.mean()
        median_val = col_data.median()
        try:
            mode_val = stats.mode(col_data).mode[0]
        except Exception:
            mode_val = np.nan

        stats_report.append({
            "Field": col,
            "Mean": round(mean_val,2),
            "Median": round(median_val,2),
            "Mode": round(mode_val,2) if not np.isnan(mode_val) else "N/A"
        })

    return pd.DataFrame(stats_report)

# CLI Menu
def menu():
    df = None
    while True:
        print("\n===== PUNE FEBRUARY 2025 WEATHER STATS =====")
        print("1. Generate Weather Data")
        print("2. View Weather Dataset")
        print("3. Compute Mean, Median, Mode Statistics")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            df = generate_pune_february_data()
            print("\n✔ Weather data for Pune February 2025 created.")

        elif choice == '2':
            if df is None:
                print("\n❗ Please generate the data first.")
            else:
                print("\n--- Pune February Weather Sample ---")
                print(df.head(10).to_string(index=False))

        elif choice == '3':
            if df is None:
                print("\n❗ Please generate the data first.")
            else:
                stats_table = compute_statistics(df)
                print("\n=== WEATHER STATISTICS (Numerical Columns) ===")
                print(stats_table.to_string(index=False))

                print("\n--- Brief Analysis ---")
                avg_max = stats_table.loc[stats_table['Field']=="MaxTemp","Mean"].values[0]
                avg_min = stats_table.loc[stats_table['Field']=="MinTemp","Mean"].values[0]
                avg_hum = stats_table.loc[stats_table['Field']=="Humidity","Mean"].values[0]

                print(f"Average high temperatures are around {avg_max}°C, indicating warm daytime conditions.")
                print(f"Average low temperatures are around {avg_min}°C, indicating fairly cool nights typical of Pune in February.")
                print(f"Mean humidity near {avg_hum}% suggests moderately dry conditions.")

        elif choice == '4':
            print("\nExiting. Stay weather-aware!")
            break
        else:
            print("\n⚠ Invalid choice — try again.")

if __name__ == "__main__":
    menu()