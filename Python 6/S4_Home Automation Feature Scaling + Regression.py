import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("SY-5, Kevin Victor, Roll No.-30")

# Show ALL columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---------------------------------------------------------
# SYNTHETIC HOME AUTOMATION DATASET
# ---------------------------------------------------------

def generate_home_automation_data(n=500):
    np.random.seed(42)

    voice_command_len = np.random.randint(1, 10, n)
    motion_count = np.random.randint(0, 30, n)
    ambient_light = np.random.uniform(0, 1000, n)
    temp_celsius = np.random.uniform(18, 30, n)
    sound_level = np.random.uniform(20, 90, n)
    kitchen_use = np.random.randint(0, 5, n)
    tv_use = np.random.randint(0, 5, n)
    smart_device_usage = np.random.randint(0, 20, n)

    energy_consumption = (
        0.3 * voice_command_len +
        0.5 * motion_count +
        0.2 * ambient_light / 100 +
        0.7 * temp_celsius +
        0.4 * sound_level / 10 +
        1.0 * kitchen_use +
        0.8 * tv_use +
        0.5 * smart_device_usage +
        np.random.normal(0, 1, n)
    ).round(2)

    df = pd.DataFrame({
        "voice_command_len": voice_command_len,
        "motion_count": motion_count,
        "ambient_light": ambient_light.round(2),
        "temp_celsius": temp_celsius.round(2),
        "sound_level": sound_level.round(2),
        "kitchen_use": kitchen_use,
        "tv_use": tv_use,
        "smart_device_usage": smart_device_usage,
        "energy_consumption": energy_consumption
    })

    return df


# ---------------------------------------------------------
# SCALE FEATURES + REGRESSION
# ---------------------------------------------------------

def scale_and_train(df, scaler_type="standard"):

    X = df.drop(columns=["energy_consumption"])
    y = df["energy_consumption"]

    print("\nOriginal Feature Ranges:")
    print(X.describe().loc[["min", "max"]])

    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
        print("\nUsing StandardScaler")
        print("Formula: Z = (X - Mean) / Std")

    else:
        scaler = MinMaxScaler()
        print("\nUsing MinMaxScaler")
        print("Formula: X_scaled = (X - Min) / (Max - Min)")

    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    print("\n--- Scaling Parameters ---")

    if scaler_type.lower() == "standard":
        scaling_info = pd.DataFrame({
            "Feature": X.columns,
            "Mean": scaler.mean_,
            "Std Dev": scaler.scale_
        })
        print(scaling_info)

    else:
        scaling_info = pd.DataFrame({
            "Feature": X.columns,
            "Min": scaler.data_min_,
            "Max": scaler.data_max_
        })
        print(scaling_info)

    print("\nFirst 5 Rows BEFORE Scaling:")
    print(X.head())

    print("\nFirst 5 Rows AFTER Scaling:")
    print(X_scaled_df.head())

    print("\nObservation:")
    print("After scaling, all features are brought to a similar range.")
    print("This prevents features like 'ambient_light' (0–1000)")
    print("from dominating features like 'kitchen_use' (0–4).")

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test) * 100

    print(f"\nModel Training Complete.")
    print(f"{scaler_type.title()} Scaling R² Score: {score:.2f}%")

# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def main_menu():
    df = generate_home_automation_data()

    while True:
        print("\n" + "-"*60)
        print("Home Automation Feature Scaling + Regression")
        print("-"*60)
        print("1. Show Sample Dataset")
        print("2. Show Statistical Summary")
        print("3. Scale Features Using StandardScaler + Train Model")
        print("4. Scale Features Using MinMaxScaler + Train Model")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            print("\nSample Data (first 10 rows):")
            print(df.head(10))

        elif choice == "2":
            print("\nSummary Statistics (All 9 Columns):")
            print(df.describe())

        elif choice == "3":
            scale_and_train(df, scaler_type="standard")

        elif choice == "4":
            scale_and_train(df, scaler_type="minmax")

        elif choice == "5":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Try again.")

# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()