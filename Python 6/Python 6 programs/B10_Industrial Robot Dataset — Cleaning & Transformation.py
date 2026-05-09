import pandas as pd
import numpy as np
import os
import ast
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("SY-5, Kevin Victor, Roll No.-30")

CSV_INPUT = "robot_learning_logs.csv"
CSV_OUTPUT = "robot_logs_cleaned.csv"

# ---------------------------------------------------------
# GENERATE LARGE ROBOT DATASET
# ---------------------------------------------------------

def generate_robot_logs(num=300000):
    """
    Generates synthetic industrial robot learning logs.
    """

    np.random.seed(42)

    timestamps = pd.date_range(
        start="2026-01-01 00:00",
        periods=num,
        freq="s"
    )

    sensor_types = np.random.choice(
        ["voice", "camera", "haptic"],
        num,
        p=[0.2, 0.5, 0.3]
    )

    feature_vectors = [
        [round(np.random.random(), 3),
         round(np.random.random(), 3),
         round(np.random.random(), 3)]
        for _ in range(num)
    ]

    detected_cmds = np.random.choice(
        ["move", "grip", "release", "stop", ""],
        num,
        p=[0.15, 0.15, 0.15, 0.10, 0.45]
    )

    haptic_force = np.where(
        sensor_types == "haptic",
        np.random.uniform(0.0, 100.0, num).round(2),
        np.nan
    )

    objects_recognized = np.where(
        sensor_types == "camera",
        np.random.choice(
            ["box", "sphere", "cylinder", "none"],
            num
        ),
        ""
    )

    pos_x = np.random.uniform(-5.0, 5.0, num).round(3)
    pos_y = np.random.uniform(-5.0, 5.0, num).round(3)
    pos_z = np.random.uniform(0.0, 2.0, num).round(3)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_type": sensor_types,
        "feature_vector": feature_vectors,
        "detected_command": detected_cmds,
        "haptic_force": haptic_force,
        "object_detected": objects_recognized,
        "position_x": pos_x,
        "position_y": pos_y,
        "position_z": pos_z
    })

    # Introduce missing values
    for col in ["feature_vector", "detected_command", "haptic_force", "object_detected"]:
        df.loc[df.sample(frac=0.1).index, col] = np.nan

    df.to_csv(CSV_INPUT, index=False)
    return df


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

def load_dataset():
    if not os.path.exists(CSV_INPUT):
        print("Generating raw robot logs (please wait)...")
        generate_robot_logs()
    return pd.read_csv(CSV_INPUT)


# ---------------------------------------------------------
# CLEANING & TRANSFORMATION
# ---------------------------------------------------------

def clean_and_transform(df):

    print("\nStarting Data Cleaning & Transformation...")

    df_clean = df.copy()

    # -------------------
    # 1️⃣ Handle Missing Values
    # -------------------

    df_clean["haptic_force"] = df_clean["haptic_force"].fillna(
        df_clean["haptic_force"].median()
    )

    df_clean["position_x"] = df_clean["position_x"].fillna(
        df_clean["position_x"].mean()
    )

    df_clean["position_y"] = df_clean["position_y"].fillna(
        df_clean["position_y"].mean()
    )

    df_clean["position_z"] = df_clean["position_z"].fillna(
        df_clean["position_z"].mean()
    )

    df_clean["detected_command"] = df_clean["detected_command"].fillna("none")
    df_clean["object_detected"] = df_clean["object_detected"].fillna("none")

    # -------------------
    # 2️⃣ FIX FEATURE VECTOR PARSING
    # -------------------

    def parse_feature_vector(x):
        if pd.isna(x):
            return [0.0, 0.0, 0.0]
        try:
            return ast.literal_eval(x)
        except:
            return [0.0, 0.0, 0.0]

    df_clean["feature_vector"] = df_clean["feature_vector"].apply(parse_feature_vector)

    fv_df = pd.DataFrame(
        df_clean["feature_vector"].tolist(),
        columns=["fv_1", "fv_2", "fv_3"]
    )

    df_clean = pd.concat(
        [df_clean.drop(columns=["feature_vector"]), fv_df],
        axis=1
    )

    # -------------------
    # 3️⃣ One-Hot Encoding
    # -------------------

    cat_cols = ["sensor_type", "detected_command", "object_detected"]

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded = encoder.fit_transform(df_clean[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    df_clean = pd.concat(
        [df_clean.drop(columns=cat_cols), encoded_df],
        axis=1
    )

    # -------------------
    # 4️⃣ Normalize Numeric Columns
    # -------------------

    num_cols = [
        "haptic_force",
        "position_x",
        "position_y",
        "position_z",
        "fv_1",
        "fv_2",
        "fv_3"
    ]

    scaler = StandardScaler()
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    df_clean.to_csv(CSV_OUTPUT, index=False)

    print("Data cleaned, transformed, and saved to:", CSV_OUTPUT)

    return df_clean


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def main_menu():

    df = load_dataset()

    while True:
        print("\n" + "-"*70)
        print("Industrial Robot Dataset — Cleaning & Transformation")
        print("-"*70)
        print("1. Show Raw Sample Data")
        print("2. Show Missing Summary")
        print("3. Perform Cleaning & Transformation")
        print("4. Show Cleaned Sample")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            print("\n===== RAW SAMPLE =====\n")
            print(df.head(15).to_string(index=False))

        elif choice == "2":
            print("\n===== MISSING DATA SUMMARY =====\n")
            print(df.isna().sum())

        elif choice == "3":
            df_clean = clean_and_transform(df)

        elif choice == "4":
            if os.path.exists(CSV_OUTPUT):
                df_cleaned = pd.read_csv(CSV_OUTPUT)
                print("\n===== CLEANED SAMPLE =====\n")
                print(df_cleaned.head(15).to_string(index=False))
            else:
                print("Cleaned dataset not yet generated. Choose option 3 first.")

        elif choice == "5":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please select 1–5.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()