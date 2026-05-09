import pandas as pd
import numpy as np

print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# DATA GENERATION – ELITE ATHLETE FITNESS RECORDS
# ---------------------------------------------------------

def generate_athlete_data():
    np.random.seed(42)

    data = {
        "Athlete_ID": range(1, 16),
        "Name": [
            "Arjun", "Mira", "Kabir", "Isha", "Rohan",
            "Ananya", "Dev", "Neha", "Vihaan", "Sara",
            "Aryan", "Tara", "Reyansh", "Aditi", "Krish"
        ],
        "Sport": [
            "Sprint", "Swimming", "Football", "Tennis", "Cycling",
            "Boxing", "Marathon", "Gymnastics", "Cricket", "Hockey",
            "Basketball", "Rowing", "Badminton", "Wrestling", "Skating"
        ],
        "Age": np.random.randint(18, 35, 15),
        "VO2_Max (ml/kg/min)": np.random.uniform(50, 75, 15).round(2),
        "Resting_Heart_Rate (bpm)": np.random.randint(45, 65, 15),
        "Body_Fat (%)": np.random.uniform(8, 18, 15).round(2),
        "Sprint_100m_Time (sec)": np.random.uniform(10, 15, 15).round(2),
        "Weekly_Training_Hours": np.random.randint(10, 30, 15)
    }

    df = pd.DataFrame(data)

    # ---------------------------------------------------------
    # DELIBERATELY INTRODUCE INCOMPLETE RECORDS
    # ---------------------------------------------------------
    df.loc[3, "VO2_Max (ml/kg/min)"] = np.nan
    df.loc[7, "Body_Fat (%)"] = np.nan
    df.loc[10, "Resting_Heart_Rate (bpm)"] = np.nan
    df.loc[12, "Sprint_100m_Time (sec)"] = np.nan
    df.loc[5, "Weekly_Training_Hours"] = np.nan

    return df


# ---------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------

def show_full_dataset(df):
    print("\n===== ELITE ATHLETE FITNESS DATASET =====\n")
    print(df.to_string(index=False))


def detect_incomplete_records(df):
    print("\n===== INCOMPLETE RECORDS =====\n")
    incomplete = df[df.isnull().any(axis=1)]

    if incomplete.empty:
        print("No incomplete records found.")
    else:
        print(incomplete.to_string(index=False))


def show_missing_summary(df):
    print("\n===== MISSING VALUES SUMMARY (Column-wise) =====\n")
    missing = df.isnull().sum()
    print(missing.to_string())


def show_completion_percentage(df):
    print("\n===== DATA COMPLETENESS =====\n")
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    completion = ((total_cells - missing_cells) / total_cells) * 100
    print(f"Dataset Completion: {completion:.2f}%")
    print(f"Missing Cells: {missing_cells}")


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def menu():
    df = generate_athlete_data()

    while True:
        print("\n" + "-" * 65)
        print("HEALTHCARE DATA QUALITY CHECK – ELITE ATHLETES")
        print("-" * 65)
        print("1. Display Full Dataset")
        print("2. Detect Incomplete Records (Row-wise)")
        print("3. Show Missing Values Summary (Column-wise)")
        print("4. Show Dataset Completion Percentage")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            show_full_dataset(df)

        elif choice == "2":
            detect_incomplete_records(df)

        elif choice == "3":
            show_missing_summary(df)

        elif choice == "4":
            show_completion_percentage(df)

        elif choice == "5":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please select between 1 and 5.")


# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    menu()