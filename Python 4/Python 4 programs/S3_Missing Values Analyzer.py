import pandas as pd
import numpy as np
import os
print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# DATA GENERATION WITH MISSING VALUES
# ---------------------------------------------------------

def generate_survey_dataset(num=100):

    np.random.seed(42)

    # Categorical fields
    research_domains = [
        "Model Optimization",
        "MoE Systems",
        "Sustainability in AI",
        "Multi-Agent Systems",
        "Autonomous Robotics",
        "Industrial Automation",
        "Physical AI"
    ]

    institution_types = [
        "University",
        "Industry Lab",
        "Government Lab",
        "Independent Researcher"
    ]

    data = {
        "Researcher_ID": np.arange(1, num + 1),
        "Research_Domain": np.random.choice(research_domains, num),
        "Model_Accuracy": np.round(np.random.uniform(50, 99, num), 2),
        "Uses_MoE": np.random.choice(["Yes", "No"], num),
        "Sustainability_Score": np.round(np.random.uniform(0, 10, num), 1),
        "Multi_Agent_Score": np.round(np.random.uniform(0, 10, num), 1),
        "Autonomy_Level": np.random.choice(["Low", "Medium", "High"], num),
        "Experience_Years": np.random.randint(0, 25, num),
        "Institution_Type": np.random.choice(institution_types, num)
    }

    df = pd.DataFrame(data)

    # Introduce missing values randomly
    for col in [
        "Model_Accuracy",
        "Uses_MoE",
        "Sustainability_Score",
        "Multi_Agent_Score",
        "Autonomy_Level",
        "Institution_Type"
    ]:
        df.loc[df.sample(frac=0.1).index, col] = np.nan

    return df


# ---------------------------------------------------------
# MISSING VALUE REPORTING
# ---------------------------------------------------------

def report_missing_values(df):

    print("\nMissing Values Summary:\n")
    total_missing = df.isna().sum()
    percent_missing = (df.isna().mean() * 100).round(2)

    report = pd.DataFrame({
        "Total Missing": total_missing,
        "Percent Missing": percent_missing
    })

    print(report)

    print("\nRows with Any Missing Values:\n")
    missing_rows = df[df.isna().any(axis=1)]
    print(missing_rows)


# ---------------------------------------------------------
# DETAILED COLUMN ANALYSIS
# ---------------------------------------------------------

def column_analysis(df):

    print("\nDetailed Column Summary:\n")
    for col in df.columns:
        print(f"Column: {col}")
        print("Data Type:", df[col].dtype)
        print("Unique Values (top 10):", df[col].dropna().unique()[:10])
        print("Missing Count:", df[col].isna().sum())
        print("-" * 40)


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def main_menu():
    df = generate_survey_dataset()

    while True:
        clear()
        print("AI/ML Research Survey Missing Values Analyzer")
        print("------------------------------------------------")
        print("1. Show Full Dataset")
        print("2. Report Missing Values")
        print("3. Detailed Column Analysis")
        print("4. Show Statistical Summary")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            print("\nFull Dataset:\n")
            print(df)
            input("\nPress Enter to continue...")

        elif choice == "2":
            report_missing_values(df)
            input("\nPress Enter to continue...")

        elif choice == "3":
            column_analysis(df)
            input("\nPress Enter to continue...")

        elif choice == "4":
            print("\nStatistical Description (Numerical Columns):\n")
            print(df.describe())
            input("\nPress Enter to continue...")

        elif choice == "5":
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Try again.")
            input("\nPress Enter to continue...")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()