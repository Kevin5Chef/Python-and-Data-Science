import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("SY-5, Kevin Victor, Roll No.-30")

# ---------------------------------------------------------
# GLOBAL DISPLAY SETTINGS (Prevents Column Truncation)
# ---------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.expand_frame_repr", False)

DATA_FILE = "ui_feature_sales_data.csv"

# ---------------------------------------------------------
# SYNTHETIC DATA GENERATION
# ---------------------------------------------------------

def generate_ui_sales_data(n=200):
    """
    Generates synthetic dataset representing:
    - Modern mobile UI feature scores (0–10)
    - Corresponding product sales
    """

    np.random.seed(42)

    df = pd.DataFrame({
        "AI_Assistant_Score": np.random.randint(0, 11, n),
        "Smart_Workflows_Score": np.random.randint(0, 11, n),
        "GlassMorphism_Score": np.random.randint(0, 11, n),
        "PanelUI_Score": np.random.randint(0, 11, n),
        "ColorGradient_Score": np.random.randint(0, 11, n),
        "Transparency_Score": np.random.randint(0, 11, n),
        "CapsuleUI_Score": np.random.randint(0, 11, n),
    })

    # Simulated sales influenced by some features
    df["Product_Sales"] = (
        df["AI_Assistant_Score"] * np.random.uniform(10, 15, n) +
        df["Smart_Workflows_Score"] * np.random.uniform(5, 10, n) +
        df["GlassMorphism_Score"] * np.random.uniform(2, 6, n) +
        df["ColorGradient_Score"] * np.random.uniform(1, 4, n) +
        np.random.normal(0, 20, n)
    ).round(2)

    df.to_csv(DATA_FILE, index=False)
    print(f"\nDataset successfully saved as '{DATA_FILE}'")

    return df


# ---------------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------------

def show_summary(df):
    print("\n=== Dataset Summary (Numeric) ===\n")
    summary = df.describe().round(2)
    print(summary.to_string())


# ---------------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------------

def show_correlation(df):

    corr = df.corr().round(2)

    print("\n=== Correlation Matrix ===\n")
    print(corr.to_string())

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar=True
    )

    plt.title("Correlation Between UI Feature Scores and Product Sales")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# LOAD DATA SAFELY
# ---------------------------------------------------------

def load_dataset():
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except FileNotFoundError:
        print("\nDataset not found. Please generate it first (Option 1).")
        return None


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def main_menu():

    df = None

    while True:
        print("\n====================================================")
        print("MOBILE UI FEATURE CORRELATION VISUALIZATION SYSTEM")
        print("====================================================")
        print("1. Generate UI Feature + Sales Dataset (Creates CSV)")
        print("2. Show Dataset Summary (All Columns)")
        print("3. Show Correlation Matrix & Heatmap (All Columns)")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            df = generate_ui_sales_data()

        elif choice == "2":
            if df is None:
                df = load_dataset()
                if df is None:
                    continue
            show_summary(df)

        elif choice == "3":
            if df is None:
                df = load_dataset()
                if df is None:
                    continue
            show_correlation(df)

        elif choice == "4":
            print("\nExiting program.")
            break

        else:
            print("Invalid choice. Please try again.")


# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()