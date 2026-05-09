import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# FORCE PANDAS TO DISPLAY ALL COLUMNS
# ---------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)

# ---------------------------------------------------------
# DATA GENERATION FOR MAGNETIC PHENOMENON
# ---------------------------------------------------------

def generate_magnetic_data(n=200):
    np.random.seed(42)

    H = np.random.uniform(-1000, 1000, n)
    B = np.tanh(H / 500) + np.random.normal(0, 0.05, n)
    Ms = np.random.normal(800000, 10000, n)
    Hc = np.abs(np.random.normal(100, 20, n))

    df = pd.DataFrame({
        "Applied_Field_H (A/m)": H.round(2),
        "Flux_Density_B (T)": B.round(3),
        "Saturation_Ms (A/m)": Ms.round(1),
        "Coercivity_Hc (A/m)": Hc.round(2)
    })

    return df

# ---------------------------------------------------------
# UNIVARIATE ANALYSIS FUNCTIONS
# ---------------------------------------------------------

def show_summary(df):
    print("\nFull Summary (Numeric Describe):\n")
    print(df.describe().round(3).to_string())

def show_mean(df):
    print("\nMean of Each Feature:\n")
    print(df.mean(numeric_only=True).round(3).to_string())

def show_median(df):
    print("\nMedian of Each Feature:\n")
    print(df.median(numeric_only=True).round(3).to_string())

def show_mode(df):
    print("\nMode of Each Feature:\n")
    modes = df.mode().round(3)
    print(modes.to_string(index=False))

def show_std(df):
    print("\nStandard Deviation of Each Feature:\n")
    print(df.std(numeric_only=True).round(3).to_string())

def plot_histograms(df):
    df.hist(bins=15, figsize=(12, 8))
    plt.suptitle("Histogram of Magnetic Measurements")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def main_menu():
    df = generate_magnetic_data()

    while True:
        print("\n" + "-"*60)
        print("MAGNETIC PHENOMENON - UNIVARIATE ANALYSIS SYSTEM")
        print("-"*60)
        print("1. Show First 10 Records")
        print("2. Show Summary Statistics")
        print("3. Show Mean")
        print("4. Show Median")
        print("5. Show Mode")
        print("6. Show Standard Deviation")
        print("7. Plot Histograms")
        print("8. Exit")

        choice = input("\nEnter your choice (1-8): ")

        if choice == "1":
            print("\nFirst 10 Records:\n")
            print(df.head(10).to_string(index=False))

        elif choice == "2":
            show_summary(df)

        elif choice == "3":
            show_mean(df)

        elif choice == "4":
            show_median(df)

        elif choice == "5":
            show_mode(df)

        elif choice == "6":
            show_std(df)

        elif choice == "7":
            plot_histograms(df)

        elif choice == "8":
            print("Exiting Univariate Analysis System.")
            break

        else:
            print("Invalid choice. Please select between 1 and 8.")

# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main_menu()