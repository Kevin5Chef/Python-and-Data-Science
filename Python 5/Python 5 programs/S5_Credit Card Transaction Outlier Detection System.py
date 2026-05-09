import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("SY-5, Kevin Victor, Roll No.-30")

# ---------------------------------------------------------
# DATA GENERATION (Integer Realistic Transactions)
# ---------------------------------------------------------

def generate_transaction_data(n=1000, n_outliers=70):

    np.random.seed(42)

    # Normal transactions: $5 – $250
    normal = np.random.randint(5, 250, n - n_outliers)

    # High fraudulent-like outliers: $3000 – $10000
    high_outliers = np.random.randint(3000, 10000, n_outliers // 2)

    # Suspicious low test transactions: $0 – $4
    low_outliers = np.random.randint(0, 5, n_outliers // 2)

    all_data = np.concatenate([normal, high_outliers, low_outliers])
    np.random.shuffle(all_data)

    df = pd.DataFrame({
        "Transaction_Amount": all_data.astype(int)
    })

    return df


# ---------------------------------------------------------
# OUTLIER DETECTION (IQR METHOD)
# ---------------------------------------------------------

def detect_outliers(df):

    q1 = df["Transaction_Amount"].quantile(0.25)
    q3 = df["Transaction_Amount"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df["Transaction_Amount"] < lower_bound) |
        (df["Transaction_Amount"] > upper_bound)
    ]

    return outliers, lower_bound, upper_bound


# ---------------------------------------------------------
# BOX PLOT (Improved Version with Log Scale)
# ---------------------------------------------------------

def plot_box(df):

    plt.figure(figsize=(9, 6))
    plt.boxplot(df["Transaction_Amount"], vert=False)

    plt.xscale("log")  # ← makes distribution readable

    plt.title("Credit Card Transaction Amounts (Log Scale)")
    plt.xlabel("Transaction Amount ($)")
    plt.grid(True)

    plt.show()


# ---------------------------------------------------------
# REPORT
# ---------------------------------------------------------

def show_outlier_report(df):

    outliers, lb, ub = detect_outliers(df)

    print("\nOutlier Detection Report")
    print("---------------------------------")
    print(f"Lower Bound: {int(lb)}")
    print(f"Upper Bound: {int(ub)}")
    print(f"Total Transactions: {len(df)}")
    print(f"Detected Outliers: {len(outliers)}")

    print("\nFirst 15 Outliers:")
    print(outliers.head(15))


# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def main_menu():

    df = generate_transaction_data()

    while True:
        print("\n" + "="*60)
        print("Credit Card Transaction Outlier Detection System")
        print("="*60)
        print("1. Show Sample Transactions (First 20)")
        print("2. Show Outlier Detection Report")
        print("3. Show Box Plot (Log Scale)")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            print("\nSample Transactions:\n")
            print(df.head(20))

        elif choice == "2":
            show_outlier_report(df)

        elif choice == "3":
            plot_box(df)

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Try again.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()