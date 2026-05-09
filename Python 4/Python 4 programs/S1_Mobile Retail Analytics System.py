import numpy as np
import pandas as pd
import os
import time

print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------

def generate_mobile_sales_data():

    np.random.seed(42)

    categories = (["Flagship"] * 20 +
                  ["Budget"] * 30 +
                  ["Mid-Range"] * 50)

    categories = np.array(categories)

    # Price distributions
    flagship_prices = np.random.normal(100000, 10000, 20)
    budget_prices = np.random.normal(12000, 2000, 30)
    mid_prices = np.random.normal(35000, 5000, 50)

    prices = np.concatenate([flagship_prices,
                             budget_prices,
                             mid_prices])

    prices = np.clip(prices, 8000, 120000)

    # Age distributions
    flagship_age = np.random.normal(26, 4, 20)
    budget_age = np.random.normal(42, 6, 30)
    mid_age = np.random.normal(32, 5, 50)

    ages = np.concatenate([flagship_age,
                           budget_age,
                           mid_age])

    ages = np.clip(ages, 18, 60)

    # Feature score mapping
    feature_score = np.where(categories == "Flagship", 9,
                     np.where(categories == "Mid-Range", 7, 4))

    # Demand simulation
    demand_factor = 100000 / prices
    age_factor = (50 - np.abs(ages - 30)) / 50

    units_sold = np.round((demand_factor * age_factor) * 10)
    units_sold = np.clip(units_sold, 1, 15).astype(int)

    df = pd.DataFrame({
        "Category": categories,
        "Price": prices.round(2),
        "Customer_Age": ages.round(1),
        "Feature_Score": feature_score,
        "Units_Sold": units_sold
    })

    df["Revenue"] = (df["Price"] * df["Units_Sold"]).round(2)

    return df


# ---------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------

def show_full_dataset(df):
    print("\nFull Dataset (100 Records)\n")
    print(df.to_string(index=False))


def numerical_analysis(df):
    print("\nNumerical Summary\n")
    print(df.describe().round(2))

    print("\nCorrelation Matrix (Numeric Columns Only)\n")
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.corr().round(2))


def categorical_analysis(df):
    print("\nCategory-wise Performance\n")

    grouped = df.groupby("Category").agg({
        "Price": "mean",
        "Customer_Age": "mean",
        "Units_Sold": "mean",
        "Revenue": "sum"
    }).round(2)

    print(grouped)

    print("\nTotal Units Sold by Category\n")
    print(df.groupby("Category")["Units_Sold"].sum())


def generate_insights(df):

    print("\nStrategic Insights\n")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    avg_price = df["Price"].mean()
    avg_age = df["Customer_Age"].mean()

    insights = [
        f"Average selling price is {avg_price:.0f}, indicating mid-range dominance in sales volume.",
        f"Average customer age is {avg_age:.1f}, showing strong participation from internet-active demographics.",
        f"Price vs Units_Sold correlation is {corr.loc['Price','Units_Sold']:.2f}, confirming affordability drives demand.",
        f"Feature_Score vs Revenue correlation is {corr.loc['Feature_Score','Revenue']:.2f}, supporting continued R&D in premium innovation.",
        "Mid-range phones provide stable and scalable revenue streams.",
        "Flagship phones generate high revenue per unit and support brand positioning.",
        "Budget phones appeal to older demographics and support volume-based market penetration."
    ]

    for statement in insights:
        print(statement)
        time.sleep(1)


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def main_menu():

    df = generate_mobile_sales_data()

    while True:
        clear()
        print("Mobile Retail Analytics System")
        print("--------------------------------")
        print("1. Show Full Dataset (100 Records)")
        print("2. Numerical Analysis")
        print("3. Category Analysis")
        print("4. Strategic Insights")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            show_full_dataset(df)
            input("\nPress Enter to continue...")

        elif choice == "2":
            numerical_analysis(df)
            input("\nPress Enter to continue...")

        elif choice == "3":
            categorical_analysis(df)
            input("\nPress Enter to continue...")

        elif choice == "4":
            generate_insights(df)
            input("\nPress Enter to continue...")

        elif choice == "5":
            print("Exiting system.")
            break

        else:
            print("Invalid choice.")
            time.sleep(1)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()