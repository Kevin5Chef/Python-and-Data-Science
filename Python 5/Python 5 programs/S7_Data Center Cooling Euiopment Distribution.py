import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("SY-5, Kevin Victor, Roll No.-30")

# ---------------------------------------------------------
# GENERATE SAMPLE CUSTOMER DISTRIBUTION DATA
# ---------------------------------------------------------

def generate_customer_data():
    np.random.seed(42)

    countries = [
        "China", "India", "Japan",
        "South Korea", "Australia",
        "Singapore", "Malaysia", "Indonesia"
    ]

    # Simulated number of customers (cooling equipment sales)
    # Based on relative data center growth in region
    sales_counts = [
        420, 310, 210, 150, 120, 90, 80, 60
    ]

    df = pd.DataFrame({
        "Country": countries,
        "Customers": sales_counts
    })

    df.to_csv("customer_distribution_apac.csv", index=False)
    return df

# ---------------------------------------------------------
# PLOT DISTRIBUTIONS
# ---------------------------------------------------------

def plot_bar_distribution(df):
    plt.figure(figsize=(10, 6))

    sns.barplot(
        x="Country",
        y="Customers",
        hue="Country",          # FIX: assign hue
        data=df,
        palette="viridis",
        legend=False            # prevent duplicate legend
    )

    plt.title("Customer Distribution Across Asia-Pacific Countries")
    plt.xlabel("Country")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pie_distribution(df):
    plt.figure(figsize=(8, 8))
    plt.pie(
        df["Customers"],
        labels=df["Country"],
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("viridis", len(df))
    )
    plt.title("Customer Distribution (Percentage) Across APAC")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def main_menu():
    df = generate_customer_data()

    while True:
        print("\n-------------------------------------------")
        print("DATA-CENTER COOLING EQUIPMENT DISTRIBUTION")
        print("-------------------------------------------")
        print("1. Show Customer Distribution Table")
        print("2. Plot Bar Chart of Customer Distribution")
        print("3. Plot Pie Chart of Customer Distribution")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            print("\n=== Customer Distribution Table ===\n")
            print(df.to_string(index=False))

        elif choice == "2":
            plot_bar_distribution(df)

        elif choice == "3":
            plot_pie_distribution(df)

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()