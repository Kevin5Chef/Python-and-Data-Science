import pandas as pd
import numpy as np
print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# DATA GENERATION – ELECTRONICS STORE SALES
# ---------------------------------------------------------

def generate_sales_data(n=500):
    np.random.seed(42)

    categories = [
        "Mobile Phones",
        "Laptops",
        "Desktop PCs",
        "Headphones",
        "Earphones",
        "Smart TVs",
        "Refrigerators",
        "Air Conditioners",
        "Air Coolers",
        "Ice Boxes",
        "Electric Kitchen Appliances"
    ]

    # Realistic weighted probabilities (Mobiles sell most)
    probabilities = [0.22, 0.12, 0.08, 0.10, 0.10,
                     0.09, 0.07, 0.06, 0.05, 0.04, 0.07]

    sales = np.random.choice(categories, size=n, p=probabilities)

    df = pd.DataFrame({
        "Transaction_ID": range(1, n + 1),
        "Product_Category": sales
    })

    return df


# ---------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------

def show_sales_table(df):
    print("\n===== ELECTRONICS STORE SALES TABLE =====\n")
    print(df.to_string(index=False))


def show_category_counts(df):
    print("\n===== CATEGORY SALES FREQUENCY =====\n")
    counts = df["Product_Category"].value_counts()
    print(counts.to_string())


def find_mode_category(df):
    mode_category = df["Product_Category"].mode()
    print("\n===== MOST FREQUENTLY SOLD PRODUCT CATEGORY =====\n")
    print(mode_category.to_string(index=False))


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def menu():
    df = generate_sales_data()

    while True:
        print("\n" + "-"*60)
        print("ELECTRONICS STORE SALES ANALYSIS SYSTEM")
        print("-"*60)
        print("1. Display Full Sales Table")
        print("2. Show Category Frequency Table")
        print("3. Find Most Frequently Sold Category (Mode)")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            show_sales_table(df)

        elif choice == "2":
            show_category_counts(df)

        elif choice == "3":
            show_category_counts(df)
            find_mode_category(df)

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please select between 1 and 4.")


# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    menu()