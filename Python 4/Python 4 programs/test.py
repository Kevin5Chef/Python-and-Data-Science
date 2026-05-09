import pandas as pd
import numpy as np
import os
print("SY-5, Kevin Victor, Roll No.-30")
CSV_FILENAME = "library_data.csv"


# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------

def generate_library_csv(num=100):

    np.random.seed(42)

    total_books = np.random.randint(1, 15, num)

    returned_on_time = np.random.randint(0, 15, num)
    returned_on_time = np.minimum(returned_on_time, total_books)

    returned_late = np.random.randint(0, 10, num)
    returned_late = np.minimum(returned_late, total_books - returned_on_time)

    data = {
        "Member_ID": np.arange(1, num + 1, dtype=int),
        "Is_Premium_Member": np.random.choice([0, 1], num, p=[0.7, 0.3]).astype(int),
        "Total_Books_Checked_Out": total_books.astype(int),
        "Books_Returned_On_Time": returned_on_time.astype(int),
        "Books_Returned_Late": returned_late.astype(int),
        "Defaulter": np.random.choice([0, 1], num, p=[0.85, 0.15]).astype(int),
        "Fine_Generated": np.random.randint(0, 201, num).astype(int),
        "Dept_CS": np.random.randint(0, 50, num).astype(int),
        "Dept_Mech": np.random.randint(0, 40, num).astype(int),
        "Dept_Electrical": np.random.randint(0, 35, num).astype(int),
        "Dept_ECE": np.random.randint(0, 30, num).astype(int),
        "Dept_Design": np.random.randint(0, 25, num).astype(int),
        "Dept_Finance": np.random.randint(0, 20, num).astype(int),
        "Dept_Humanities": np.random.randint(0, 45, num).astype(int)
    }

    df = pd.DataFrame(data)
    df.to_csv(CSV_FILENAME, index=False)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_library_data():

    if not os.path.exists(CSV_FILENAME):
        generate_library_csv()

    df = pd.read_csv(CSV_FILENAME)

    # Ensure integer types
    df = df.astype(int)

    return df


# ---------------------------------------------------------
# DISPLAY FUNCTIONS
# ---------------------------------------------------------

def show_full_dataset(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)

    print("\nFull Dataset:\n")
    print(df.to_string(index=False))


def calculate_basic_statistics(df):

    print("\nBasic Statistics\n")

    numeric_cols = df.select_dtypes(include=[np.number])

    mean_vals = numeric_cols.mean().round(2)
    median_vals = numeric_cols.median().astype(int)
    mode_vals = numeric_cols.mode().iloc[0].astype(int)

    print("Mean:\n", mean_vals)
    print("\nMedian:\n", median_vals)
    print("\nMode:\n", mode_vals)


def show_summary(df):
    print("\nStatistical Summary\n")
    print(df.describe().round(2))


# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def main_menu():

    df = load_library_data()

    while True:
        print("\n" + "-"*50)
        print("Library Management System Dataset Analysis")
        print("-"*50)
        print("1. Show Full Dataset")
        print("2. Basic Statistics (Mean, Median, Mode)")
        print("3. Summary Description")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            show_full_dataset(df)

        elif choice == "2":
            calculate_basic_statistics(df)

        elif choice == "3":
            show_summary(df)

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()