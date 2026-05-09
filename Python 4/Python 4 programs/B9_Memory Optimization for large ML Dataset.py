import numpy as np
import pandas as pd
print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------

def generate_large_ml_dataset(n=1_000_000):
    np.random.seed(42)

    df = pd.DataFrame({
        "feature1": np.random.randn(n),            # float64
        "feature2": np.random.randn(n) * 100,      # float64
        "feature3": np.random.randint(0, 1000, n), # int64
        "label": np.random.choice(["cat", "dog", "mouse"], n) # object
    })

    return df

# ---------------------------------------------------------
# MEMORY DISPLAY FUNCTIONS
# ---------------------------------------------------------

def show_memory_usage(df, description):
    total_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"\n{description}: {total_mem:.2f} MB")
    return total_mem

def show_column_memory(df):
    print("\nColumn-wise Memory Usage (MB):")
    col_mem = df.memory_usage(deep=True) / (1024 ** 2)
    print(col_mem.round(3))

def show_dtypes(df):
    print("\nCurrent Data Types:")
    print(df.dtypes)

# ---------------------------------------------------------
# OPTIMIZATION FUNCTION
# ---------------------------------------------------------

def optimize_memory(df):
    df_opt = df.copy()

    print("\n--- Optimizing Data Types ---")

    # FLOAT64 → FLOAT32
    float_cols = df_opt.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        print(f"Converting {col}: float64 → float32")
        df_opt[col] = df_opt[col].astype("float32")

    # INT64 → INT32
    int_cols = df_opt.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        print(f"Converting {col}: int64 → int32")
        df_opt[col] = df_opt[col].astype("int32")

    # OBJECT → CATEGORY
    obj_cols = df_opt.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        print(f"Converting {col}: object → category")
        df_opt[col] = df_opt[col].astype("category")

    print("\nOptimization Complete.")
    return df_opt

# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def main_menu():
    df = generate_large_ml_dataset()

    while True:
        print("\n" + "-"*60)
        print("Memory Optimization for Large ML Dataset")
        print("-"*60)
        print("1. Show Initial Memory Details")
        print("2. Optimize Dataset (Show Conversion Details)")
        print("3. Compare Before vs After (Detailed)")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            show_dtypes(df)
            show_column_memory(df)
            show_memory_usage(df, "Total Memory Before Optimization")

        elif choice == "2":
            df_opt = optimize_memory(df)
            show_dtypes(df_opt)
            show_column_memory(df_opt)
            show_memory_usage(df_opt, "Total Memory After Optimization")

        elif choice == "3":
            print("\n--- BEFORE OPTIMIZATION ---")
            mem_before = show_memory_usage(df, "Total Memory")

            df_opt = optimize_memory(df)

            print("\n--- AFTER OPTIMIZATION ---")
            mem_after = show_memory_usage(df_opt, "Total Memory")

            reduction = ((mem_before - mem_after) / mem_before) * 100
            print(f"\nMemory Reduced By: {reduction:.2f}%")

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Try again.")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()