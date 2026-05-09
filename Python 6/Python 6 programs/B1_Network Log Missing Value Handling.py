import pandas as pd
import numpy as np
import os

print("SY-5, Kevin Victor, Roll No.-30")

# ---------------------------------------------------------
# FILE NAME
# ---------------------------------------------------------

CSV_FILENAME = "network_logs.csv"

# ---------------------------------------------------------
# DATA GENERATION – NETWORK PACKET ROUTING LOGS
# ---------------------------------------------------------

def generate_network_logs(num=300):
    """
    Generates synthetic ISP packet routing logs
    with intentional missing values.
    """

    np.random.seed(42)

    timestamps = pd.date_range(
        start="2026-01-01 00:00",
        periods=num,
        freq="min"
    )

    src_ips = np.random.choice(
        ["192.168.1.1", "192.168.1.2", "10.0.0.1", "10.0.0.2"],
        num
    )

    dst_ips = np.random.choice(
        ["172.16.0.1", "172.16.0.2", "192.168.1.10"],
        num
    )

    packet_size = np.random.randint(100, 1500, num)  # bytes
    latency_ms = np.random.normal(50, 15, num).round(2)  # milliseconds

    status_codes = np.random.choice(
        ["OK", "TIMEOUT", "DROP", "RETRY"],
        num,
        p=[0.8, 0.05, 0.1, 0.05]
    )

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Src_IP": src_ips,
        "Dst_IP": dst_ips,
        "Packet_Size": packet_size,
        "Latency_ms": latency_ms,
        "Status": status_codes
    })

    # Introduce missing values (8% per selected column)
    for col in ["Packet_Size", "Latency_ms", "Status"]:
        missing_indices = df.sample(frac=0.08).index
        df.loc[missing_indices, col] = np.nan

    df.to_csv(CSV_FILENAME, index=False)
    return df


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_logs():
    if not os.path.exists(CSV_FILENAME):
        generate_network_logs()
    return pd.read_csv(CSV_FILENAME)


# ---------------------------------------------------------
# IMPUTATION METHODS (FIXED – NO CHAINED ASSIGNMENT)
# ---------------------------------------------------------

def impute_mean(df):
    df_imputed = df.copy()

    df_imputed["Packet_Size"] = df_imputed["Packet_Size"].fillna(
        round(df_imputed["Packet_Size"].mean())
    )

    df_imputed["Latency_ms"] = df_imputed["Latency_ms"].fillna(
        round(df_imputed["Latency_ms"].mean(), 2)
    )

    return df_imputed


def impute_median(df):
    df_imputed = df.copy()

    df_imputed["Packet_Size"] = df_imputed["Packet_Size"].fillna(
        df_imputed["Packet_Size"].median()
    )

    df_imputed["Latency_ms"] = df_imputed["Latency_ms"].fillna(
        df_imputed["Latency_ms"].median()
    )

    return df_imputed


def impute_mode(df):
    df_imputed = df.copy()

    df_imputed["Status"] = df_imputed["Status"].fillna(
        df_imputed["Status"].mode()[0]
    )

    return df_imputed


# ---------------------------------------------------------
# DISPLAY FUNCTIONS
# ---------------------------------------------------------

def show_dataset(df):
    print("\n===== SAMPLE NETWORK PACKET LOGS =====\n")
    print(df.head(15).to_string(index=False))


def show_missing(df):
    print("\n===== MISSING VALUES SUMMARY =====\n")
    print(df.isna().sum())


# ---------------------------------------------------------
# MENU SYSTEM
# ---------------------------------------------------------

def main_menu():
    df = load_logs()

    while True:
        print("\n" + "-" * 60)
        print("Network Log Missing Value Handling")
        print("-" * 60)
        print("1. View Sample Dataset")
        print("2. Show Missing Value Summary")
        print("3. Impute Missing with MEAN (Numeric Columns)")
        print("4. Impute Missing with MEDIAN (Numeric Columns)")
        print("5. Impute Missing with MODE (Categorical Column)")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ")

        if choice == "1":
            show_dataset(df)

        elif choice == "2":
            show_missing(df)

        elif choice == "3":
            df_mean = impute_mean(df)
            print("\n===== AFTER MEAN IMPUTATION =====")
            show_missing(df_mean)

        elif choice == "4":
            df_median = impute_median(df)
            print("\n===== AFTER MEDIAN IMPUTATION =====")
            show_missing(df_median)

        elif choice == "5":
            df_mode = impute_mode(df)
            print("\n===== AFTER MODE IMPUTATION =====")
            show_missing(df_mode)

        elif choice == "6":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()