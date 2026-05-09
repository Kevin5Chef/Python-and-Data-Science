import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
print("SY-5, Kevin Victor, Roll No.-30")
print("\nIndustrial Acoustic Monitoring System")

DATA_FILE = "factory_sound_data.csv"
CORRUPTED_FILE = "factory_sound_corrupted.csv"

# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------

def generate_sound_data(n=1000):
    np.random.seed(42)

    loudness = np.random.normal(75, 5, n)
    pitch = np.random.normal(2000, 400, n)
    tone = np.random.normal(0.85, 0.05, n)
    amplitude = np.random.normal(0.5, 0.1, n)

    df = pd.DataFrame({
        "Loudness_dB": loudness,
        "Pitch_Hz": pitch,
        "Tone_Stability": tone,
        "Amplitude_Modulation": amplitude
    })

    df.to_csv(DATA_FILE, index=False)
    print(f"Dataset saved as {DATA_FILE}")

    return df


# ---------------------------------------------------------
# INJECT OUTLIERS
# ---------------------------------------------------------

def inject_outliers(df, rate=0.07):
    df = df.copy()
    n = len(df)
    num = int(n * rate)

    indices = np.random.choice(n, num, replace=False)

    df.loc[indices, "Loudness_dB"] = np.random.uniform(100, 120, num)
    df.loc[indices, "Pitch_Hz"] = np.random.uniform(8000, 12000, num)
    df.loc[indices, "Tone_Stability"] = np.random.uniform(0.2, 0.4, num)

    df.to_csv(CORRUPTED_FILE, index=False)
    print(f"Corrupted dataset saved as {CORRUPTED_FILE}")

    return df


# ---------------------------------------------------------
# OUTLIER DETECTION
# ---------------------------------------------------------

def detect_outliers(df, threshold=3):

    z_scores = np.abs(stats.zscore(df))

    # Convert back to DataFrame
    z_df = pd.DataFrame(z_scores, columns=df.columns)

    outlier_flags = z_df > threshold
    outlier_counts = outlier_flags.sum()

    return outlier_flags, outlier_counts


# ---------------------------------------------------------
# COUNT PLOT
# ---------------------------------------------------------

def plot_outlier_counts(outlier_flags):

    melted = outlier_flags.melt(
        var_name="Feature",
        value_name="Is_Outlier"
    )

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=melted,
        x="Feature",
        hue="Is_Outlier",
        palette="Set2"
    )

    plt.title("Count of Outliers per Acoustic Feature", fontsize=14)
    plt.xlabel("Acoustic Feature")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.legend(title="Outlier Status")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def menu():
    df = None
    df_corrupt = None
    outlier_flags = None

    while True:
        print("\n--------------------------------------------")
        print("FACTORY SOUND OUTLIER MONITORING SYSTEM")
        print("--------------------------------------------")
        print("1. Generate Sound Dataset (Creates CSV)")
        print("2. Inject Acoustic Outliers (Creates CSV)")
        print("3. Detect Outliers")
        print("4. Plot Outlier Count (Seaborn)")
        print("5. Exit")

        choice = input("Enter choice (1-5): ")

        if choice == "1":
            df = generate_sound_data()

        elif choice == "2":
            if df is None:
                if os.path.exists(DATA_FILE):
                    df = pd.read_csv(DATA_FILE)
                else:
                    print("Generate dataset first.")
                    continue
            df_corrupt = inject_outliers(df)

        elif choice == "3":
            if df_corrupt is None:
                if os.path.exists(CORRUPTED_FILE):
                    df_corrupt = pd.read_csv(CORRUPTED_FILE)
                else:
                    print("Inject outliers first.")
                    continue
            outlier_flags, counts = detect_outliers(df_corrupt)
            print("\nOutlier counts per feature:")
            print(counts)

        elif choice == "4":
            if outlier_flags is None:
                print("Run outlier detection first.")
            else:
                plot_outlier_counts(outlier_flags)

        elif choice == "5":
            print("Exiting system.")
            break

        else:
            print("Invalid choice.")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    menu()