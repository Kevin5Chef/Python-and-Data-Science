import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# DATA GENERATION (Realistic Warming Trend Simulation)
# ---------------------------------------------------------

def generate_pacific_temperature_data():
    """
    Generate realistic synthetic Pacific Ocean average SST data
    from 1880 to 2026 based on historical warming trends.
    """

    years = np.arange(1880, 2027)

    baseline_temp = 15.0  # Approximate late-1800s baseline SST (°C)

    # Slow warming early period
    temps = baseline_temp + 0.008 * (years - 1880)

    # Accelerated warming after 1970
    temps += 0.015 * np.where(years >= 1970, years - 1970, 0)

    # Add realistic annual variability
    np.random.seed(42)
    temps += np.random.normal(0, 0.12, len(years))

    df = pd.DataFrame({
        "Year": years,
        "Avg_Pacific_SST": temps.round(2)
    })

    return df


# ---------------------------------------------------------
# GRADIENT LINE PLOT FUNCTION
# ---------------------------------------------------------

def plot_temperature_gradient(df):

    x = df["Year"].values
    y = df["Avg_Pacific_SST"].values

    # Custom gradient: Sky Blue → Orange → Red
    cmap = LinearSegmentedColormap.from_list(
        "ocean_warming_gradient",
        ["skyblue", "orange", "red"]
    )

    # Create segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(x.min(), x.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x)
    lc.set_linewidth(2.5)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.add_collection(lc)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.3, y.max() + 0.3)

    ax.set_title("Pacific Ocean Average Surface Temperature (1880–2026)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg SST (°C)")
    ax.grid(True)

    # Add color bar legend
    cbar = plt.colorbar(lc)
    cbar.set_label("Year Progression")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# DATA DISPLAY FUNCTION
# ---------------------------------------------------------

def show_data(df):
    print("\nFirst 20 Rows of Pacific SST Dataset:\n")
    pd.set_option('display.max_columns', None)
    print(df.head(20).to_string(index=False))


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def main_menu():

    df = generate_pacific_temperature_data()

    while True:
        print("\n" + "="*55)
        print("Pacific Ocean Temperature Variation Analyzer")
        print("="*55)
        print("1. Show Temperature Data (First 20 Years)")
        print("2. Plot Gradient Line Graph (1880–2026)")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == "1":
            show_data(df)

        elif choice == "2":
            plot_temperature_gradient(df)

        elif choice == "3":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()