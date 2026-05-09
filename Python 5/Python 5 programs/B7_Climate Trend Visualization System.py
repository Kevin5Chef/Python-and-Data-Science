import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

print("SY-5, Kevin Victor, Roll No.-30")

# ---------------------------------------------------------
# GENERATE LONG-TERM CLIMATE DATA (1880–2026)
# ---------------------------------------------------------

def generate_climate_data():
    years = np.arange(1880, 2027)

    # Global temperature anomaly
    temp = (
        0.02 * (years - 1880)
        + 0.015 * np.clip(years - 1950, 0, None)
        + 0.03 * np.clip(years - 1980, 0, None)
    )

    # CO2 concentration (ppm)
    co2 = 280 + 0.8 * (years - 1880)

    # Sea Surface Temperature (°C)
    sst = 14 + 0.018 * (years - 1880) + 0.01 * np.clip(years - 1950, 0, None)

    # Ozone hole area (million km²)
    ozone = np.zeros_like(years, dtype=float)
    mask = years >= 1979
    ozone[mask] = (
        10 + 0.5 * (years[mask] - 1979)
        - 0.3 * np.clip(years[mask] - 2000, 0, None)
    )
    ozone = np.clip(ozone, 0, None)

    return years, temp, co2, sst, ozone


# ---------------------------------------------------------
# FEATURE SCALING (Min-Max)
# ---------------------------------------------------------

def scale_features(*arrays):
    scaler = MinMaxScaler()
    stacked = np.column_stack(arrays)
    scaled = scaler.fit_transform(stacked)
    return scaled.T


# ---------------------------------------------------------
# GRADIENT LINE FUNCTION
# ---------------------------------------------------------

def plot_gradient_line(x, y, cmap, label):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(x.min(), x.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x)
    lc.set_linewidth(3)

    plt.gca().add_collection(lc)
    plt.plot([], [], color=cmap(0.8), linewidth=3, label=label)


# ---------------------------------------------------------
# PROFESSIONAL PLOT
# ---------------------------------------------------------

def plot_climate_trends():

    years, temp, co2, sst, ozone = generate_climate_data()

    # Feature scaling
    temp_s, co2_s, sst_s, ozone_s = scale_features(temp, co2, sst, ozone)

    plt.figure(figsize=(14, 8))

    # Custom Gradients
    temp_cmap = LinearSegmentedColormap.from_list(
        "temp_grad", ["skyblue", "orange", "red"]
    )

    co2_cmap = LinearSegmentedColormap.from_list(
        "co2_grad", ["lightgrey", "grey", "black"]
    )

    sst_cmap = LinearSegmentedColormap.from_list(
        "sst_grad", ["lightblue", "blue", "navy"]
    )

    ozone_cmap = LinearSegmentedColormap.from_list(
        "ozone_grad", ["#FFF44F", "orange"]
    )

    # Plot gradient lines
    plot_gradient_line(years, temp_s, temp_cmap, "Global Temp Anomaly (Scaled)")
    plot_gradient_line(years, co2_s, co2_cmap, "CO2 Concentration (Scaled)")
    plot_gradient_line(years, sst_s, sst_cmap, "Sea Surface Temp (Scaled)")
    plot_gradient_line(years, ozone_s, ozone_cmap, "Ozone Hole Size (Scaled)")

    plt.title("Long-Term Climate Trends (1880–2026)\nFeature-Scaled Visualization",
              fontsize=16, fontweight="bold")

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Scaled Value (0–1)", fontsize=12)

    plt.legend(loc="upper left", fontsize=11)
    plt.grid(alpha=0.3)

    plt.xlim(years.min(), years.max())
    plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def menu():
    while True:
        print("\n------------------------------------------------")
        print("CLIMATE TREND PROFESSIONAL VISUALIZATION SYSTEM")
        print("------------------------------------------------")
        print("1. Plot Scaled Multi-Gradient Climate Graph")
        print("2. Exit")

        choice = input("Enter your choice (1-2): ")

        if choice == "1":
            plot_climate_trends()
        elif choice == "2":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Try again.")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    menu()