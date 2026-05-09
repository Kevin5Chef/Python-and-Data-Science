import pandas as pd
import numpy as np
import os


print("SY-5, Kevin Victor, Roll No.-30")

CSV_FILENAME = "smart_grid_logs.csv"

# ---------------------------------------------------------
# GENERATE LARGE GRID LOG DATASET
# ---------------------------------------------------------

def generate_smart_grid_logs(num=1_000_000):
    np.random.seed(42)

    timestamps = pd.date_range(start="2025-01-01", periods=num, freq="S")
    grid_zones = np.random.choice(["North", "South", "East", "West"], num)

    voltage = np.random.normal(230, 10, num).round(2)
    current = np.random.uniform(0, 100, num).round(2)
    power = (voltage * current).round(2)
    load = np.random.uniform(10, 100, num).round(2)

    event_type = np.random.choice(
        ["Normal", "Dip", "Spike"],
        num, p=[0.95, 0.03, 0.02]
    )

    frequency = np.random.normal(50, 0.3, num).round(3)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "grid_zone": grid_zones,
        "voltage": voltage,
        "current": current,
        "power": power,
        "load": load,
        "event_type": event_type,
        "frequency": frequency
    })

    df.to_csv(CSV_FILENAME, index=False)

# ---------------------------------------------------------
# MEMORY-EFFICIENT CHUNK SUMMARY
# ---------------------------------------------------------

def chunk_summary(file_path, chunksize=200_000):
    total_rows = 0
    zone_power = {}
    dips = 0
    spikes = 0
    chunk_counter = 0

    print("\nProcessing file in chunks...")
    print("Chunk Size:", chunksize)
    print("-" * 50)

    for chunk in pd.read_csv(file_path, chunksize=chunksize, parse_dates=["timestamp"]):
        chunk_counter += 1
        total_rows += len(chunk)

        mem_usage = chunk.memory_usage(deep=True).sum() / (1024 * 1024)

        print(f"\nChunk {chunk_counter}")
        print(f"Rows in chunk: {len(chunk)}")
        print(f"Memory used by this chunk: {mem_usage:.2f} MB")

        # Count dips / spikes
        dips += (chunk["event_type"] == "Dip").sum()
        spikes += (chunk["event_type"] == "Spike").sum()

        # Incremental aggregation
        zone_grp = chunk.groupby("grid_zone")["power"].sum()

        for z, p in zone_grp.items():
            zone_power[z] = zone_power.get(z, 0) + p

    print("\nAll chunks processed successfully without loading full dataset into memory.")

    return total_rows, zone_power, dips, spikes

# ---------------------------------------------------------
# VOLTAGE STATS WITH INTERNAL WORKING
# ---------------------------------------------------------

def chunk_voltage_stats(file_path, chunksize=200_000):
    total_sum = 0
    total_count = 0
    global_min = float("inf")
    global_max = float("-inf")
    chunk_counter = 0

    print("\nComputing Voltage Statistics Incrementally...")
    print("Chunk Size:", chunksize)
    print("-" * 50)

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunk_counter += 1
        chunk_v = chunk["voltage"].astype(float)

        mem_usage = chunk.memory_usage(deep=True).sum() / (1024 * 1024)

        print(f"\nChunk {chunk_counter}")
        print(f"Rows processed: {len(chunk_v)}")
        print(f"Chunk memory usage: {mem_usage:.2f} MB")

        # Incremental mean calculation
        total_sum += chunk_v.sum()
        total_count += len(chunk_v)

        # Min/Max update
        global_min = min(global_min, chunk_v.min())
        global_max = max(global_max, chunk_v.max())

        print(f"Running Mean Voltage: {(total_sum/total_count):.2f} V")

    mean_voltage = total_sum / total_count

    print("\nFinal aggregation completed safely using incremental updates.")

    return mean_voltage, global_min, global_max

# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------

def main_menu():
    if not os.path.exists(CSV_FILENAME):
        print("Generating large smart grid dataset...")
        generate_smart_grid_logs()

    while True:
        print("\n" + "-"*62)
        print("Smart Electrical Grid — Memory Efficient Processing")
        print("-"*62)
        print("1. Show Sample Data")
        print("2. Show Voltage Statistics (with internal working)")
        print("3. Show Energy per Grid Zone (in MW)")
        print("4. Show Dip/Spike Counts")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            sample = pd.read_csv(CSV_FILENAME, nrows=10)
            print(sample.to_string(index=False))

        elif choice == "2":
            mean_v, v_min, v_max = chunk_voltage_stats(CSV_FILENAME)
            print("\nVoltage Statistics (Final Results):")
            print(f"Mean Voltage: {mean_v:.2f} V")
            print(f"Min Voltage : {v_min:.2f} V")
            print(f"Max Voltage : {v_max:.2f} V")

        elif choice == "3":
            total_rows, zone_power, dips, spikes = chunk_summary(CSV_FILENAME)

            print("\n=== Total Rows Processed:", total_rows)
            print("\n=== Total Power Generation by Grid Zone (in MW):")

            for zone, pwr in zone_power.items():
                power_mw = pwr / 1_000_000
                print(f"{zone} : {power_mw:.2f} MW")

        elif choice == "4":
            total_rows, zone_power, dips, spikes = chunk_summary(CSV_FILENAME)
            print("\n=== Dip Events:", dips)
            print("=== Spike Events:", spikes)

        elif choice == "5":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Try again.")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()