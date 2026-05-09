import pandas as pd
import numpy as np
import os

print("SY-5, Kevin Victor, Roll No.-30")

CSV_FILENAME = "robot_learning_logs.csv"

# ---------------------------------------------------------
# GENERATE LARGE DATASET (SIMULATED)
# ---------------------------------------------------------
def generate_robot_logs(num=1_000_000):
    """
    Creates a large simulated dataset of robot sensor logs
    with multiple sensor types.
    """

    np.random.seed(42)

    timestamps = pd.date_range(
        start="2026-01-01 00:00",
        periods=num,
        freq="S"
    )

    sensor_types = np.random.choice(
        ["voice", "camera", "haptic"],
        num,
        p=[0.2, 0.5, 0.3]
    )

    # FEATURE VECTOR as three floats
    feature_vectors = [
        [round(np.random.random(), 3),
         round(np.random.random(), 3),
         round(np.random.random(), 3)]
        for _ in range(num)
    ]

    detected_cmds = np.random.choice(
        ["move", "grip", "release", "stop", ""], num,
        p=[0.15, 0.15, 0.15, 0.10, 0.45]
    )

    haptic_force = np.where(
        sensor_types == "haptic",
        np.random.uniform(0.0, 100.0, num).round(2),
        np.nan
    )

    objects_recognized = np.where(
        sensor_types == "camera",
        np.random.choice(
            ["box", "sphere", "cylinder", "none"],
            num, p=[0.25, 0.25, 0.25, 0.25]
        ),
        ""
    )

    pos_x = np.random.uniform(-5.0, 5.0, num).round(3)
    pos_y = np.random.uniform(-5.0, 5.0, num).round(3)
    pos_z = np.random.uniform(0.0, 2.0, num).round(3)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_type": sensor_types,
        "feature_vector": feature_vectors,
        "detected_command": detected_cmds,
        "haptic_force": haptic_force,
        "object_detected": objects_recognized,
        "position_x": pos_x,
        "position_y": pos_y,
        "position_z": pos_z
    })

    df.to_csv(CSV_FILENAME, index=False)
    return df

# ---------------------------------------------------------
# CHUNK PROCESSING FUNCTIONS
# ---------------------------------------------------------
def show_chunk_summary(chunk):
    """
    Shows statistics for a chunk.
    """
    print("\nChunk Timestamp Range:", chunk['timestamp'].min(),
          "to", chunk['timestamp'].max())

    print("Sensor Type Counts:\n", chunk['sensor_type'].value_counts())

    mean_pos_x = chunk['position_x'].mean()
    mean_pos_y = chunk['position_y'].mean()
    mean_pos_z = chunk['position_z'].mean()

    print("\nMean Position (x,y,z):",
          round(mean_pos_x, 3), round(mean_pos_y, 3), round(mean_pos_z, 3))


def process_in_chunks(file_path, chunksize=200_000):
    """
    Reads and processes file in chunks.
    """
    print(f"\nProcessing in chunks of {chunksize} rows...\n")
    chunk_num = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunk_num += 1
        print(f"\n----- CHUNK {chunk_num} -----")
        show_chunk_summary(chunk)


# ---------------------------------------------------------
# MENU
# ---------------------------------------------------------
def main_menu():
    if not os.path.exists(CSV_FILENAME):
        print("Generating large dataset (this may take a moment)...")
        generate_robot_logs()

    while True:
        print("\n" + "-"*70)
        print("Industrial Robot Log Chunk Processing System")
        print("-"*70)
        print("1. Show Initial Sample (first 15 rows)")
        print("2. Show Missing Data Summary")
        print("3. Process Dataset in Chunks")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == "1":
            df = pd.read_csv(CSV_FILENAME, nrows=15)
            print("\n===== SAMPLE DATA =====\n")
            print(df.to_string(index=False))

        elif choice == "2":
            df = pd.read_csv(CSV_FILENAME)
            print("\n===== MISSING SUMMARY =====\n")
            print(df.isna().sum())

        elif choice == "3":
            process_in_chunks(CSV_FILENAME)

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