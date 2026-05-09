import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("SY-5, Kevin Victor, Roll No.-30")

# ---------------------------------------------------------
# DATA GENERATION – MUSICAL STORE ITEMS
# ---------------------------------------------------------

def generate_instrument_data():
    instruments = [
        "acoustic guitar", "electric guitar", "keyboard",
        "grand piano", "electric piano", "amplifiers",
        "acoustic drums", "electric drums", "bass guitar",
        "processors", "mixers", "synthesizers",
        "digital organ", "violin", "cello"
    ]

    # Generate a DataFrame of 100 random sales
    np.random.seed(42)
    sampled = np.random.choice(instruments, 100)

    df = pd.DataFrame({
        "Instrument": sampled
    })
    return df

# ---------------------------------------------------------
# LABEL ENCODING
# ---------------------------------------------------------

def apply_label_encoding(df):
    encoder = LabelEncoder()
    df_encoded = df.copy()
    df_encoded["Label_Code"] = encoder.fit_transform(df_encoded["Instrument"])
    return df_encoded, encoder

# ---------------------------------------------------------
# DISPLAY FUNCTIONS
# ---------------------------------------------------------

def show_dataset(df):
    print("\n===== MUSICAL STORE INVENTORY =====\n")
    print(df.head(15).to_string(index=False))


def show_encoded(df):
    print("\n===== LABEL ENCODED DATA =====\n")
    print(df.head(15).to_string(index=False))


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def main_menu():
    df = generate_instrument_data()

    while True:
        print("\n" + "-"*60)
        print("Musical Store Label Encoding System")
        print("-"*60)
        print("1. Show Original Dataset")
        print("2. Apply Label Encoding")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == "1":
            show_dataset(df)

        elif choice == "2":
            df_encoded, encoder = apply_label_encoding(df)
            show_encoded(df_encoded)

            print("\nEncoded Mapping (Instrument → Code):")
            mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            for inst, code in mapping.items():
                print(f"{inst} → {code}")

        elif choice == "3":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Try again.")

# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main_menu()