import pandas as pd
import matplotlib.pyplot as plt
print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------

def generate_meme_data():
    """
    Generates a sample dataframe with:
    - Meme name
    - Year started
    - Avg engaged user age
    - Popularity score (0-100)
    """
    data = [
        ["Dancing Baby", 1996, 42, 45],
        ["All Your Base", 2000, 38, 55],
        ["Badger Badger", 2003, 32, 60],
        ["Rickroll", 2007, 27, 85],
        ["Nyan Cat", 2011, 24, 75],
        ["Doge", 2013, 22, 80],
        ["Distracted BF", 2017, 26, 70],
        ["Woman Yelling @ Cat", 2019, 30, 78],
        ["AI Meme Boom", 2023, 28, 90],
        ["Future Meme 2026", 2026, 25, 88]
    ]

    df = pd.DataFrame(data, columns=[
        "Meme", "Year", "Avg_User_Age", "Popularity_Score"
    ])
    return df


# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------

def plot_scatter(df):

    x = df["Avg_User_Age"]
    y = df["Popularity_Score"]
    labels = df["Meme"]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue")

    # Annotate each point
    for i, txt in enumerate(labels):
        plt.annotate(txt, (x[i] + 0.3, y[i] + 0.3), fontsize=8)

    plt.title("Internet Meme Popularity vs. User Age")
    plt.xlabel("Average Engaged User Age")
    plt.ylabel("Popularity Score (0–100)")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------
# CLI MENU
# ---------------------------------------------------------

def main_menu():

    df = generate_meme_data()

    while True:
        print("\n" + "-"*50)
        print("Meme Popularity Scatter Plot Generator")
        print("------------------------------------------------")
        print("1. Show Meme Dataset")
        print("2. Plot Scatter (User Age vs Popularity)")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == "1":
            print("\nMeme Dataset:\n")
            print(df)
            print("\n")

        elif choice == "2":
            plot_scatter(df)

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