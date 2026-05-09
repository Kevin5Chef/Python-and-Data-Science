import matplotlib.pyplot as plt
print("SY-5, Kevin Victor, Roll No.-30")
def plot_route_durations():

    # Curated flight route data (route: (airline, duration_in_hours))
    flight_data = {
        "SIN→JFK": ("Singapore Airlines", 18.83),
        "SIN→EWR": ("Singapore Airlines", 18.42),
        "DOH→AKL": ("Qatar Airways", 17.58),
        "PER→LHR": ("Qantas", 17.33),
        "DXB→AKL": ("Emirates", 17.17),
        "SIN→LAX": ("Singapore Airlines", 17.83)
    }

    routes = list(flight_data.keys())
    durations = [flight_data[r][1] for r in routes]

    plt.figure(figsize=(10, 6))
    plt.bar(routes, durations, color="skyblue")
    plt.title("Famous Long-Haul Flight Durations")
    plt.xlabel("Route")
    plt.ylabel("Duration (Hours)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def main_menu():
    while True:
        print("\n" + "-" * 50)
        print("Aviation Route Duration Bar Chart Generator")
        print("------------------------------------------------")
        print("1. Show Bar Chart of Flight Durations")
        print("2. Exit")

        choice = input("\nEnter your choice (1-2): ")

        if choice == "1":
            plot_route_durations()
        elif choice == "2":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()