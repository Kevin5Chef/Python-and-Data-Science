import pandas as pd
import time
import numpy as np
import os
print("SY-5, Kevin Victor, Roll No.-30")
# -------------------------------------------------------------------
# 1) SEMICONDUCTOR FAB METRIC DEFINITIONS (industry referenced)
# -------------------------------------------------------------------
# These are key KPIs tracked in wafer fab operations:
# - "Wafer_Yield": % of wafers produced without defects (higher is better). :contentReference[oaicite:2]{index=2}
# - "Defect_Density": defects per cm² (lower is better). :contentReference[oaicite:3]{index=3}
# - "Throughput_WPD": Wafers processed per day. :contentReference[oaicite:4]{index=4}
# - "Cycle_Time_days": elapsed fab cycle time (days). :contentReference[oaicite:5]{index=5}
# - "OEE_pct": Overall Equipment Effectiveness percentage. :contentReference[oaicite:6]{index=6}
# - "WIP": Work In Progress: active wafers still in process. :contentReference[oaicite:7]{index=7}

# -------------------------------------------------------------------
# 2) SIMULATED FAB DATA GENERATOR (dummy representative values)
# -------------------------------------------------------------------
def generate_dummy_fab_data(num=100):
    """Generate dummy semiconductor fab data resembling real KPI ranges."""
    return pd.DataFrame({
        "Wafer_Yield": np.random.normal(88, 4, num).clip(50, 99),
        "Defect_Density": np.random.normal(0.6, 0.15, num).clip(0.1, 2.0),
        "Throughput_WPD": np.random.normal(1200, 200, num).clip(800, 2000),
        "Cycle_Time_days": np.random.normal(5, 1, num).clip(2, 10),
        "OEE_pct": np.random.normal(82, 5, num).clip(50, 95),
        "WIP": np.random.randint(800, 1700, num),
    })

# -------------------------------------------------------------------
# 3) DESCRIPTIVE STATISTICS + ANALYSIS FUNCTIONS
# -------------------------------------------------------------------
def show_descriptive_statistics(df):
    """Print summary statistics in tabular form."""
    print("\n=== Descriptive Statistics ===")
    print(df.describe().round(2))

def generate_insights(df):
    """Generate one-sentence insights based on current fab data."""
    insights = []
    insights.append(
        f"Average wafer yield ({df['Wafer_Yield'].mean():.1f}%) "
        f"suggests room to improve quality control to >90%."
    )
    insights.append(
        f"Average defect density ({df['Defect_Density'].mean():.2f} defects/cm²) "
        f"indicates closer monitoring at critical process steps."
    )
    insights.append(
        f"Throughput ({df['Throughput_WPD'].mean():.1f} WPD) is moderate – optimizing bottleneck tools can increase production."
    )
    insights.append(
        f"Cycle time (~{df['Cycle_Time_days'].mean():.1f} days) indicates lead time might be reduced with better line balance."
    )
    insights.append(
        f"OEE (~{df['OEE_pct'].mean():.1f}%) shows productivity can improve via automation and reduced downtime."
    )
    insights.append(
        f"Work in progress (avg {df['WIP'].mean():.0f}) suggests potential rebalancing to reduce congestion."
    )
    return insights

# -------------------------------------------------------------------
# 4) HELPER FUNCTIONS
# -------------------------------------------------------------------
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# -------------------------------------------------------------------
# 5) INTERACTIVE CLI MENU
# -------------------------------------------------------------------
def fab_analysis_menu():
    clear_console()
    print("SEMICONDUCTOR FAB STATISTICS ANALYZER (Simulation)\n")
    print("1. Generate New Fab Data & Run Descriptive Analysis")
    print("2. Exit\n")

    try:
        choice = int(input("Enter option (1/2): "))
    except:
        choice = 0

    if choice == 1:
        df = generate_dummy_fab_data(120)
        clear_console()
        show_descriptive_statistics(df)

        print("\n--- Live Insights (1-sec interval simulation) ---")
        insights = generate_insights(df)
        for insight in insights:
            print(f"⏱ {insight}")
            time.sleep(1)
        input("\nPress Enter to return to the main menu...")
        fab_analysis_menu()

    elif choice == 2:
        print("Exiting simulation. Goodbye!")
    else:
        print("Invalid option, retry...\n")
        time.sleep(1)
        fab_analysis_menu()

# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    fab_analysis_menu()