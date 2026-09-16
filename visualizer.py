'''Python 3.14.5 (tags/v3.14.5:5607950, May 10 2026, 10:43:50) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BaseAnalyzer:
    """Base class for OOP Inheritance."""
    def __init__(self, name="Sales Dataset"):
        self.dataset_name = name

    def display_info(self):
        print(f"\n--- Analyzer initialized for: {self.dataset_name} ---")


class SalesDataAnalyzer(BaseAnalyzer):
    """Main analyzer class encapsulating data analysis and visualization."""

    def __init__(self, file_path=None):
        super().__init__(name="Corporate Sales Analysis")
        self.data = None
        if file_path:
            self.load_data(file_path)

    def __del__(self):
        print("\n[Cleanup] SalesDataAnalyzer instance destroyed.")

    def load_data(self, file_path):
        """Load data from CSV or generate sample data if missing."""
        try:
            if os.path.exists(file_path):
                self.data = pd.read_csv(file_path)
                print(f"Loaded {file_path}")
            else:
                self._generate_sample_data()
        except Exception as e:
            print(f"Error loading file: {e}")
            self._generate_sample_data()

    def _generate_sample_data(self):
        """Generates synthetic sales dataset."""
        print("Generating sample data...")
        np.random.seed(42)
        self.data = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=100),
            "Product": np.random.choice(["Laptop", "Phone", "Tablet", "Monitor", "Headphones"], 100),
            "Region": np.random.choice(["North", "South", "East", "West"], 100),
            "Sales": np.random.randint(100, 1500, 100).astype(float),
            "Units_Sold": np.random.randint(1, 15, 100),
            "Profit": np.random.uniform(20, 400, 100).round(2)
        })
        self.data.loc[[5, 12], ["Sales", "Profit"]] = np.nan
        self.data.to_csv("sample_sales_data.csv", index=False)

    def _check_data(self):
        if self.data is None:
            print("No data loaded.")
            return False
        return True

    def explore_data(self):
        if not self._check_data(): return
        print(f"\n--- EXPLORATION ---\n1. Head:\n{self.data.head()}")
        print(f"\n2. Info:")
        self.data.info()
        print(f"\n3. Description:\n{self.data.describe()}")

    def clean_data(self):
        if not self._check_data(): return
        print("\n--- CLEANING ---")
        for col in ["Sales", "Profit", "Units_Sold"]:
            if col in self.data:
                self.data[col] = self.data[col].fillna(self.data[col].median())
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        print("Cleaning complete. Missing values remaining:\n", self.data.isnull().sum())

    def numpy_operations(self):
        if not self._check_data(): return
        s, p = self.data["Sales"].values, self.data["Profit"].values
        print("\n--- NUMPY OPS ---")
        print(f"Slicing (2-7): {s[2:7]}\n5th Element: {s[4]}")
        print("Profit Margin % (First 5):", ((p / s) * 100)[:5].round(2))

    def combine_and_split(self):
        if not self._check_data(): return
        splits = {reg: self.data[self.data["Region"] == reg] for reg in self.data["Region"].unique()}
        recombined = pd.concat(splits.values(), axis=0)
        print(f"\nSplit into {len(splits)} regions. Recombined row count: {len(recombined)}")

    def search_sort_filter(self):
        if not self._check_data(): return
        print("\n--- SEARCH & SORT ---")
        print("Sales > 1000 Count:", len(self.data[self.data["Sales"] > 1000]))
        print("Top 3 Profitable:\n", self.data.sort_values(by="Profit", ascending=False)[["Product", "Profit"]].head(3))

    def statistical_analysis(self):
        if not self._check_data(): return
        s = self.data["Sales"]
        print(f"\n--- STATS ---\nStd: {s.std():.2f} | Var: {s.var():.2f}")
        print(f"25th Pct: {s.quantile(0.25)} | 75th Pct: {s.quantile(0.75)}")

    def create_pivot_table(self):
        if not self._check_data(): return
        pivot = pd.pivot_table(self.data, values=["Sales", "Profit"], index="Region", columns="Product", aggfunc="sum", fill_value=0)
        print("\n--- PIVOT TABLE ---\n", pivot)
        return pivot

    def visualize_matplotlib(self):
        if not self._check_data(): return
        os.makedirs("output_plots", exist_ok=True)
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Sales Visualizations (Matplotlib)")

        p_sales = self.data.groupby("Product")["Sales"].sum()
        axs[0, 0].bar(p_sales.index, p_sales.values, color="skyblue")
        axs[0, 0].tick_params(axis='x', rotation=30)
        axs[0, 0].set_title("Sales by Product")

        d_sales = self.data.groupby("Date")["Sales"].sum()
        axs[0, 1].plot(d_sales.index, d_sales.values, color="green")
        axs[0, 1].set_title("Sales Trend")

        axs[0, 2].scatter(self.data["Sales"], self.data["Profit"], color="purple")
        axs[0, 2].set_title("Sales vs Profit")

        r_sales = self.data.groupby("Region")["Sales"].sum()
        axs[1, 0].pie(r_sales.values, labels=r_sales.index, autopct="%1.1f%%")
        axs[1, 0].set_title("Regional Share")

        axs[1, 1].hist(self.data["Sales"], bins=10, color="orange")
        axs[1, 1].set_title("Sales Distribution")

        p_daily = self.data.pivot_table(index="Date", columns="Region", values="Sales", aggfunc="sum").fillna(0)
        axs[1, 2].stackplot(p_daily.index, p_daily.T, labels=p_daily.columns)
        axs[1, 2].set_title("Stack Plot")

        plt.tight_layout()
        plt.savefig("output_plots/matplotlib_summary.png")
        plt.show()

    def visualize_seaborn(self):
        if not self._check_data(): return
        sns.set_theme(style="whitegrid")
        os.makedirs("output_plots", exist_ok=True)

        plt.figure(figsize=(7, 5))
        sns.heatmap(self.data.pivot_table(index="Region", columns="Product", values="Sales", aggfunc="mean"), annot=True, fmt=".1f", cmap="YlGnBu")
        plt.title("Avg Sales Heatmap")
        plt.savefig("output_plots/seaborn_heatmap.png")
        plt.show()

        plt.figure(figsize=(7, 4))
        sns.boxplot(x="Product", y="Profit", data=self.data, palette="Set2")
        plt.title("Profit Distribution")
        plt.savefig("output_plots/seaborn_boxplot.png")
        plt.show()

    def __add__(self, other):
        if isinstance(other, SalesDataAnalyzer):
            res = SalesDataAnalyzer()
            res.data = pd.concat([self.data, other.data], ignore_index=True)
            return res
        return self


def main():
    analyzer = SalesDataAnalyzer("sales_data.csv")
    menu = {
        "1": analyzer.explore_data,
        "2": analyzer.clean_data,
        "3": analyzer.numpy_operations,
        "4": analyzer.combine_and_split,
        "5": analyzer.search_sort_filter,
        "6": analyzer.statistical_analysis,
        "7": analyzer.create_pivot_table,
        "8": analyzer.visualize_matplotlib,
        "9": analyzer.visualize_seaborn,
    }

    while True:
        print("\n=== MENU ===\n1. Explore  2. Clean  3. NumPy Ops  4. Combine/Split  5. Search/Sort")
        print("6. Stats    7. Pivot  8. Matplotlib 9. Seaborn        10. Exit")
        choice = input("Choice (1-10): ").strip()
        
        if choice == "10":
            print("Goodbye!")
            break
        menu.get(choice, lambda: print("Invalid option."))()


if __name__ == "__main__":
    main()
