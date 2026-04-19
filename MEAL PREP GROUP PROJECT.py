#!/usr/bin/env python
# coding: utf-8

# In[53]:


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import pulp
import requests

# =========================
# CONFIG
# =========================
API_KEY = "YOUR_API_KEY_HERE"  # for price API (RapidAPI or Kroger)

# =========================
# LOAD DATA
# =========================
def load_data(filepath):
    df = pd.read_csv(filepath)

    df['Avg. Price'] = df['Avg. Price'].replace('[$,]', '', regex=True).astype(float)
    df['Ingredient'] = df['Ingredient'].str.replace(r'^\d+\.', '', regex=True).str.strip()

    return df


# In[54]:


# =========================
# INTERNET PRICE FETCHER
# =========================
def fetch_price_online(ingredient):
    """
    Example using a generic API structure.
    Replace with real endpoint.
    """
    try:
        url = "https://api.example.com/price"
        headers = {"X-API-Key": API_KEY}

        response = requests.get(url, headers=headers, params={"query": ingredient})

        if response.status_code == 200:
            data = response.json()
            return float(data["price"])
    except:
        pass

    return None  # fallback if fails


def update_prices(df):
    for i in range(len(df)):
        ingredient = df.loc[i, "Ingredient"]
        new_price = fetch_price_online(ingredient)

        if new_price:
            df.loc[i, "Avg. Price"] = new_price

    return df


# =====================================================
# DIET PRESETS
# min calories, min protein, max fat, group minimums
# =====================================================
DIETS = {
    "High Protein": {
        "calories": 2000 * 7,
        "protein": 150 * 7,
        "fat": 80 * 7,
        "groups": {
            "protein": 14,
            "carb": 10,
            "fat": 5,
            "produce": 14
        }
    },

    "Keto": {
        "calories": 2000 * 7,
        "protein": 110 * 7,
        "fat": 150 * 7,
        "groups": {
            "protein": 12,
            "carb": 3,
            "fat": 10,
            "produce": 10
        }
    },

    "Vegan": {
        "calories": 2000 * 7,
        "protein": 85 * 7,
        "fat": 70 * 7,
        "groups": {
            "protein": 12,
            "carb": 10,
            "fat": 5,
            "produce": 14
        }
    },

    "Vegetarian": {
        "calories": 2000 * 7,
        "protein": 90 * 7,
        "fat": 75 * 7,
        "groups": {
            "protein": 12,
            "carb": 10,
            "fat": 5,
            "produce": 14
        }
    },

    "Weight Loss": {
        "calories": 1500 * 7,
        "protein": 100 * 7,
        "fat": 50 * 7,
        "groups": {
            "protein": 14,
            "carb": 7,
            "fat": 4,
            "produce": 14
        }
    }
}


# In[55]:


# =====================================================
# OPTIMIZER
# =====================================================
def optimize(df, calories, protein, fat, group_mins):
    prob = pulp.LpProblem("MealPlan", pulp.LpMinimize)

    n = len(df)

    # -------------------------------------------------
    # Decision variables
    # -------------------------------------------------
    # servings of each food
    x = [
        pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer")
        for i in range(n)
    ]

    # binary usage variables (food selected or not)
    y = [
        pulp.LpVariable(f"y{i}", cat="Binary")
        for i in range(n)
    ]

    # slack for soft food-group constraints
    slack = {
        grp: pulp.LpVariable(f"slack_{grp}", lowBound=0)
        for grp in group_mins
    }

    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------
    PENALTY = 1000      # soft group penalty
    BIG_M = 50         # max servings if selected
    MIN_DISTINCT = 15   # required number of different foods

    # -------------------------------------------------
    # Objective
    # -------------------------------------------------
    prob += (
        pulp.lpSum(
            x[i] * df.loc[i, "Avg. Price"]
            for i in range(n)
        )
        +
        PENALTY * pulp.lpSum(slack[g] for g in slack)
    )

    # -------------------------------------------------
    # Hard Nutrition Constraints
    # -------------------------------------------------
    prob += pulp.lpSum(
        x[i] * df.loc[i, "Cal"]
        for i in range(n)
    ) >= calories

    prob += pulp.lpSum(
        x[i] * df.loc[i, "Protein (g)"]
        for i in range(n)
    ) >= protein

    prob += pulp.lpSum(
        x[i] * df.loc[i, "Fat (g)"]
        for i in range(n)
    ) <= fat

    # -------------------------------------------------
    # Soft Food Group Minimums
    # -------------------------------------------------
    for grp, minimum in group_mins.items():
        prob += (
            pulp.lpSum(
                x[i]
                for i in range(n)
                if df.loc[i, "Food Group"] == grp
            )
            + slack[grp]
            >= minimum
        )

    # -------------------------------------------------
    # Distinct Foods Constraint
    # -------------------------------------------------
    for i in range(n):
        prob += x[i] <= BIG_M * y[i]
        prob += x[i] >= y[i]

    # At least 5 different foods used
    prob += pulp.lpSum(y[i] for i in range(n)) >= MIN_DISTINCT

    # -------------------------------------------------
    # Solve
    # -------------------------------------------------
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]

    # -------------------------------------------------
    # Build output
    # -------------------------------------------------
    result = []
    total_cost = 0

    for i in range(n):
        val = x[i].value()

        if val is not None and val > 0:
            qty = int(round(val))
            cost = qty * df.loc[i, "Avg. Price"]
            total_cost += cost

            result.append(
                (
                    df.loc[i, "Ingredient"],
                    qty,
                    cost,
                    df.loc[i, "Food Group"],
                    df.loc[i, "Unit"]
                )
            )

    total_cal = sum(
        (x[i].value() or 0) * df.loc[i, "Cal"]
        for i in range(n)
    )

    total_protein = sum(
        (x[i].value() or 0) * df.loc[i, "Protein (g)"]
        for i in range(n)
    )

    total_fat = sum(
        (x[i].value() or 0) * df.loc[i, "Fat (g)"]
        for i in range(n)
    )

    slack_vals = {
        g: slack[g].value() or 0
        for g in slack
    }

    return (
        status,
        total_cost,
        result,
        total_cal,
        total_protein,
        total_fat,
        slack_vals
    )


# In[56]:


# =====================================================
# GUI
# =====================================================
class App:
    def __init__(self, root):
        self.root = root
        self.df = None

        root.title("Meal Prep Optimizer")
        root.geometry("760x700")

        ttk.Button(
            root,
            text="Load CSV",
            command=self.load_file
        ).pack(pady=5)

        ttk.Button(
            root,
            text="Update Prices (Internet)",
            command=self.update_prices_gui
        ).pack(pady=5)

        # Diet preset dropdown
        self.diet_var = tk.StringVar()

        self.combo = ttk.Combobox(
            root,
            textvariable=self.diet_var,
            values=list(DIETS.keys()) + ["Custom"],
            state="readonly"
        )
        self.combo.pack(pady=5)
        self.combo.bind("<<ComboboxSelected>>", self.autofill)

        # Inputs
        self.cal = self.make_entry("Min Calories / week")
        self.protein = self.make_entry("Min Protein (g) / week")
        self.fat = self.make_entry("Max Fat (g) / week")

        ttk.Button(
            root,
            text="Run Optimization",
            command=self.run
        ).pack(pady=10)

        self.output = tk.Text(root, height=30, width=90)
        self.output.pack(padx=10, pady=10)

    def make_entry(self, label):
        ttk.Label(self.root, text=label).pack()
        e = ttk.Entry(self.root)
        e.pack()
        return e

    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if not path:
            return

        try:
            self.df = load_data(path)
            messagebox.showinfo(
                "Success",
                "CSV loaded successfully."
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                str(e)
            )

    def update_prices_gui(self):
        if self.df is None:
            return

        self.df = update_prices(self.df)

        messagebox.showinfo(
            "Updated",
            "Prices updated."
        )

    def autofill(self, event):
        diet = self.diet_var.get()

        if diet in DIETS:
            vals = DIETS[diet]

            self.cal.delete(0, tk.END)
            self.cal.insert(0, vals["calories"])

            self.protein.delete(0, tk.END)
            self.protein.insert(0, vals["protein"])

            self.fat.delete(0, tk.END)
            self.fat.insert(0, vals["fat"])

    def run(self):
        if self.df is None:
            messagebox.showwarning(
                "No Data",
                "Load a CSV first."
            )
            return

        try:
            calories = float(self.cal.get())
            protein = float(self.protein.get())
            fat = float(self.fat.get())
        except:
            messagebox.showerror(
                "Input Error",
                "Enter valid numeric values."
            )
            return

        preset = self.diet_var.get()

        if preset in DIETS:
            group_mins = DIETS[preset]["groups"]
        else:
            # Custom defaults
            group_mins = {
                "protein": 10,
                "carb": 8,
                "fat": 4,
                "produce": 10
            }

        (
            status,
            total_cost,
            items,
            total_cal,
            total_protein,
            total_fat,
            slack_vals
        ) = optimize(
            self.df,
            calories,
            protein,
            fat,
            group_mins
        )

        self.output.delete(1.0, tk.END)

        self.output.insert(
            tk.END,
            f"Solver Status: {status}\n"
        )

        self.output.insert(
            tk.END,
            f"Total Cost: ${total_cost:.2f}\n\n"
        )

        self.output.insert(
            tk.END,
            "MEAL PLAN\n"
            "-----------------------------------------------------------|\n"
        )
        self.output.insert(
            tk.END,
            "Ingredient        |Food Group   |Amount  |Unit Type |Cost  |\n"
            "-----------------------------------------------------------|\n"
        )
        
        for name, qty, cost, grp in items:
            self.output.insert(
                tk.END,
                f"{name:<18}|{grp:<13}|{qty} units |{unit:<10}|${cost:.2f} |\n""
            )

        self.output.insert(
            tk.END,
            "-----------------------------------------------------------|\n"
            "\nNUTRITION TOTALS\n"
            "------------------------------\n"
        )

        self.output.insert(
            tk.END,
            f"Calories: {total_cal:.0f}\n"
        )
        self.output.insert(
            tk.END,
            f"Protein: {total_protein:.0f}g\n"
        )
        self.output.insert(
            tk.END,
            f"Fat: {total_fat:.0f}g\n"
        )


# In[57]:


# =========================
# RUN
# =========================
root = tk.Tk()
app = App(root)
root.mainloop()

