#!/usr/bin/env python
# coding: utf-8

# In[19]:


import tkinter as tk  # standard Python GUI library
from tkinter import ttk, filedialog, messagebox  # themed widgets + file dialogs + popups
import pandas as pd  # data handling (tables / CSVs)
import pulp  # linear programming / optimization solver
import requests  # used for making API calls to fetch prices

# API abandoned due to implementation issues

# =========================
# LOAD DATA
# =========================
def load_data(filepath):
    df = pd.read_csv(filepath)  # read CSV file into a pandas DataFrame

    # remove dollar signs and commas from price column and convert to float
    df['Avg. Price'] = df['Avg. Price'].replace('[$,]', '', regex=True).astype(float)

    # clean ingredient names by removing numbering like "1. Chicken"
    df['Ingredient'] = df['Ingredient'].str.replace(r'^\d+\.', '', regex=True).str.strip()

    return df  # return cleaned DataFrame


# In[20]:


# =====================================================
# DIET PRESETS
# min calories, min protein, max fat, group minimums
# =====================================================
DIETS = {
    "High Protein": {
        "calories": 2000 * 7,  # weekly calories
        "protein": 150 * 7,    # weekly protein
        "fat": 80 * 7,         # max weekly fat
        "groups": {            # minimum servings per food group
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


# In[24]:


# =====================================================
# OPTIMIZER
# =====================================================

def optimize(df, calories, protein, fat, group_mins, preset="Custom"):

    # -------------------------------------------------
    # Apply Vegan / Vegetarian filtering first
    # -------------------------------------------------
    working_df = df.copy()

    if preset == "Vegan":
        working_df = working_df[
            working_df["Vegan"].astype(str).str.lower() == "true"
        ].reset_index(drop=True)

    elif preset == "Vegetarian":
        working_df = working_df[
            working_df["Vegetarian"].astype(str).str.lower() == "true"
        ].reset_index(drop=True)

    # Replace df after filtering
    df = working_df
    n = len(df)

    # If no foods remain after filtering
    if n == 0:
        return (
            "No Feasible Foods",
            0,
            [],
            0,
            0,
            0,
            {}
        )

    # -------------------------------------------------
    # NOW create optimization model
    # -------------------------------------------------
    prob = pulp.LpProblem("MealPlan", pulp.LpMinimize)

    # servings variables
    x = [
        pulp.LpVariable(f"x{i}", lowBound=0, cat="Integer")
        for i in range(n)
    ]

    # distinct food indicators
    y = [
        pulp.LpVariable(f"y{i}", cat="Binary")
        for i in range(n)
    ]

    # slack vars
    slack = {
        grp: pulp.LpVariable(f"slack_{grp}", lowBound=0)
        for grp in group_mins
    }

    PENALTY = 1000
    BIG_M = 50
    MIN_DISTINCT = 15

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
    # Nutrition constraints
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
    # Food group constraints
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
    # Distinct foods
    # -------------------------------------------------
    for i in range(n):
        prob += x[i] <= BIG_M * y[i]
        prob += x[i] >= y[i]

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


# In[25]:


# =====================================================
# GUI
# =====================================================
class App:
    def __init__(self, root):
        self.root = root  # store root window
        self.df = None    # will hold dataset

        root.title("Meal Prep Optimizer")  # window title
        root.geometry("760x700")  # window size

        # button to load CSV file
        ttk.Button(
            root,
            text="Load CSV",
            command=self.load_file
        ).pack(pady=5)



        # variable to store selected diet
        self.diet_var = tk.StringVar()

        # dropdown menu for diet selection
        self.combo = ttk.Combobox(
            root,
            textvariable=self.diet_var,
            values=list(DIETS.keys()) + ["Custom"],
            state="readonly"
        )
        self.combo.pack(pady=5)

        # trigger autofill when diet selected
        self.combo.bind("<<ComboboxSelected>>", self.autofill)

        # input fields for constraints
        self.cal = self.make_entry("Min Calories / week")
        self.protein = self.make_entry("Min Protein (g) / week")
        self.fat = self.make_entry("Max Fat (g) / week")

        # run optimization button
        ttk.Button(
            root,
            text="Run Optimization",
            command=self.run
        ).pack(pady=10)

        # output text box for displaying results
        self.output = tk.Text(root, height=30, width=90)
        self.output.pack(padx=10, pady=10)

    def make_entry(self, label):
        ttk.Label(self.root, text=label).pack()  # label for input
        e = ttk.Entry(self.root)  # entry box
        e.pack()
        return e  # return entry widget

    def load_file(self):
        # open file dialog to select CSV
        path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if not path:
            return  # exit if user cancels

        try:
            self.df = load_data(path)  # load dataset
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
            return  # do nothing if no data loaded

        self.df = update_prices(self.df)  # update prices

        messagebox.showinfo(
            "Updated",
            "Prices updated."
        )

    def autofill(self, event):
        diet = self.diet_var.get()  # get selected diet

        if diet in DIETS:
            vals = DIETS[diet]

            # autofill input fields with preset values
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
            # read user input
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

        # choose group constraints
        if preset in DIETS:
            group_mins = DIETS[preset]["groups"]
        else:
            # default values for custom diet
            group_mins = {
                "protein": 10,
                "carb": 8,
                "fat": 4,
                "produce": 10
            }

        # run optimizer
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
            group_mins,
            preset
        )

        # clear previous output
        self.output.delete(1.0, tk.END)

        # display results
        self.output.insert(
            tk.END,
            f"Solver Status: {status}\n"
        )

        self.output.insert(
            tk.END,
            f"Total Cost: ${total_cost:.2f}\n\n"
        )

        # table header
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

        # print each selected food
        for name, qty, cost, grp, unit in items:
            self.output.insert(
                tk.END,
                f"{name:<18}|{grp:<13}|{qty} units |{unit:<10}|${cost:.2f} |\n"
            )

        # print totals
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


# In[26]:


# =========================
# RUN
# =========================
root = tk.Tk()  # create main window
app = App(root)  # initialize app
root.mainloop()  # start GUI event loop

