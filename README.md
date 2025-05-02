# Netflix Viewing History Analysis Project

## Overview

This project analyzes personal Netflix watch history data for two individuals—**Isaac** and **Rohan**—with the goal of uncovering viewing patterns, trends, and differences in media preferences. By transforming raw CSV data into meaningful visualizations and statistical insights, the project showcases how data analysis can be applied to everyday digital behavior.

---

## Objectives

- **Clean and standardize** Netflix viewing history datasets.
- **Visualize trends** in user behavior (e.g., time of year, weekday preferences, movie vs. show ratios).
- Identify and compare **top-watched series**.
- Use statistical tests (Welch’s t-test) to determine whether the proportion of shows watched differs significantly between the two users in 2024.

---

## Why This Project Matters

With streaming being a major part of daily life, analyzing personal watch data helps us better understand:
- **Digital consumption habits**
- **Time management and screen time trends**
- **Differences in content preferences** between individuals

This kind of analysis demonstrates the power of data science to reflect real-world behavior and enhance self-awareness.

---

## Tools and Technologies

- **Python** (data analysis and visualization)
- **Jupyter Notebook** (interactive environment)
- **Libraries used:**
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `matplotlib` for plotting
  - `scipy.stats` for statistical testing

---

## Project Files

```
📁 Project Root
├── analysis.ipynb              # Main notebook with full analysis and charts
├── utils.py                    # Helper functions for data loading, cleaning, plotting, and testing
├── IsaacNetflixHistory.csv     # Isaac's Netflix viewing history
├── RohanNetflixHistory.csv     # Rohan's Netflix viewing history
└── README.md                   # Project overview and documentation
```

---

## How to Run the Project

1. **Install Python** (version 3.8 or higher recommended).
2. Install required libraries:
   ```
   pip install pandas numpy matplotlib scipy
   ```
3. Open `analysis.ipynb` in Jupyter Notebook.
4. Run the notebook cells in order to:
   - Load and clean data using `utils.py`
   - Generate all visualizations
   - Perform statistical testing

---

## Visualizations Included

- **Movies vs Shows Percentage**  
- **Yearly Watch Activity**  
- **Most-Watched Series (Top 10)**  
- **Viewing Trends by Month and Weekday**  
- **Line Graph of Viewing Over Time**  
- **Top 5 Shows Only**  

Each graph is customized for Isaac and Rohan individually, making it easy to compare behaviors visually.

---

## Statistical Test

A one-sided **Welch’s t-test** is used to determine if there's a statistically significant difference in the **proportion of shows** watched by each user in **2024**.

