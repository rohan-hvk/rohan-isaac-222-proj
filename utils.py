import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load one Netflix CSV and:
      - parse Date (auto‐detect ISO or M/D/YY)
      - fill missing Series/Episode
      - return standard columns
    """
    df = pd.read_csv(path)

    # parse ISO dates only (all CSVs are now YYYY-MM-DD)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

    # — now fill the other missing fields —
    df['Series Name']        = df['Series Name'].fillna('N/A')
    df['Season and Episode'] = df['Season and Episode'].fillna('N/A')

    return df[['Title','Date','Type','Series Name',
               'Season and Episode','Watch Year']]


def plot_type_distribution(df: pd.DataFrame, user: str):
    """
    Bar chart of % Movies vs Shows.
    """
    ax = df['Type'].value_counts(normalize=True).plot(
        kind='bar', title=f"{user}: Movies vs Shows %"
    )
    ax.set_ylabel("Proportion")
    plt.show()


def plot_yearly_activity(df: pd.DataFrame, user: str):
    ax = df['Watch Year']\
           .value_counts()\
           .sort_index()\
           .plot(kind='bar', title=f"{user}: Watch Activity by Year")
    ax.set_ylabel("Count")
    ax.set_xlabel("Year")
    plt.show()


def plot_top_series(df: pd.DataFrame, user: str, n=10):
    ax = df['Series Name']\
           .value_counts()\
           .head(n)\
           .plot(kind='barh', title=f"{user}: Top {n} Series")
    ax.set_ylabel("Series Name")
    plt.show()

def plot_top_shows(df: pd.DataFrame, user: str, n: int = 5):
    """
    Bar chart of the top n most-watched shows (by Series Name).
    Only counts rows where Type == 'Show'.
    """
    shows = df[df['Type'] == 'Show']
    top5 = shows['Series Name'].value_counts().head(n)
    ax = top5.plot(
        kind='barh',
        title=f"{user}: Top {n} Shows",
        figsize=(6, 4)
    )
    ax.invert_yaxis()
    ax.set_xlabel("View Count")
    ax.set_ylabel("Series Name")
    plt.tight_layout()
    plt.show()


def plot_viewing_over_time(df: pd.DataFrame, user: str, freq='ME'):
    # 0) ensure Date is datetime
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 1) drop bad dates
    tmp = df.dropna(subset=['Date']).set_index('Date')
    if tmp.empty:
        print(f"{user}: no valid dates to plot.")
        return

    # 2) resample
    series = tmp.resample(freq).size()
    if series.empty:
        print(f"{user}: no data after resampling at freq={freq}.")
        return

    # 3) plot
    ax = series.plot(kind='line', title=f"{user}: Viewing Over Time")
    ax.set_ylabel("Sessions")
    ax.set_xlabel("Date")
    plt.show()



def ttest_shows_2024(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    One‐sided Welch’s t-test on 2024 Show proportions.
    Returns (t_stat, p_value).
    """
    a = (df1[df1['Watch Year']==2024]['Type']=='Show').astype(int)
    b = (df2[df2['Watch Year']==2024]['Type']=='Show').astype(int)
    return ttest_ind(a, b, equal_var=False)
