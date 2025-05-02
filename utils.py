import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import calendar
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


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

def ttest_shows_2024(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    One‐sided Welch’s t-test on 2024 Show proportions.
    Returns (t_stat, p_value).
    """
    a = (df1[df1['Watch Year']==2024]['Type']=='Show').astype(int)
    b = (df2[df2['Watch Year']==2024]['Type']=='Show').astype(int)
    return ttest_ind(a, b, equal_var=False)

def plot_weekday_distribution(df: pd.DataFrame, user: str):
    """
    Bar chart of total Netflix sessions by weekday (Mon → Sun).
    """
    # 1) pull out only valid dates
    tmp = df.dropna(subset=['Date'])
    # 2) count how many sessions fall on each weekday name
    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    counts = (
        tmp['Date']
          .dt
          .day_name()
          .value_counts()
          .reindex(weekdays, fill_value=0)
    )
    # 3) plot
    ax = counts.plot(
        kind='bar',
        title=f"{user}: Views by Weekday",
        figsize=(6,4)
    )
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Session Count")
    plt.tight_layout()
    plt.show()

def plot_monthly_distribution(df: pd.DataFrame, user: str):
    """
    Bar chart of total watch sessions aggregated by calendar month.
    Shows Jan (1) through Dec (12) counts so you can see
    which part of the year each user watches most.
    """
    # extract only valid dates
    dates = df.dropna(subset=['Date'])['Date']
    # count by month number
    counts = dates.dt.month.value_counts(sort=False).sort_index()
    
    # convert 1–12 into Jan, Feb, … Dec
    labels = [calendar.month_abbr[m] for m in counts.index]
    
    ax = counts.plot(
        kind='bar',
        title=f"{user}: Sessions by Month of Year",
        figsize=(8,4),
        width=0.8
    )
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Sessions")
    plt.tight_layout()
    plt.show()



def ztest_breaking_bad(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    One-sided z-test on proportion of Breaking Bad views.
    Assumes 'Series Name' is cleaned and consistently labeled.
    Returns (z_stat, p_value).
    """
    # Count of Breaking Bad views per user
    isaac_count = (df1['Series Name'] == 'Breaking Bad').sum()
    rohan_count = (df2['Series Name'] == 'Breaking Bad').sum()
    
    # Total view counts
    isaac_total = len(df1)
    rohan_total = len(df2)

    # Values for proportions_ztest
    counts = [isaac_count, rohan_count]
    nobs = [isaac_total, rohan_total]

    # One-sided test: Isaac > Rohan
    stat, pval = proportions_ztest(counts, nobs, alternative='larger')
    return stat, pval


def plot_decision_tree(user_csv_1: str, user_csv_2: str):
    """
    Load and label user data, engineer features, train a decision tree classifier,
    and display the classification tree.
    """
    # Load and label data
    df_isaac = pd.read_csv(user_csv_1)
    df_rohan = pd.read_csv(user_csv_2)
    df_isaac['user'] = 'Isaac'
    df_rohan['user'] = 'Rohan'
    df = pd.concat([df_isaac, df_rohan], ignore_index=True)

    # Feature engineering
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['weekday'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month
    df['is_movie'] = (df['Type'].str.lower() == 'movie').astype(int)
    df['is_2024_show'] = ((df['Date'].dt.year == 2024) & (df['Type'].str.lower() == 'show')).astype(int)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    for m in [5, 6, 11]:
        df[f'month_{m}'] = (df['month'] == m).astype(int)

    # Reduce series names
    top10 = df['Series Name'].value_counts().nlargest(10).index
    df['series_short'] = df['Series Name'].where(df['Series Name'].isin(top10), 'Other').fillna('Other')
    
    # One-hot encoding
    enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    series_ohe = enc.fit_transform(df[['series_short']]).toarray()
    series_cols = enc.get_feature_names_out(['series_short'])
    df_series = pd.DataFrame(series_ohe, columns=series_cols, index=df.index)
    df = pd.concat([df, df_series], axis=1)

    # Prepare features/target
    feature_cols = [
        'weekday','month','is_movie',
        'is_2024_show','is_weekend',
        'month_5','month_6','month_11'
    ] + list(series_cols)

    X = df[feature_cols]
    y = df['user']

    # Train + plot tree
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(15, 8))
    tree.plot_tree(
        clf,
        feature_names=feature_cols,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        ax=ax
    )
    plt.tight_layout()
    plt.show()


def run_knn_classification(user_csv_1: str, user_csv_2: str, k_range=range(1, 21)):
    """
    Load labeled data, engineer features, one-hot encode series names, and
    train KNN across a range of k values. Reports best accuracy and k.
    """
    # Load and label
    df1 = pd.read_csv(user_csv_1)
    df2 = pd.read_csv(user_csv_2)
    df1['user'] = 'Isaac'
    df2['user'] = 'Rohan'
    df = pd.concat([df1, df2], ignore_index=True)

    # Feature engineering
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['weekday'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month
    df['is_movie'] = (df['Type'].str.lower() == 'movie').astype(int)

    # Top 10 series → one-hot encode
    top10 = df['Series Name'].value_counts().nlargest(10).index
    df['series_short'] = df['Series Name'].where(df['Series Name'].isin(top10), 'Other').fillna('Other')
    enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    series_ohe = enc.fit_transform(df[['series_short']]).toarray()
    series_cols = enc.get_feature_names_out(['series_short'])
    df_series = pd.DataFrame(series_ohe, columns=series_cols, index=df.index)
    df = pd.concat([df, df_series], axis=1)

    # Define feature matrix and target
    feature_cols = ['weekday', 'month', 'is_movie'] + list(series_cols)
    X = df[feature_cols]
    y = df['user']

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=0
    )

    # Tune k
    best_k = None
    best_acc = 0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        print(f"k = {k:2d} → Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_k, best_acc = k, acc

    print(f"\nBest k = {best_k} with Accuracy = {best_acc:.4f}")


def evaluate_classifiers(path1, path2):
    import pandas as pd

    # Load and label
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df1['user'] = 'Isaac'
    df2['user'] = 'Rohan'
    df = pd.concat([df1, df2], ignore_index=True)

    # Feature engineering
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['weekday'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month
    df['is_movie'] = (df['Type'].str.lower() == 'movie').astype(int)

    top10 = df['Series Name'].value_counts().nlargest(10).index
    df['series_short'] = df['Series Name'].where(df['Series Name'].isin(top10), 'Other').fillna('Other')

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    series_ohe = enc.fit_transform(df[['series_short']]).toarray()
    series_cols = enc.get_feature_names_out(['series_short'])
    df_series = pd.DataFrame(series_ohe, columns=series_cols, index=df.index)
    df = pd.concat([df, df_series], axis=1)

    feature_cols = ['weekday', 'month', 'is_movie'] + list(series_cols)
    X = df[feature_cols]
    y = df['user']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=0
    )

    # k-NN
    best_k = None
    best_acc = 0
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        if acc > best_acc:
            best_k = k
            best_acc = acc

    # Decision Tree
    tree_model = DecisionTreeClassifier(random_state=0)
    tree_model.fit(X_train, y_train)
    tree_acc = tree_model.score(X_test, y_test)

    return best_k, best_acc, tree_acc
