import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree

def load_and_clean(path):
    df = pd.read_csv(path)  # load CSV
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')  # convert date to datetime
    df['Series Name'] = df['Series Name'].fillna('N/A')  # fill missing series name
    df['Season and Episode'] = df['Season and Episode'].fillna('N/A')  # fill missing episode info
    return df[['Title', 'Date', 'Type', 'Series Name', 'Season and Episode', 'Watch Year']]

def plot_type_distribution(df, user):
    ax = df['Type'].value_counts(normalize=True).plot(kind='bar', title=f"{user}: Movies vs Shows %")  # bar plot of type
    ax.set_ylabel("Proportion")
    plt.show()

def plot_yearly_activity(df, user):
    ax = df['Watch Year'].value_counts().sort_index().plot(kind='bar', title=f"{user}: Watch Activity by Year")  # year counts
    ax.set_ylabel("Count")
    ax.set_xlabel("Year")
    plt.show()

def plot_top_shows(df, user, n=5):
    shows = df[df['Type'] == 'Show']  # filter shows only
    top5 = shows['Series Name'].value_counts().head(n)  # top n shows
    ax = top5.plot(kind='barh', title=f"{user}: Top {n} Shows", figsize=(6, 4))
    ax.invert_yaxis()
    ax.set_xlabel("View Count")
    ax.set_ylabel("Series Name")
    plt.tight_layout()
    plt.show()

def ttest_shows_2024(df1, df2):
    a = (df1[df1['Watch Year'] == 2024]['Type'] == 'Show').astype(int)  # convert to binary
    b = (df2[df2['Watch Year'] == 2024]['Type'] == 'Show').astype(int)
    return ttest_ind(a, b, equal_var=False)  # one-sided t-test

def ztest_breaking_bad(df1, df2):
    isaac_count = (df1['Series Name'] == 'Breaking Bad').sum()
    rohan_count = (df2['Series Name'] == 'Breaking Bad').sum()
    counts = [isaac_count, rohan_count]
    nobs = [len(df1), len(df2)]
    return proportions_ztest(counts, nobs, alternative='larger')  # one-sided z-test

def plot_weekday_distribution(df, user):
    tmp = df.dropna(subset=['Date'])  # drop null dates
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts = tmp['Date'].dt.day_name().value_counts().reindex(weekdays, fill_value=0)  # count weekdays
    ax = counts.plot(kind='bar', title=f"{user}: Views by Weekday", figsize=(6, 4))
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Session Count")
    plt.tight_layout()
    plt.show()

def plot_monthly_distribution(df, user):
    dates = df.dropna(subset=['Date'])['Date']
    counts = dates.dt.month.value_counts(sort=False).sort_index()
    labels = [calendar.month_abbr[m] for m in counts.index]
    ax = counts.plot(kind='bar', title=f"{user}: Sessions by Month of Year", figsize=(8, 4), width=0.8)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Sessions")
    plt.tight_layout()
    plt.show()

def plot_decision_tree(user_csv_1, user_csv_2):
    df1 = pd.read_csv(user_csv_1)
    df2 = pd.read_csv(user_csv_2)
    df1['user'] = 'Isaac'
    df2['user'] = 'Rohan'
    df = pd.concat([df1, df2])

    # create features
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['weekday'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month
    df['is_movie'] = (df['Type'].str.lower() == 'movie').astype(int)
    df['is_2024_show'] = ((df['Date'].dt.year == 2024) & (df['Type'].str.lower() == 'show')).astype(int)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    for m in [5, 6, 11]:
        df[f'month_{m}'] = (df['month'] == m).astype(int)

    # encode top 10 series names
    top10 = df['Series Name'].value_counts().nlargest(10).index
    df['series_short'] = df['Series Name'].where(df['Series Name'].isin(top10), 'Other')
    enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    ohe = enc.fit_transform(df[['series_short']]).toarray()
    series_cols = enc.get_feature_names_out(['series_short'])
    df = pd.concat([df, pd.DataFrame(ohe, columns=series_cols, index=df.index)], axis=1)

    features = ['weekday', 'month', 'is_movie', 'is_2024_show', 'is_weekend', 'month_5', 'month_6', 'month_11'] + list(series_cols)
    X = df[features]
    y = df['user']

    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(15, 8))
    tree.plot_tree(clf, feature_names=features, class_names=clf.classes_, filled=True, rounded=True, ax=ax)
    plt.tight_layout()
    plt.show()

def run_knn_classification(user_csv_1, user_csv_2, k_range=range(1, 21)):
    df1 = pd.read_csv(user_csv_1)
    df2 = pd.read_csv(user_csv_2)
    df1['user'] = 'Isaac'
    df2['user'] = 'Rohan'
    df = pd.concat([df1, df2])

    # feature engineering
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['weekday'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month
    df['is_movie'] = (df['Type'].str.lower() == 'movie').astype(int)

    top10 = df['Series Name'].value_counts().nlargest(10).index
    df['series_short'] = df['Series Name'].where(df['Series Name'].isin(top10), 'Other')

    enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    ohe = enc.fit_transform(df[['series_short']]).toarray()
    series_cols = enc.get_feature_names_out(['series_short'])
    df = pd.concat([df, pd.DataFrame(ohe, columns=series_cols, index=df.index)], axis=1)

    X = df[['weekday', 'month', 'is_movie'] + list(series_cols)]
    y = df['user']
    X_scaled = MinMaxScaler().fit_transform(X)  # normalize features
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=0)

    best_k = None
    best_acc = 0
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"k={k:2d}, acc={acc:.4f}")
        if acc > best_acc:
            best_k, best_acc = k, acc

    print(f"\nBest k = {best_k}, accuracy = {best_acc:.4f}")

def evaluate_classifiers(path1, path2):
    # load CSVs and label the user
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df1['user'] = 'Isaac'
    df2['user'] = 'Rohan'
    df = pd.concat([df1, df2])  # combine both datasets

    # basic feature engineering
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['weekday'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month
    df['is_movie'] = (df['Type'].str.lower() == 'movie').astype(int)

    # encode top 10 series names
    top10 = df['Series Name'].value_counts().nlargest(10).index
    df['series_short'] = df['Series Name'].where(df['Series Name'].isin(top10), 'Other')
    enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    ohe = enc.fit_transform(df[['series_short']]).toarray()
    series_cols = enc.get_feature_names_out(['series_short'])
    df = pd.concat([df, pd.DataFrame(ohe, columns=series_cols, index=df.index)], axis=1)

    # define feature set and labels
    X = df[['weekday', 'month', 'is_movie'] + list(series_cols)]
    y = df['user']

    # normalize features
    X_scaled = MinMaxScaler().fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=0
    )

    # run kNN for k=1 to 20, keep best accuracy
    knn_acc = 0
    for k in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        if acc > knn_acc:
            best_k, knn_acc = k, acc  # store best k and accuracy

    # train decision tree classifier
    dt_model = DecisionTreeClassifier(random_state=0)
    dt_model.fit(X_train, y_train)
    dt_acc = dt_model.score(X_test, y_test)

    return best_k, knn_acc, dt_acc