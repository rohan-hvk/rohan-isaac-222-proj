import pandas as pd

def clean_enriched_netflix_data(df):
    # parse any style automatically:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df['Series Name'] = df['Series Name'].fillna('N/A')
    df['Season and Episode'] = df['Season and Episode'].fillna('N/A')
    return df[['Title','Date','Type','Series Name','Season and Episode','Watch Year']]
