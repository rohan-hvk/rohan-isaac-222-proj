import pandas as pd

def clean_enriched_netflix_data(df):
    # Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
    
    # imssing stuff is n/a
    df['Series Name'] = df['Series Name'].fillna('N/A')
    df['Season and Episode'] = df['Season and Episode'].fillna('N/A')
    
    # Reorder columns
    df = df[['Title', 'Date', 'Type', 'Series Name', 'Season and Episode', 'Watch Year']]
    
    return df



