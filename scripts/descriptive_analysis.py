import pandas as pd
import re

def headline_length_stats(data: pd.DataFrame) -> pd.Series:
    """
    Calculate and return descriptive statistics for the length of headlines.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
    
    Returns:
        pd.Series: Descriptive statistics for headline lengths.
    """
    data['headline_length'] = data['headline'].apply(len)
    
    return data['headline_length'].describe()

def articles_per_publisher(data: pd.DataFrame) -> pd.Series:
    """
    Count the number of articles per publisher.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
    
    Returns:
        pd.Series: Counts of articles per publisher.
    """
    return data['publisher'].value_counts()

def articles_by_day_of_week(data: pd.DataFrame) -> pd.Series:
    """
    Analyze the distribution of articles by day of the week.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
    
    Returns:
        pd.Series: Counts of articles by day of the week.
    """
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        raise ValueError("The 'date' column must be in datetime format.")
    
    data['day_of_week'] = data['date'].dt.day_name()
    
    return data['day_of_week'].value_counts()


def articles_by_time(data: pd.DataFrame) -> pd.Series:
    """
    Analyze the distribution of articles by time of the day.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data with a 'date' column in datetime format.
    
    Returns:
        pd.Series: Counts of articles by time of the day.
    """
    # Ensure the 'date' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        raise ValueError("The 'date' column must be in datetime format.")
    
    # Extract the hour of the day
    data['time'] = data['date'].dt.time
    
    # Return counts of articles by hour of the day
    return data['time'].value_counts().sort_index()


def extract_domains(email: str) -> str:
    """
    Extract domain from an email address.
    
    Parameters:
    - email (str): Email address to extract the domain from.
    
    Returns:
    - str: Domain part of the email address.
    """
    match = re.search(r"@([\w\.-]+)", email)
    return match.group(1) if match else None

def identify_unique_domains(data: pd.DataFrame) -> pd.DataFrame:
    """
    Identify unique domains from email addresses used as publisher names.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the articles with a 'publisher' column.
    
    Returns:
    - pd.DataFrame: DataFrame containing unique domains and their frequency.
    """
    data['domain'] = data['publisher'].apply(extract_domains)
    domain_counts = data['domain'].value_counts().reset_index()
    domain_counts.columns = ['domain', 'count']
    return domain_counts