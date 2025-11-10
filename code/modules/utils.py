#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:01:04 2024

@author: olya
"""
def save_variables(filepath,key,value,rounding=True):
    """
    Saves the given values in a file for future reference, e.g. to have an easy access
    when writing papers and for updating paper numbers automatically based on the values from
    the resulting output file. 
    
    Parameters
    ----------
    filepath : str
             where to store the variable values
    key: str
            an identifier
    value: any-type
           the target value

    Returns
    -------
    """

    import csv
    import numpy as np
    import fileinput
    
    updated = False
    
    # Only add quotes if the variable name contains a comma
    formatted_key = f'"{key}"' if ',' in key else key
    new_line = f"{formatted_key},{np.round(value,2) if rounding else value}\n"
    
    # Modify only the necessary line
    with fileinput.input(filepath, inplace=True) as file:
        for line in file:
            if line.startswith(f"{formatted_key},"):  # Ensure it matches the correct format
                print(new_line, end="")  # Replace only this line
                updated = True
            else:
                print(line, end="")  # Keep all other lines unchanged
    
    # If key wasn't in the file, append it at the end
    if not updated:
        with open(filepath, "a", newline="") as f:
            f.write(new_line)  # Append new key-value pair
                
def save_descriptive_stats(filepath,key,data):

    """
    Calculates the descriptive statistics of a given variable and saves them in a file
    
    Parameters
    ----------
    filepath : str
             where to store the variable values
    key: str
            an identifier
    value: any-type
           the target value

    Returns
    -------
    """

    import numpy as np
    from scipy.stats import median_abs_deviation as mad
    save_variables(filepath,f'{key}-N',sum(~np.isnan(data)))
    save_variables(filepath,f'{key}-mean',np.nanmean(data))
    save_variables(filepath,f'{key}-median',np.nanmedian(data))
    save_variables(filepath,f'{key}-std',np.nanstd(data))
    save_variables(filepath,f'{key}-mad',mad(data,nan_policy='omit'))
    save_variables(filepath,f'{key}-min',min(data))
    save_variables(filepath,f'{key}-max',max(data))


def map_7point_likert(filepath, question_column):
    """
    Maps a 7-point Likert scale to numerical values.

    Parameters:
    - survey_df (pd.DataFrame): The survey dataset.
    - response_column (str): The column containing 7-point Likert scale responses.

    Returns:
    - pd.DataFrame: The survey dataset with mapped numerical values.
    """
    
    import pandas as pd
    import numpy as np
    
    # Load the survey
    survey_df = pd.read_excel(filepath)
    
    # Define a mapping for the 7-point Likert scale
    likert_mapping = {'Disagree strongly = 1': 1, 'Agree strongly = 7': 7, 'No answer': np.nan}
    
    # Apply the mapping, remove nan values and return
    survey_df.replace(likert_mapping, inplace=True)
    mapped_survey_df = survey_df.dropna(subset=[question_column])    
    return mapped_survey_df [['external_id', question_column]]    


def significance_asterisk(value,lowest_threshold=0.05,N_bonferroni=1):
    import numpy as np
    if value < 0.001/N_bonferroni:
        return f'{value:.4f}***'
    elif value < 0.01/N_bonferroni:
        return f'{value:.4f}**'
    elif value <= lowest_threshold/N_bonferroni:
        return f'{value:.3f}*'
    else:
       return   f'{value:.3f}'

def align_monthly_data(donation_messages, ego_messages):
    """Align donation and ego messages to a common monthly timeline."""
    import pandas as pd
    all_messages_monthly = donation_messages.resample('M', on='datetime').sum()
    ego_messages_monthly = ego_messages.resample('M', on='datetime').sum()

    idx_range = pd.date_range(
        start=min(all_messages_monthly.index.min(), ego_messages_monthly.index.min()),
        end=max(all_messages_monthly.index.max(), ego_messages_monthly.index.max()),
        freq='M')

    all_messages_monthly = all_messages_monthly.reindex(idx_range, fill_value=0)
    ego_messages_monthly = ego_messages_monthly.reindex(idx_range, fill_value=0)

    return all_messages_monthly, ego_messages_monthly 

            
def bin_probability(values,bins,bin_labels):
    """
    Calculate the probability that the values fall within the specified bins.

    Parameters
    ----------
    values : array-like
        Variable values.
    bins : list of tuple
        Each tuple contains the lower and upper bounds of a bin.
    bin_labels : list
        Labels for each bin.

    Returns
    -------
    dict
        Probabilities for each bin.
    """
    response_bins = {key: None for key in bin_labels}
    for ind, bin_range in enumerate(bins):
        response_bins[bin_labels[ind]] = len([x for x in values if bin_range[0]<=x<bin_range[-1]])/len(values)
    
    # Check if the sum of probabilities is close to 1 or 0
    total_prob = sum(response_bins.values())
    if abs(1 - total_prob) >= 0.05:
        raise ValueError("The sum of probabilities is not close to 1.")   
        
    return response_bins
    
def get_relevant_messages(messages_table, donation_id, ego_id):
    """
    Get messages related to a specific donation and donor.
    
    Parameters
    ----------
    messages_table : pd.DataFrame
        A dataframe containing all message records with at least'donation_id', 'sender_id', 'conversation_id', 'datetime', 'word_count' columns.
    donation_id : int or str
        The identifier for the donation.
    ego_id : int or str
        The identifier for the donor (ego).

    Returns
    -------
    tuple of pd.DataFrame
        donation_messages : Messages related to the specified donation.
        ego_messages : Messages sent by the donor in that donation.
    """
    import pandas as pd
    donation_messages = messages_table[messages_table['donation_id'] == donation_id]
    ego_messages = donation_messages[donation_messages['sender_id'] == ego_id]
    return donation_messages, ego_messages

def validate_values(x, range_min, range_max,variable=''):
    """Validate that all values are within the expected range."""
    if not all((range_min <= item <= range_max) or np.isnan(item) for item in x):
        raise ValueError("The {variable} values are not within the expected range or are nan.")

def get_last_non_nan_value(x):
    """Get the last non-NaN value from a list of PDI values."""
    import math
    for item in reversed(x):
        if not math.isnan(item):
            return item
    return None

def convert_to_cumulative_time_labels(labels):
    """

    The input is strings representing time bins (e.g., '<5min', '5-10min', '>30min').
    The output simplifies these to cumulative time point markers:
        - '<5min' becomes '5min'
        - '5-10min' becomes '10min'
        - '>30min' becomes '30+min'

    Parameters:
        labels (list of str): List of time bin labels.

    Returns:
        list of str: Transformed labels as cumulative time points.
    """
    
    cumulative_labels = []
    for i in range(len(labels)):
        if i == 0:
            time_range = labels[i].split('min')[0].split('<')[-1]
            cumulative_labels.append(f'{time_range}min')
        elif i == len(labels) - 1:
            time_range = labels[i].split('min')[0].split('>')[-1]
            cumulative_labels.append(f'{time_range}+min')
        else:
            time_range = labels[i].split('min')[0].split('-')[-1]
            cumulative_labels.append(f'{time_range}min')
    return cumulative_labels


def get_outlier_bounds(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound,upper_bound


# Function to compute LOWESS smoothing
def lowess_smooth(x, y, frac=1):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(y, x, frac=frac)
    return smoothed[:, 0], smoothed[:, 1]

def display_wilcoxon_results(analysis, question_code, data, W, p_value, Z, effect_size,save = True):
    from tabulate import tabulate
    zero_proportion = round(len(data[f'{question_code}_diff'][data[f'{question_code}_diff'] == 0]) / len(data[f'{question_code}_diff']) * 100, 3)
    non_zero_proportion = round(100 - zero_proportion, 3)
    
    results = [
        [f"{analysis}-Wilcoxon_Stat", round(W, 3)],
        [f"{analysis}-Wilcoxon_p", round(p_value, 3)],
        [f"{analysis}-Wilcoxon_Z", round(Z, 3)],
        [f"{analysis}-ES", round(effect_size, 3)],
        [f"{analysis}-Zero_Diff", f"{zero_proportion}%"],
        [f"{analysis}-Non-Zero_Diff", f"{non_zero_proportion}%"] ]
    
    desc_stats = []
    for stat in ["Mean", "Median", "Std", "Min", "Max"]:
        for test in ["pre", "post", "diff"]:
            value = getattr(data[f"{question_code}_{test.split('-')[0]}"], stat.lower())()
            desc_stats.append([f"{analysis}-{test}-{stat}", round(value, 3)])
    
    df_results = pd.DataFrame(results, columns=["Metric", "Value"])
    df_stats = pd.DataFrame(desc_stats, columns=["Metric", "Value"])
    
    print(tabulate(df_results, headers='keys', tablefmt='grid'))
    print(tabulate(df_stats, headers='keys', tablefmt='grid'))
    if save:
        df_combined = pd.concat([df_results, df_stats], ignore_index=True)
    df_combined.to_csv(f"{data_reports_path}", index=False, header=False, sep=' ')
    