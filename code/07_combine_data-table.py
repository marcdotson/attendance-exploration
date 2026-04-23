# This script combines modeling and exploratory data into CSV files for analysis.
# It outputs both full-history and post-COVID datasets.
# Full-history outputs are written to both legacy and canonical all-grades file names.

import pandas as pd
import pickle

# Define the years to process
years = [2017, 2018, 2022, 2023, 2024, 2025]

# Post Covid years
years_temp = pd.Series(years)
post_covid_years = years_temp[years_temp >= 2022].tolist()

######################################################################################################################################################
# ----------------------------------
# PART 1: Phase One Combine Data
# ----------------------------------

# Load in the modeling datasets that will be joined later
# []_model represents modeling files
academic_model = pd.read_csv('data/02_academic_modeling.csv')
demographic_model = pd.read_csv('data/03_demographic_modeling.csv')
assessment_model = pd.read_csv('data/04_assessment_data.csv')
# teacher_model = pd.read_csv('data/05_teacher_modeling_data.csv')
school_model = pd.read_csv('data/06_school_modeling_data.csv')

# Load in the exploratory datasets that will be joined later
# []_df represents exploratory files
academic_df = pd.read_csv('data/02_academic_exploratory.csv')
demographic_df = pd.read_csv('data/03_demographic_exploratory.csv')
assessment_df = pd.read_csv('data/04_assessment_data.csv')
# teacher_df = pd.read_csv('data/05_teacher_exploratory_data.csv')
school_df = pd.read_csv('data/06_school_exploratory_data.csv')

# Load the pickled data (student_tables)
with open('./data/student_data.pkl', 'rb') as f:
    student_tables = pickle.load(f)

# Function to process student data for given years (so post_covid years and all years can be processed at the same time)
def process_student_data(years, prefix="", write_all_grades_alias=False):
    """
    Processes student data for the specified years and exports both exploratory and modeling datasets.
    Parameters:
    - years (list): The list of years to process.
    - prefix (str): Optional prefix for output file names (e.g., "post_covid_").
    - write_all_grades_alias (bool): If True, also export to *_all_grades file names.
    """

    all_students = []

    for year in years:
        student_table = student_tables[year]
        student_table['year'] = year
        all_students.append(student_table[['student_number', 'year']])

    df = pd.concat(all_students, ignore_index=True).drop_duplicates(subset=['student_number', 'year'], keep='first')
    model_df = df.copy()
    model_df = model_df.drop(columns='year')
    model_df = model_df.drop_duplicates(keep='first')

    # Merge model_df with all modeling datasets
    model_df = pd.merge(model_df, academic_model, on='student_number', how='left')
    model_df = pd.merge(model_df, demographic_model, on='student_number', how='left')
    model_df = pd.merge(model_df, assessment_model, on='student_number', how='left')
    # model_df = pd.merge(model_df, teacher_model, on='student_number', how='left')
    model_df = pd.merge(model_df, school_model, on='student_number', how='left')

    # Merge df with all exploratory datasets
    df = pd.merge(df, academic_df, on=['student_number', 'year'], how='left')
    df = pd.merge(df, demographic_df, on=['student_number', 'year'], how='left')
    df = pd.merge(df, assessment_df, on=['student_number'], how='left')
    # df = pd.merge(df, teacher_df, on=['student_number', 'year'], how='left')
    df = pd.merge(df, school_df, on=['student_number', 'year'], how='left')

    df = df.drop_duplicates(keep='first')
    df.to_csv(f'./data/{prefix}exploratory_data.csv', index=False)
    model_df.to_csv(f'./data/{prefix}modeling_data.csv', index=False)

    # Keep canonical "all_grades" outputs in sync when processing full-history data.
    if write_all_grades_alias:
        df.to_csv(f'./data/{prefix}exploratory_data_all_grades.csv', index=False)
        model_df.to_csv(f'./data/{prefix}modeling_data_all_grades.csv', index=False)

# Process full historical data
process_student_data(years, prefix="", write_all_grades_alias=True)

# Process post-COVID data
process_student_data(post_covid_years, prefix="post_covid_")

print('===========================================')
print('Data exported successfully!')
print("Part one workflow is complete!")
print('===========================================')

