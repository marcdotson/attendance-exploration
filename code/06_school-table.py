# The code will output two data files: 06_school_exploratory_data.csv and 06_school_modeling_data.csv
import pandas as pd
import pickle

# Define the years to process
years = [2017, 2018, 2022, 2023, 2024, 2025]

# Specify the columns and their corresponding data types for each dataset to speed up the concatenation
membership_columns = {'StudentNumber': 'int32', 'SchoolNumber': 'int16'}
date = ['CourseEntryDate']

# Load the pickled data (student_tables) — Student sheet: roster/grade, no school (school comes
# from Course Membership below).
with open('./data/student_data.pkl', 'rb') as f:
    student_tables = pickle.load(f)

# Concatenate all years of data for each dataset into single DataFrames
# This combines data from all years to create a complete record of the schools attended by students

# Initialize empty lists to store year-specific DataFrames
all_membership = []
all_student_tables = []

# Loop through each year and load the data and drop duplicates immediatly to clean the data and speed up the loop
for year in years:
    membership_year = pd.read_excel(
        f'data/{year} EOY Data - USU.xlsx', sheet_name='Course Membership', usecols=list(membership_columns.keys()) + date,
    dtype=membership_columns,
    parse_dates=date
        ).drop_duplicates()
    
    # Retrieve the data for the specified year from the student_tables dictionary
    student_table_year = student_tables[year]

    # Assign the year to student_table_year table
    # I will later use this to join with students current school, to make sure data is accurate on yearly basis
    student_table_year['year'] = year  
    membership_year['year'] = pd.to_datetime(membership_year['CourseEntryDate'], errors='coerce').dt.year

    # Ensure year is an integer
    student_table_year['year'] = student_table_year['year'].astype(int)
    membership_year['year'] = membership_year['year'].astype(int)

    # Roster: one row per student per year (school is not on the Student sheet in the pickle)
    all_membership.append(membership_year)
    all_student_tables.append(student_table_year[['student_number', 'year']])


######################################################################################################################################################
# Concatenate and clean the DataFrames for each dataset across all years
membership = pd.concat(all_membership, ignore_index=True)
student_table = pd.concat(all_student_tables, ignore_index=True)

# Rename column names in membership table
membership = membership.rename(columns={'StudentNumber': 'student_number', 'SchoolNumber': 'school_number'})

# Remove all duplicates from the dataframes after the data is concatenated
membership = membership.drop_duplicates(keep='first')
student_table = student_table.drop_duplicates(keep='first')

# Exploratory student-year index from roster. School per year is resolved from Course Membership
# later (see merge into current_schools and _pick_one_membership_row).
# - model_df: model data (one student_number column from roster rows, then merged)
df = student_table[['student_number', 'year']].copy()
model_df = student_table['student_number'].copy()

# Merge model_df and the membership table
student_school = pd.merge(model_df, membership, on='student_number', how='left')

# Drop useless columns from student_school
student_school = student_school.drop(columns=['CourseEntryDate', 'year'])
student_school.head()


######################################################################################################################################################
# Pivot the merged table to create a grid of student_numbers by school_number
school_grid = student_school.pivot_table(
    index='student_number',
    columns='school_number',
    values='school_number',  # Use school_number to create binary attendance indicators
    aggfunc=lambda x: 1,     # Set 1 to indicate attendance
    fill_value=0             # Fill missing values with 0 to indicate no attendance
)

# Rename the columns to make them more descriptive (i.e. school_[x])
school_grid.columns = [f'school_{int(col)}' for col in school_grid.columns]

# Reset the index to include student_number as a column for merging
school_grid = school_grid.reset_index()

school_grid.head()

# Merge school_grid with the model_df
model_df = pd.merge(model_df, school_grid, on='student_number', how='left')

model_df.head()


######################################################################################################################################################
# Define the school number to name mapping. This will be used later in the script
# Full school number to name mapping
school_name_map = {
    710: "Cache High", 706: "Sky View", 705: "Ridgeline", 703: "Green Canyon", 702: "Mountain Crest",
    410: "South Cache Middle", 406: "North Cache Middle", 330: "Spring Creek Middle", 170: "Wellesville Elem.", 166: "Sunrise Elem.",
    164: "Summit Elem.", 160: "River Heights Elem.", 156: "Providence Elem.", 152: "White Pine Elem.", 144: "North Park Elem.",
    140: "Nibley Elem.", 132: "Millville Elem.", 130: "Mountainside Elem.", 128: "Lincoln Elem.", 124: "Lewiston Elem.",
    120: "Heritage Elem.", 118: "Greenville Elem.", 111: "Cedar Ridge Elem.", 109: "Canyon Elem.", 106: "Birch Creek Elem."
}

# Level buckets (align with school_name_map). Used to pick one school when Course Membership has
# multiple SchoolNumbers in the same year (e.g. 6th grader with courses at both elem. and middle).
# Canonical elementary school numbers (aligned with school_name_map)
elementary_school_ids = {
    170, 166, 164, 160, 156, 152, 144, 140, 132, 130, 128, 124, 120, 118, 111, 109, 106,
}
middle_school_ids = [330, 406, 410]
high_school_ids = [710, 706, 705, 703, 702]

# Function to map school number to school name, handling unknowns and zeros
def map_school_name(x):
    # If school number is 0, return "None"
    if x == 0:
        return "None"
    # Try to get the name from the school_name_map dictionary
    # If the school number isn't found, return "Other"
    return school_name_map.get(x, "Other")


# Per-year grade (roster) — used to resolve multiple Course Membership rows (same student, same year)
grade_rows = []
for _y in years:
    _g = student_tables[_y][["student_number", "GradeLevel"]].copy()
    _g = _g.rename(columns={"GradeLevel": "current_grade"})
    _g["year"] = _y
    _g["current_grade"] = pd.to_numeric(_g["current_grade"], errors="coerce")
    _g = _g.drop_duplicates(subset=["student_number", "year"], keep="first")
    grade_rows.append(_g)
grades_long = pd.concat(grade_rows, ignore_index=True)


def _pick_one_membership_row(gdf: pd.DataFrame) -> pd.Series:
    """One membership row per (student_number, year). If the EOY file lists multiple school_numbers
    for the same year, keep the one that matches roster grade: K-6 -> elementary, 7-8 -> middle,
    9-12 -> high; otherwise fall back to the latest CourseEntryDate. Without this, drop_duplicates
    (keep='first') picked an arbitrary row, often middle school for 6th graders in dual-list cases.
    """
    if len(gdf) == 1:
        return gdf.iloc[0]
    g = gdf["current_grade"].iloc[0]
    if pd.isna(g):
        return gdf.iloc[0]
    gi = int(g)
    if 0 <= gi <= 6:
        sub = gdf[gdf["school_number"].isin(elementary_school_ids)]
    elif 7 <= gi <= 8:
        sub = gdf[gdf["school_number"].isin(middle_school_ids)]
    elif 9 <= gi <= 12:
        sub = gdf[gdf["school_number"].isin(high_school_ids)]
    else:
        sub = gdf.iloc[0:0]
    if not sub.empty:
        gdf = sub
    return gdf.sort_values("CourseEntryDate", ascending=False, na_position="last").iloc[0]


######################################################################################################################################################
# Create columns current_school, schools_attended, and school_count for the df
df = df.drop_duplicates()

# Merge df with membership to get school info (can be >1 row per student-year)
current_schools = pd.merge(df, membership, on=['student_number', 'year'], how='left')
current_schools = pd.merge(current_schools, grades_long, on=['student_number', 'year'], how='left')

current_schools = pd.DataFrame(
    [
        _pick_one_membership_row(g)
        for _, g in current_schools.groupby(["student_number", "year"], sort=False)
    ]
).reset_index(drop=True)

# Sort to keep the most recent year per student
current_schools = current_schools.sort_values(by=['year'], ascending=False)

# Rename for clarity
current_schools = current_schools.rename(columns={'school_number': 'current_school'})

# Fill null current_school values with 0
current_schools['current_school'] = current_schools['current_school'].fillna(0)

# Apply the school name mapping to the current_school column
current_schools['current_school'] = current_schools['current_school'].apply(map_school_name)

# All distinct school names seen for this student (membership years) — used to impute null current_school
student_school_list = (
    current_schools.groupby("student_number", as_index=False)["current_school"]
    .agg(_all_schools=lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
)
current_schools = pd.merge(current_schools, student_school_list, on="student_number", how="left")

# Roster grade already on current_schools from the membership resolution merge

for idx in current_schools.index:
    cs = current_schools.at[idx, "current_school"]
    if cs != "None":
        continue
    slist = current_schools.at[idx, "_all_schools"]
    names = [str(x).strip() for x in slist if str(x).strip() and str(x).strip().lower() != "none"]
    if not names:
        continue
    g = current_schools.at[idx, "current_grade"]
    if pd.isna(g):
        continue
    g = int(g)
    picked = None
    if 0 <= g <= 6:
        cands = [n for n in names if "elem." in n.lower()]
        if cands:
            picked = min(cands)
    elif 7 <= g <= 8:
        cands = [n for n in names if "middle" in n.lower()]
        if cands:
            picked = min(cands)
    elif 9 <= g <= 12:
        cands = [n for n in names if "elem." not in n.lower() and "middle" not in n.lower()]
        if cands:
            picked = min(cands)
    if picked is not None:
        current_schools.at[idx, "current_school"] = picked

current_schools = current_schools.drop(columns=["_all_schools", "current_grade"])

# DataFrame: schools_attended and school_count (from imputed current_school)
school_history = (
    current_schools.groupby("student_number")["current_school"]
    .agg(schools_attended=lambda x: sorted(x.unique()))
    .reset_index()
)

# Create column school_count
school_history['school_count'] = school_history['schools_attended'].apply(len)

# Make sure student_number is a string
current_schools['student_number'] = current_schools['student_number'].astype(str)
school_history['student_number'] = school_history['student_number'].astype(str)
df['student_number'] = df['student_number'].astype(str)

# Drop useless columns from current_schools
current_schools = current_schools.drop(columns=['CourseEntryDate'])

# Merge student_history and current_schools
current_schools = pd.merge(current_schools, school_history, on='student_number', how='left')
# Merge with df
df = pd.merge(df, current_schools, on=['student_number', 'year'], how='left')

df.head()

######################################################################################################################################################
# Create two new columns: high_school and middle_school for the model_df
# - Both are assigned based on the most recent school a student attended
# (middle_school_ids / high_school_ids are defined with school_name_map)

middle_schools = membership.copy()

# Filter for middle schools
middle_schools = middle_schools[middle_schools['school_number'].isin(middle_school_ids)]

# Sort by year to get the most recent record per student
middle_schools = middle_schools.sort_values(by=['year'], ascending=False)

# Keep only the most recent middle school entry per student
middle_schools = middle_schools.drop_duplicates(subset=['student_number'], keep='first')

# Rename school_number to middle_school
middle_schools = middle_schools.rename(columns={'school_number': 'middle_school'})

# Drop extra columns that aren't needed after processing
middle_schools = middle_schools.drop(columns=['CourseEntryDate', 'year'])

# Merge with model_df
model_df = pd.merge(model_df, middle_schools, on='student_number', how='left')

#=======================================================================================
# Do the same process, now for high schools
high_schools = membership.copy()

# Filter for high schools only
high_schools = high_schools[high_schools['school_number'].isin(high_school_ids)]

# Sort by year to get the most recent record per student
high_schools = high_schools.sort_values(by=['year'], ascending=False)

# Drop duplicates to keep the most recent high school per student
high_schools = high_schools.drop_duplicates(subset=['student_number'], keep='first')

# Rename school_number to high_school
high_schools = high_schools.rename(columns={'school_number': 'high_school'})

# Drop unnecessary columns
high_schools = high_schools.drop(columns=['CourseEntryDate', 'year'])

# Merge with model_df
model_df = pd.merge(model_df, high_schools, on='student_number', how='left')

model_df = model_df.fillna(0)

# Map the values in the current_school, high_school and middle_school columns
model_df['high_school'] = model_df['high_school'].apply(map_school_name)
model_df['middle_school'] = model_df['middle_school'].apply(map_school_name)

# Replace "None" values with 0
model_df['middle_school'] = model_df['middle_school'].replace("None", 0)
model_df['high_school'] = model_df['high_school'].replace("None", 0)

model_df = model_df.drop_duplicates(keep='first')
model_df.head()

######################################################################################################################################################
# Drop the school_grid columns from model_df since they are no longer needed for export, but keep the original code above
model_df = model_df.drop(columns=[col for col in model_df.columns if col.startswith('school_')])

# Export the data
df.to_csv('./data/06_school_exploratory_data.csv', index=False)
model_df.to_csv('./data/06_school_modeling_data.csv', index=False)

print('===========================================')
print('School data exported successfully!')
print("Next, run: 07_combine_data-table.py")
print('===========================================')
