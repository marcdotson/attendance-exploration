import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load dataset
df = pd.read_csv('C:/Users/emmar/Git Repositories/attendance-exploration-1/data/exploratory_data_all_grades.csv')

# calculate the attendance rate using the dictionary's logic
df['attendance_rate'] = (df['days_attended'] / df['school_membership']) * 100

# group by year 
yearly_stats = df.groupby('year')['attendance_rate'].mean().reset_index()

# plotting
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

sns.lineplot(data=yearly_stats, x='year', y='attendance_rate', marker='o', color='#1a73e8', linewidth=2.5)

# adding a marker for the COVID-19 period
plt.axvspan(2019, 2021, color='lightgray', alpha=0.3, label='COVID-19 Disruption')

plt.title('Cache County: Longitudinal Attendance Trend', fontsize=16)
plt.ylabel('Average Attendance Rate (%)')
plt.xlabel('Academic Year')
plt.ylim(85, 100) 
plt.legend()

plt.savefig('attendance_time_series.png')



# --- DATA PREP ---
# 1. Filter for 'Stable' students (Enrollment > 90 days) to match your previous logic
# This ensures the attendance rate is statistically meaningful
df = df[df['school_membership'] >= 90].copy()

# 2. Create an Attendance 'Bin' for readability (grouping every 2.5%)
# This prevents overplotting and shows the 'Hinge Point' clearly
df['att_bin'] = (df['percent_days_attended'] // 2.5) * 2.5

# 3. Q3 Trend Prep: Identify Persistent Chronic Absence
# We flag students who were under 90% attendance in a given year
df['is_chronic'] = df['percent_days_attended'] < 90
# Group by student to see how many years they've been chronic
chronic_counts = df.groupby('student_number')['is_chronic'].sum().reset_index()
chronic_counts.columns = ['student_number', 'years_chronic_total']
df = df.merge(chronic_counts, on='student_number')

# --- VISUALIZATION 1: Q2 - The 'Mobility' Driver ---
# Does changing schools make you more 'fragile' to absences?
plt.figure(figsize=(12, 6))
# We limit to school_count 1, 2, and 3+ for clarity
df['mobility_group'] = df['school_count'].apply(lambda x: '1 School' if x == 1 else ('2 Schools' if x == 2 else '3+ Schools'))

sns.lineplot(data=df, x='att_bin', y='overall_gpa', hue='mobility_group', 
             marker='o', err_style='band')

plt.title('Q2: Does School Mobility Change the Impact of Absences?', fontsize=14)
plt.xlabel('Attendance Rate (2.5% Bins)')
plt.ylabel('Average GPA')
plt.xlim(70, 100) # Zooming in on the 70-100% range where the action is
plt.grid(True, alpha=0.2)
plt.savefig('q2_mobility_driver.png')
plt.show()

# --- VISUALIZATION 2: Q3 - The 'Persistence' Trend ---
# Does one bad year of attendance hurt as much as three?
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df['current_grade'] >= 9], x='years_chronic_total', y='overall_gpa', palette='viridis')

plt.title('Q3: Long-term Impact - Total Years Chronically Absent vs. High School GPA', fontsize=14)
plt.xlabel('Total Years Student was Chronically Absent (<90%)')
plt.ylabel('Overall GPA')
plt.savefig('q3_persistence_trend.png')
plt.show()

# --- PREP FOR POWER BI ---
# We save this enriched file so Power BI already has the 'years_chronic' and 'att_bin' columns
# df.to_csv('power_bi_ready_data.csv', index=False)
# print("Files saved: q2_mobility_driver.png, q3_persistence_trend.png, and power_bi_ready_data.csv")

# Scale the scores so they can be compared on one chart
# ACT is out of 36, GPA is out of 4.0. We normalize for the visual.
df['norm_gpa'] = df['overall_gpa'] / 4.0
df['norm_act'] = df['composite_score'] / 36.0

# Melt the data for a comparison plot
melted = df.melt(id_vars=['att_bin'], value_vars=['norm_gpa', 'norm_act'])

plt.figure(figsize=(10, 6))
sns.lineplot(data=melted, x='att_bin', y='value', hue='variable')
plt.title('Q4: Is Test Performance More Sensitive to Absences than GPA?')
plt.ylabel('Normalized Score (0 to 1)')
plt.show()