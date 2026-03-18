# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Specify all years
years = [2017, 2018, 2022, 2023, 2024, 2025]

# Set up figures directory
_fig_dir = os.path.join('figures', 'eda')
try:
    os.makedirs(_fig_dir, exist_ok=True)
except Exception:
    pass

# Set warnings to ignore
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# USU color palette
navy = "#00274C"
gray = "#9EA2A2"
usu_colors = [navy, gray]
darker_red = "#8B0000"
blue = "#6693F5"

# Set global font to Montserrat and apply USU colors
plt.rcParams.update({
    "font.family": "Montserrat",  # Global font
    "axes.titlesize": 16,         # Title font size
    "axes.labelsize": 12,         # Axis label size
    "axes.labelcolor": navy,      # Axis label color
    "xtick.color": navy,          # X-tick color
    "ytick.color": navy,          # Y-tick color
    "text.color": navy            # General text color
})
sns.set_palette(usu_colors)  # Apply USU color palette to Seaborn globally
sns.set_style("whitegrid")  # Set Seaborn style

# Load the data from CSV files
data = pd.read_csv(os.path.join('data', 'exploratory_data.csv'))

# === Utility Functions ===
def print_section_header(title):
    """Utility function to print section headers."""
    print("\n" + title)
    print("=" * 50)


##########################################################################
# PHASE THREE - DLI (Dual Language Immersion) Analysis and Visualizations
# Compare DLI vs Non-DLI students across various academic metrics

def analyze_dli_students(data):
    """
    Analyze DLI students and create comprehensive visualizations comparing DLI vs Non-DLI students.
    """
    print_section_header("DLI Student Analysis")
    
    # Filter data to only include students with DLI indicator
    dli_data = data[data['dli_ind'].notna()].copy()
    
    if len(dli_data) == 0:
        print("No DLI data available for analysis.")
        return
    
    # Basic DLI statistics
    total_students = len(dli_data)
    dli_students = len(dli_data[dli_data['dli_ind'] == 1])
    non_dli_students = len(dli_data[dli_data['dli_ind'] == 0])
    
    print(f"Total students with DLI data: {total_students:,}")
    print(f"DLI students: {dli_students:,} ({dli_students/total_students*100:.1f}%)")
    print(f"Non-DLI students: {non_dli_students:,} ({non_dli_students/total_students*100:.1f}%)")
    
    # Create comprehensive DLI comparison visualizations
    create_dli_comparison_plots(dli_data)
    
    # College Enrollment Analysis for DLI vs Non-DLI
    analyze_dli_college_enrollment(dli_data)
    
    # ACT Score Analysis for DLI vs Non-DLI
    analyze_dli_act_scores(dli_data)

def create_dli_comparison_plots(data):
    """
    Create comparison plots for DLI vs Non-DLI students.
    Each plot will be in its own pane for better display.
    """
    print_section_header("DLI vs Non-DLI Academic Performance Comparison")
    
    # Plot 1: GPA Comparison
    dli_gpa = data[data['dli_ind'] == 1]['overall_gpa'].dropna()
    non_dli_gpa = data[data['dli_ind'] == 0]['overall_gpa'].dropna()
    
    if len(dli_gpa) > 0 and len(non_dli_gpa) > 0:
        plt.figure(figsize=(10, 6))
        gpa_data = [dli_gpa, non_dli_gpa]
        gpa_labels = ['DLI Students', 'Non-DLI Students']
        
        bp1 = plt.boxplot(gpa_data, labels=gpa_labels, patch_artist=True)
        bp1['boxes'][0].set_facecolor(blue)  # Use the new blue color for DLI
        bp1['boxes'][1].set_facecolor(usu_colors[1])  # Keep gray for Non-DLI
        # Make median lines black for better visibility
        bp1['medians'][0].set_color('black')
        bp1['medians'][1].set_color('black')
        
        plt.ylabel('Overall GPA', fontsize=12)
        plt.title('GPA Comparison: DLI vs Non-DLI Students', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add sample size annotations
        plt.text(0.02, 0.98, f'DLI: n={len(dli_gpa)}\nNon-DLI: n={len(non_dli_gpa)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(_fig_dir, 'DLI_vs_NonDLI_GPA_Comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot 2: Advanced Course Participation
    dli_ac = data[data['dli_ind'] == 1]['ac_ind'].mean() * 100
    non_dli_ac = data[data['dli_ind'] == 0]['ac_ind'].mean() * 100
    
    plt.figure(figsize=(10, 6))
    categories = ['DLI Students', 'Non-DLI Students']
    percentages = [dli_ac, non_dli_ac]
    colors = [blue, usu_colors[1]]  # Use new blue for DLI, gray for Non-DLI
    
    bars = plt.bar(categories, percentages, color=colors, alpha=0.8, edgecolor='black')
    plt.ylabel('Percentage Taking Advanced Courses', fontsize=12)
    plt.title('Advanced Course Participation: DLI vs Non-DLI Students', fontsize=16, fontweight='bold')
    plt.ylim(0, max(percentages) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(_fig_dir, 'DLI_vs_NonDLI_Advanced_Courses.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Extracurricular Participation
    dli_extracurricular = data[data['dli_ind'] == 1]['extracurricular_ind'].mean() * 100
    non_dli_extracurricular = data[data['dli_ind'] == 0]['extracurricular_ind'].mean() * 100
    
    plt.figure(figsize=(10, 6))
    extracurricular_percentages = [dli_extracurricular, non_dli_extracurricular]
    
    bars = plt.bar(categories, extracurricular_percentages, color=colors, alpha=0.8, edgecolor='black')
    plt.ylabel('Percentage in Extracurricular Activities', fontsize=12)
    plt.title('Extracurricular Participation: DLI vs Non-DLI Students', fontsize=16, fontweight='bold')
    plt.ylim(0, max(extracurricular_percentages) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, percentage in zip(bars, extracurricular_percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(_fig_dir, 'DLI_vs_NonDLI_Extracurricular_Participation.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Attendance Comparison
    dli_attendance = data[data['dli_ind'] == 1]['percent_days_attended'].dropna()
    non_dli_attendance = data[data['dli_ind'] == 0]['percent_days_attended'].dropna()
    
    if len(dli_attendance) > 0 and len(non_dli_attendance) > 0:
        plt.figure(figsize=(10, 6))
        attendance_data = [dli_attendance, non_dli_attendance]
        
        bp4 = plt.boxplot(attendance_data, labels=categories, patch_artist=True)
        bp4['boxes'][0].set_facecolor(blue)  # Use the new blue color for DLI
        bp4['boxes'][1].set_facecolor(usu_colors[1])  # Keep gray for Non-DLI
        # Make median lines black for better visibility
        bp4['medians'][0].set_color('black')
        bp4['medians'][1].set_color('black')
        
        plt.ylabel('Percent Days Attended', fontsize=12)
        plt.title('Attendance Comparison: DLI vs Non-DLI Students', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add sample size annotations
        plt.text(0.02, 0.98, f'DLI: n={len(dli_attendance)}\nNon-DLI: n={len(non_dli_attendance)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(_fig_dir, 'DLI_vs_NonDLI_Attendance_Comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Print summary statistics
    print("\n=== DLI vs Non-DLI Summary Statistics ===")
    print(f"\nGPA Statistics:")
    if len(dli_gpa) > 0:
        print(f"  DLI Students: n={len(dli_gpa)}, mean={dli_gpa.mean():.2f}, std={dli_gpa.std():.2f}")
    if len(non_dli_gpa) > 0:
        print(f"  Non-DLI Students: n={len(non_dli_gpa)}, mean={non_dli_gpa.mean():.2f}, std={non_dli_gpa.std():.2f}")
    
    print(f"\nAdvanced Course Participation:")
    print(f"  DLI Students: {dli_ac:.1f}%")
    print(f"  Non-DLI Students: {non_dli_ac:.1f}%")
    
    print(f"\nExtracurricular Participation:")
    print(f"  DLI Students: {dli_extracurricular:.1f}%")
    print(f"  Non-DLI Students: {non_dli_extracurricular:.1f}%")

def analyze_dli_college_enrollment(data):
    """
    Analyze college enrollment for DLI vs Non-DLI students (if data is available).
    """
    print_section_header("DLI vs Non-DLI College Enrollment Analysis")
    
    # Check if college enrollment data is available
    if 'start_college_y' not in data.columns:
        print("College enrollment data not available for analysis.")
        return
    
    # Filter out students without college enrollment data
    college_data = data.dropna(subset=['start_college_y'])
    
    if len(college_data) == 0:
        print("No college enrollment data available for analysis.")
        return
    
    # Calculate college enrollment percentages
    dli_college_count = college_data[college_data['dli_ind'] == 1]['start_college_y'].sum()
    dli_college_total = college_data[college_data['dli_ind'] == 1]['start_college_y'].count()
    dli_college_percentage = (dli_college_count / dli_college_total) * 100 if dli_college_total > 0 else 0
    
    non_dli_college_count = college_data[college_data['dli_ind'] == 0]['start_college_y'].sum()
    non_dli_college_total = college_data[college_data['dli_ind'] == 0]['start_college_y'].count()
    non_dli_college_percentage = (non_dli_college_count / non_dli_college_total) * 100 if non_dli_college_total > 0 else 0
    
    print(f"DLI Students enrolled in college: {dli_college_count}/{dli_college_total} ({dli_college_percentage:.1f}%)")
    print(f"Non-DLI Students enrolled in college: {non_dli_college_count}/{non_dli_college_total} ({non_dli_college_percentage:.1f}%)")
    
    # Create college enrollment visualization
    plt.figure(figsize=(10, 6))
    categories = ['DLI Students', 'Non-DLI Students']
    percentages = [dli_college_percentage, non_dli_college_percentage]
    counts = [dli_college_count, non_dli_college_count]
    total_counts = [dli_college_total, non_dli_college_total]
    
    bars = plt.bar(categories, percentages, color=[blue, usu_colors[1]], alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, percentage, count, total) in enumerate(zip(bars, percentages, counts, total_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{percentage:.1f}%\n({count}/{total})', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Percentage of Students Enrolled in College', fontsize=12)
    plt.title('College Enrollment: DLI vs Non-DLI Students', fontsize=16, fontweight='bold')
    plt.ylim(0, max(percentages) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add note about data source
    plt.figtext(0.5, 0.02, 'Note: Based on National Student Clearinghouse data (start_college_y = 1.0)', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(_fig_dir, 'DLI_vs_NonDLI_College_Enrollment.png'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_dli_act_scores(data):
    """
    Analyze ACT scores for DLI vs Non-DLI students.
    """
    print_section_header("DLI vs Non-DLI ACT Score Analysis")
    
    # Check if ACT score data is available (excluding writing_score as it's optional)
    act_columns = ['composite_score', 'english_score', 'math_score', 'reading_score', 'science_score']
    available_act_columns = [col for col in act_columns if col in data.columns]
    
    if not available_act_columns:
        print("ACT score data not available for analysis.")
        return
    
    # Filter out students without ACT scores (score = 0 indicates no test taken)
    act_data = data[data['composite_score'] > 0].copy()
    
    if len(act_data) == 0:
        print("No ACT score data available for analysis.")
        return
    
    # Basic ACT statistics
    dli_act_students = len(act_data[act_data['dli_ind'] == 1])
    non_dli_act_students = len(act_data[act_data['dli_ind'] == 0])
    
    print(f"Students with ACT scores: {len(act_data):,}")
    print(f"DLI students with ACT scores: {dli_act_students:,}")
    print(f"Non-DLI students with ACT scores: {non_dli_act_students:,}")
    
    # Create comprehensive ACT score visualizations
    create_dli_act_comparison_plots(act_data, available_act_columns)
    
    # Print detailed ACT statistics
    print_dli_act_statistics(act_data, available_act_columns)

def create_dli_act_comparison_plots(data, act_columns):
    """
    Create comprehensive ACT score comparison plots for DLI vs Non-DLI students.
    Each plot will be in its own pane for better display.
    """
    print_section_header("DLI vs Non-DLI ACT Score Comparison")
    
    # Create individual plots for each ACT component
    for score_type in act_columns:
        # Get data for DLI and Non-DLI students, filtering for valid ACT scores (0-36)
        dli_scores = data[data['dli_ind'] == 1][score_type].dropna()
        dli_scores = dli_scores[(dli_scores >= 0) & (dli_scores <= 36)]
        
        non_dli_scores = data[data['dli_ind'] == 0][score_type].dropna()
        non_dli_scores = non_dli_scores[(non_dli_scores >= 0) & (non_dli_scores <= 36)]
        
        if len(dli_scores) > 0 and len(non_dli_scores) > 0:
            # Create individual figure for each ACT component
            plt.figure(figsize=(10, 6))
            
            # Create boxplot
            score_data = [dli_scores, non_dli_scores]
            labels = ['DLI Students', 'Non-DLI Students']
            
            bp = plt.boxplot(score_data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor(blue)  # Use the new blue color for DLI
            bp['boxes'][1].set_facecolor(usu_colors[1])  # Keep gray for Non-DLI
            # Make median lines black for better visibility
            bp['medians'][0].set_color('black')
            bp['medians'][1].set_color('black')
            
            # Formatting
            score_name = score_type.replace('_score', '').replace('_', ' ').title()
            plt.ylabel(f'ACT {score_name} Score', fontsize=12)
            plt.title(f'ACT {score_name} Comparison: DLI vs Non-DLI Students', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add sample size annotations
            plt.text(0.02, 0.98, f'DLI: n={len(dli_scores)}\nNon-DLI: n={len(non_dli_scores)}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(_fig_dir, f'DLI_vs_NonDLI_ACT_{score_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print(f"No data available for {score_type.replace('_score', '').replace('_', ' ').title()}")

def print_dli_act_statistics(data, act_columns):
    """
    Print detailed ACT statistics for DLI vs Non-DLI students.
    """
    print_section_header("DLI vs Non-DLI ACT Score Statistics")
    
    for score_type in act_columns:
        print(f"\n{score_type.replace('_score', '').replace('_', ' ').title()} Statistics:")
        
        # DLI students - filter for valid ACT scores (0-36)
        dli_scores = data[data['dli_ind'] == 1][score_type].dropna()
        dli_scores = dli_scores[(dli_scores >= 0) & (dli_scores <= 36)]
        if len(dli_scores) > 0:
            print(f"  DLI Students: n={len(dli_scores)}, mean={dli_scores.mean():.2f}, std={dli_scores.std():.2f}, "
                  f"min={dli_scores.min():.0f}, max={dli_scores.max():.0f}")
        else:
            print(f"  DLI Students: No data available")
        
        # Non-DLI students - filter for valid ACT scores (0-36)
        non_dli_scores = data[data['dli_ind'] == 0][score_type].dropna()
        non_dli_scores = non_dli_scores[(non_dli_scores >= 0) & (non_dli_scores <= 36)]
        if len(non_dli_scores) > 0:
            print(f"  Non-DLI Students: n={len(non_dli_scores)}, mean={non_dli_scores.mean():.2f}, std={non_dli_scores.std():.2f}, "
                  f"min={non_dli_scores.min():.0f}, max={non_dli_scores.max():.0f}")
        else:
            print(f"  Non-DLI Students: No data available")
        
        # Calculate difference if both groups have data
        if len(dli_scores) > 0 and len(non_dli_scores) > 0:
            mean_diff = dli_scores.mean() - non_dli_scores.mean()
            print(f"  Mean Difference (DLI - Non-DLI): {mean_diff:+.2f}")

# Run the DLI analysis
analyze_dli_students(data)

