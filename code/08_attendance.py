from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "exploratory_data_all_grades.csv"
FIG_DIR   = ROOT / "figures"

GOOD_ATTENDANCE_PCT_MIN     = (170 / 180) * 100
GOOD_ATTENDANCE_VLINE_LABEL = "Good attendance ref.: >170/180 days, membership 180"
MIN_N              = 30
MIN_MEMBERSHIP     = 60
CHRONIC_ABSENT_PCT = 10.0

_hs_path = ROOT / "data" / "hs_teacher_data.csv"
_ms_path = ROOT / "data" / "ms_teacher_data.csv"
_es_path = ROOT / "data" / "es_teacher_data.csv"
_tg_path = ROOT / "data" / "all_teachers_grid_by_year.csv"

# ── Load & coerce ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
for _col in ["percent_days_attended", "school_membership", "overall_gpa", "composite_score",
             "current_grade", "days_absent", "scram_membership", "year"]:
    if _col in df.columns:
        df[_col] = pd.to_numeric(df[_col], errors="coerce")
df["student_number"] = df["student_number"].astype(str)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared bases ───────────────────────────────────────────────────────────────

# Chronic / school charts (3, 4, 6, 7)
_cbase = df[(df["school_membership"] >= MIN_MEMBERSHIP) & df["current_grade"].between(0, 12)].copy()
_cbase["_grade"] = _cbase["current_grade"].astype(int)
_cbase["_pct_absent"] = 100.0 * _cbase["days_absent"].fillna(0) / _cbase["school_membership"]
_cbase["_chronic"] = _cbase["_pct_absent"] >= CHRONIC_ABSENT_PCT
_cbase["_school"] = (_cbase["current_school"].astype(str).str.strip()
                     .replace({"nan": np.nan, "None": np.nan, "": np.nan}))

# GPA subset of _cbase (Charts 4, 7)
_cbase_gpa = _cbase[_cbase["overall_gpa"].notna() & (_cbase["_grade"] >= 7)].copy()

# Demographics chart (5)
_dbase = df[df["school_membership"] > 119].copy()
_dbase["_pct_absent"] = 100.0 * _dbase["days_absent"].fillna(0) / _dbase["school_membership"]
_dbase["_chronic"] = _dbase["_pct_absent"] >= CHRONIC_ABSENT_PCT
_dbase["_disability_group"] = np.where(
    _dbase["scram_membership"].fillna(0) > 0, "Disability", "No disability"
)

# Improver charts base (9–15) — filtered to match 09_attendance_q2.py
_base_imp = df[
    (df["school_membership"] >= MIN_MEMBERSHIP) &
    df["year"].notna() &
    df["percent_days_attended"].notna()
].copy()
if "exit_code" in _base_imp.columns:
    _exit_num = pd.to_numeric(_base_imp["exit_code"], errors="coerce")
    _base_imp = _base_imp[_exit_num.isna() | (_exit_num == 0)].copy()
_base_imp = _base_imp[_base_imp["current_school"].astype(str).str.strip().str.lower() != "other"].copy()
_base_imp = _base_imp[_base_imp["year"] >= 2022].copy()
_base_imp["current_school"] = _base_imp["current_school"].astype(str).str.strip()
_base_imp = _base_imp[
    (_base_imp["current_school"] != "") & (_base_imp["current_school"] != "nan")
].copy()

_poor_rows = _base_imp[_base_imp["percent_days_attended"] < 90.0].copy()

# Fix: first good year after first poor year (not global min good year)
_first_poor_yr = _poor_rows.groupby("student_number")["year"].min().rename("first_poor_year")
_good_rows = (_base_imp[_base_imp["percent_days_attended"] >= 95.0][["student_number", "year"]]
              .join(_first_poor_yr, on="student_number"))
_good_rows = _good_rows[_good_rows["year"] > _good_rows["first_poor_year"]]
_first_good_yr = _good_rows.groupby("student_number")["year"].min().rename("first_good_year")
_improvers = pd.concat([_first_poor_yr, _first_good_yr], axis=1).dropna()

# Anchor at first_good_year - 1 (the year the student was transitioning out of poor attendance)
_anchor = _improvers.reset_index()[["student_number", "first_good_year"]].copy()
_anchor["anchor_year"] = _anchor["first_good_year"] - 1

_sg_lookup = (_base_imp[["student_number", "year", "current_school", "current_grade"]]
              .drop_duplicates(subset=["student_number", "year"], keep="first"))
_trans_anchor = (_anchor
    .merge(_sg_lookup, left_on=["student_number", "anchor_year"], right_on=["student_number", "year"], how="inner")
    .drop_duplicates("student_number", keep="first"))

_anc_grade  = _trans_anchor.dropna(subset=["current_grade"])
_anc_school = _trans_anchor.dropna(subset=["current_school"])

# Denominator: poor rows in anchor years only (matches 09_attendance_q2.py poor_rows_for_anchor)
_poor_rows_anchor = (_poor_rows[_poor_rows["year"].isin(_anchor["anchor_year"].unique())]
                     .drop_duplicates(subset=["student_number", "year"], keep="first"))
_poor_rows_anchor_school = _poor_rows_anchor.dropna(subset=["current_school"])


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Overall attendance bins through the district
# ═══════════════════════════════════════════════════════════════════════════════

_base1 = df[(df["school_membership"] > 119) & df["percent_days_attended"].between(0, 100)].copy()
_bins1 = np.arange(0, 105, 5)
_labels1 = [f"{i}–{i+5}" for i in range(0, 100, 5)]
_base1["att_bucket"] = pd.cut(_base1["percent_days_attended"], bins=_bins1, labels=_labels1, include_lowest=True, right=True)
_counts1 = _base1.groupby("att_bucket", observed=True).size().reset_index(name="n")

fig, ax = plt.subplots(figsize=(11, 5))
sns.set_theme(style="whitegrid")
sns.barplot(data=_counts1, x="att_bucket", y="n", color="#4c72b0", ax=ax)
ax.set_xlabel("Percent days attended (5-point buckets)")
ax.set_ylabel("Student-years")
ax.set_title("Student Count by Attendance Bucket")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(FIG_DIR / "eda_q1_attendance_pct_histogram_5pt.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — GPA hinge: binned mean GPA vs attendance (overall + subgroups)
# ═══════════════════════════════════════════════════════════════════════════════

_base_gpa = df[
    (df["school_membership"] > 119) &
    df["overall_gpa"].notna() & (df["overall_gpa"] > 0) &
    (df["current_grade"] >= 7)
].copy()
_base_gpa = _base_gpa[_base_gpa["percent_days_attended"].between(50, 100)]
_bins_gpa = np.linspace(_base_gpa["percent_days_attended"].min(), 100, 21)
_base_gpa["att_bin"] = pd.cut(_base_gpa["percent_days_attended"], bins=_bins_gpa, include_lowest=True)
_agg_gpa = (
    _base_gpa.groupby("att_bin", observed=True)
    .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"), n=("student_number", "count"))
    .reset_index()
)

_base_act = df[(df["school_membership"] > 119) & (df["composite_score"] > 1)].copy()
_base_act = _base_act[_base_act["percent_days_attended"].between(50, 100)]
_bins_act = np.linspace(_base_act["percent_days_attended"].min(), 100, 17)
_base_act["att_bin"] = pd.cut(_base_act["percent_days_attended"], bins=_bins_act, include_lowest=True)
_agg_act = (
    _base_act.groupby("att_bin", observed=True)
    .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"))
    .reset_index()
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.set_theme(style="whitegrid")

ax = axes[0, 0]
ax.plot(_agg_gpa["mid_att"], _agg_gpa["mean_gpa"], marker="o", color="#1f77b4")
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8, label=GOOD_ATTENDANCE_VLINE_LABEL)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean cumulative GPA")
ax.set_title("A. Mean GPA by attendance (20 equal-width bins)")
ax.legend(loc="lower right", fontsize=8)

ax = axes[0, 1]
_grades = sorted(_base_gpa["current_grade"].dropna().unique().astype(int).tolist())
_grades = [g for g in _grades if 1 <= g <= 12]
for _g in _grades[:8]:
    _sub = _base_gpa[_base_gpa["current_grade"] == _g].copy()
    if len(_sub) < 200:
        continue
    _sub["att_bin"] = pd.cut(_sub["percent_days_attended"], bins=_bins_gpa, include_lowest=True)
    _gagg = (
        _sub.groupby("att_bin", observed=True)
        .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
        .reset_index()
    )
    ax.plot(_gagg["mid_att"], _gagg["mean_gpa"], marker=".", label=f"Grade {_g}")
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean GPA")
ax.set_title("B. Mean GPA vs attendance by grade")
ax.legend(title="Grade", fontsize=7, ncol=2)

ax = axes[1, 0]
if "gender" in _base_gpa.columns:
    _gender_vals = _base_gpa["gender"].astype(str).str.strip().str.upper()
    for _gend in ["M", "F"]:
        _sub = _base_gpa[_gender_vals == _gend].copy()
        if len(_sub) < 300:
            continue
        _sub["att_bin"] = pd.cut(_sub["percent_days_attended"], bins=_bins_gpa, include_lowest=True)
        _gagg = (
            _sub.groupby("att_bin", observed=True)
            .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_gagg["mid_att"], _gagg["mean_gpa"], marker=".", label=_gend)
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean GPA")
ax.set_title("C. Mean GPA vs attendance by gender")
ax.legend(fontsize=8)

ax = axes[1, 1]
if len(_base_act) < 500:
    ax.text(0.5, 0.5, "Insufficient data\n(n < 500 test-takers)", ha="center", va="center", transform=ax.transAxes)
else:
    ax.plot(_agg_act["mid_att"], _agg_act["mean_act"], marker="o", color="#2ca02c")
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean ACT composite")
ax.set_title("D. Mean ACT composite vs attendance (test-takers only)")

fig.suptitle(
    "Relationship Between Attendance and GPA/ACT\n"
    "Gray line = good attendance ref. (>170/180, membership 180).",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "narrative_act2a_gpa_hinge.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Chronic absenteeism rate by grade
# ═══════════════════════════════════════════════════════════════════════════════

_agg3 = (
    _cbase.groupby("_grade", dropna=True)
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_agg3 = _agg3[_agg3["n"] >= MIN_N]
_agg3["chronic_rate"] = _agg3["n_chronic"] / _agg3["n"]

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_theme(style="whitegrid")
_bars = ax.bar(_agg3["_grade"], _agg3["chronic_rate"], color="#c44e52")
for _bar, _row in zip(_bars, _agg3.itertuples()):
    ax.text(
        _bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.003,
        f"n={_row.n:,}", ha="center", fontsize=7, color="#555555"
    )
ax.set_xlabel("Grade")
ax.set_ylabel("Share of eligible student-years chronically absent")
ax.set_title(
    "Chronic absenteeism rate by grade\n"
    f"(eligible = school_membership ≥ {MIN_MEMBERSHIP}; chronic = ≥{CHRONIC_ABSENT_PCT:.0f}% days missed)",
    fontsize=11,
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xticks(_agg3["_grade"])
fig.tight_layout()
fig.savefig(FIG_DIR / "grade_chronic_rate_by_grade.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Mean GPA: chronic vs non-chronic by grade (7–12)
# ═══════════════════════════════════════════════════════════════════════════════

_rows4 = []
for _g, _grp in _cbase_gpa.groupby("_grade", dropna=True):
    _n_chr = int(_grp["_chronic"].sum())
    _n_not = int((~_grp["_chronic"]).sum())
    if _n_chr < MIN_N or _n_not < MIN_N:
        continue
    _rows4.append({
        "grade":       _g,
        "gpa_chronic": _grp.loc[_grp["_chronic"], "overall_gpa"].mean(),
        "gpa_not":     _grp.loc[~_grp["_chronic"], "overall_gpa"].mean(),
    })

if _rows4:
    _gpa_df4 = pd.DataFrame(_rows4).sort_values("grade")
    _x = np.arange(len(_gpa_df4))
    _w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    ax.bar(_x - _w / 2, _gpa_df4["gpa_not"],    _w, color="#4c72b0", label="Not chronically absent")
    ax.bar(_x + _w / 2, _gpa_df4["gpa_chronic"], _w, color="#c44e52", label="Chronically absent")
    ax.set_xticks(_x)
    ax.set_xticklabels(_gpa_df4["grade"].astype(int))
    ax.set_xlabel("Grade")
    ax.set_ylabel("Mean cumulative GPA")
    ax.set_title(
        "Mean GPA by grade (7–12): chronically absent vs not\n"
        f"(suppressed where n < {MIN_N} in either group)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "grade_mean_gpa_chronic_vs_not.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Chronic absenteeism by demographics
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
sns.set_theme(style="whitegrid")

ax = axes[0]
_pos = {"1", "y", "yes", "true", "t"}
_eth_dummy_cols = ["amerindian_alaskan", "asian", "black_african_amer", "hawaiian_pacific_isl", "white"]
_eth_rates = {}
for _col in _eth_dummy_cols:
    if _col in _dbase.columns:
        _mask = _dbase[_col].astype(str).str.strip().str.lower().isin(_pos)
        if _mask.sum() >= MIN_N:
            _eth_rates[_col] = _dbase.loc[_mask, "_chronic"].mean()
if "ethnicity" in _dbase.columns:
    _lat_mask = _dbase["ethnicity"].astype(str).str.strip().str.lower().isin(_pos)
    if _lat_mask.sum() >= MIN_N:
        _eth_rates["latinx"] = _dbase.loc[_lat_mask, "_chronic"].mean()
if _eth_rates:
    _eth_series = pd.Series(_eth_rates).sort_values()
    ax.barh(_eth_series.index, _eth_series.values, color="#4c72b0")
    ax.set_xlabel("Share chronically absent")
    ax.set_title("A. Chronic absenteeism rate by ethnicity")

ax = axes[1]
if "gender" in _dbase.columns:
    _gvals = _dbase["gender"].astype(str).str.strip().str.upper()
    _glabels = {"M": "Male", "F": "Female"}
    _grps = [g for g in ["M", "F"] if (_gvals == g).any()]
    ax.bar([_glabels[g] for g in _grps], [_dbase.loc[_gvals == g, "_chronic"].mean() for g in _grps], color="#c44e52")
    ax.set_ylabel("Share chronically absent")
    ax.set_title("B. Chronic absenteeism rate by gender")

ax = axes[2]
_dis_rates = (
    _dbase.groupby("_disability_group", observed=True)["_chronic"]
    .mean()
    .reindex(["No disability", "Disability"])
    .dropna()
)
if not _dis_rates.empty:
    ax.bar(_dis_rates.index, _dis_rates.values, color=["#4c72b0", "#c44e52"])
    ax.set_ylabel("Share chronically absent")
    ax.set_title("C. Chronic absenteeism rate by disability status")

fig.suptitle("Chronic Absenteeism by Demographics", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_absenteeism_by_demographics.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Chronic absenteeism rate by school
# ═══════════════════════════════════════════════════════════════════════════════

_sagg = (
    _cbase[_cbase["_school"].notna()]
    .groupby("_school", dropna=True)
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_sagg["chronic_rate"] = _sagg["n_chronic"] / _sagg["n"]
_sagg = _sagg[_sagg["n"] >= MIN_N].sort_values("chronic_rate", ascending=True)

fig, ax = plt.subplots(figsize=(9, max(4, len(_sagg) * 0.45)))
sns.set_theme(style="whitegrid")
_sbars = ax.barh(_sagg["_school"], _sagg["chronic_rate"], color="#c44e52")
for _bar, _n in zip(_sbars, _sagg["n"]):
    ax.text(
        _bar.get_width() + 0.002, _bar.get_y() + _bar.get_height() / 2,
        f"n={_n:,}", va="center", fontsize=7, color="#555555"
    )
ax.set_xlabel("Share of eligible student-years chronically absent")
ax.set_title(
    "Chronic absenteeism rate by school\n"
    f"(eligible = school_membership ≥ {MIN_MEMBERSHIP}; chronic = ≥{CHRONIC_ABSENT_PCT:.0f}% days missed)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_rate_by_school.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote chronic_rate_by_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Mean GPA: chronic vs non-chronic by grade (7–12), by school
# ═══════════════════════════════════════════════════════════════════════════════

_rows7 = []
for _g, _grp in _cbase_gpa.groupby("_grade", dropna=True):
    _n_chr = int(_grp["_chronic"].sum())
    _n_not = int((~_grp["_chronic"]).sum())
    if _n_chr < MIN_N or _n_not < MIN_N:
        continue
    _rows7.append({
        "grade":       _g,
        "gpa_chronic": _grp.loc[_grp["_chronic"], "overall_gpa"].mean(),
        "gpa_not":     _grp.loc[~_grp["_chronic"], "overall_gpa"].mean(),
    })

if _rows7:
    _gpa_df7 = pd.DataFrame(_rows7).sort_values("grade")
    _x = np.arange(len(_gpa_df7))
    _w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    ax.bar(_x - _w / 2, _gpa_df7["gpa_not"],    _w, color="#4c72b0", label="Not chronically absent")
    ax.bar(_x + _w / 2, _gpa_df7["gpa_chronic"], _w, color="#c44e52", label="Chronically absent")
    ax.set_xticks(_x)
    ax.set_xticklabels(_gpa_df7["grade"].astype(int))
    ax.set_xlabel("Grade")
    ax.set_ylabel("Mean cumulative GPA")
    ax.set_title(
        "Mean GPA by grade (7–12): chronically absent vs not\n"
        f"(suppressed where n < {MIN_N} in either group)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "grade_mean_gpa_chronic_vs_not_by_school.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Long-term impact: total years chronically absent vs high school GPA
# ═══════════════════════════════════════════════════════════════════════════════

_df8 = df[df["school_membership"] >= 90].copy()
_df8["_is_chronic_yr"] = _df8["percent_days_attended"] < 90
_chronic_yrs8 = (
    _df8.groupby("student_number")["_is_chronic_yr"]
    .sum()
    .reset_index(name="years_chronic_total")
)
_df8 = _df8.merge(_chronic_yrs8, on="student_number")
_df8 = _df8[(_df8["current_grade"] >= 9) & _df8["overall_gpa"].notna()]

fig, ax = plt.subplots(figsize=(12, 6))
sns.set_theme(style="whitegrid")
sns.boxplot(data=_df8, x="years_chronic_total", y="overall_gpa", palette="viridis", ax=ax)
ax.set_xlabel("Total Years Student was Chronically Absent (<90%)")
ax.set_ylabel("Overall GPA")
ax.set_title("Q3: Long-term Impact - Total Years Chronically Absent vs. High School GPA", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "Chronic Absenteeism.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote Chronic Absenteeism.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Improvers by first good-attendance grade
# ═══════════════════════════════════════════════════════════════════════════════

_m9 = _improvers.reset_index().merge(
    _base_imp[["student_number", "year", "current_grade"]].dropna(subset=["current_grade"]),
    on="student_number",
)
_m9 = _m9[(_m9["year"] == _m9["first_good_year"]) & _m9["current_grade"].between(1, 12)]
_m9 = _m9.drop_duplicates("student_number", keep="first")
_counts9 = _m9.groupby("current_grade").size().reset_index(name="n").sort_values("current_grade")

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_theme(style="whitegrid")
sns.barplot(data=_counts9, x="current_grade", y="n", color="#4c72b0", ax=ax)
ax.set_xlabel("First Good-Attendance Grade")
ax.set_ylabel("Total Students")
ax.set_title("Improvers by First Good-Attendance Grade")
fig.tight_layout()
fig.savefig(FIG_DIR / "improvers_count_by_first_good_attendance_grade.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote improvers_count_by_first_good_attendance_grade.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 10 — Percentage of improvers by grade
# ═══════════════════════════════════════════════════════════════════════════════

_anc10  = _anc_grade[_anc_grade["current_grade"].between(1, 12)]
_poor10 = _poor_rows_anchor[_poor_rows_anchor["current_grade"].between(1, 12)]

_n_trans10 = _anc10.groupby("current_grade")["student_number"].nunique().rename("n_trans")
_n_poor10  = _poor10.groupby("current_grade")["student_number"].nunique().rename("n_poor")
_grade_counts = pd.concat([_n_trans10, _n_poor10], axis=1).fillna({"n_trans": 0, "n_poor": 0}).reset_index()
_grade_counts = _grade_counts.rename(columns={"current_grade": "grade"})
_grade_counts = _grade_counts[_grade_counts["n_poor"] > 0].copy()
_grade_counts["n"] = _grade_counts["n_trans"] / _grade_counts["n_poor"]
_grade_counts = _grade_counts.sort_values("grade")

if not _grade_counts.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=_grade_counts, x="grade", y="n", color="#8172b3", ax=ax)
    ax.set_xlabel("Grade (at Last Poor-Attending Year)")
    ax.set_ylabel("Improvers / All Poor Attenders")
    ax.set_title("Percentage of Improvers by Grade")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "percentage_of_improvers_by_grade.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote percentage_of_improvers_by_grade.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 11 — Improvers by school
# ═══════════════════════════════════════════════════════════════════════════════

_improver_school_counts = (
    _anc_school.groupby("current_school", as_index=False).size()
    .rename(columns={"size": "n"})
    .sort_values("n", ascending=False)
)
if not _improver_school_counts.empty:
    fig, ax = plt.subplots(figsize=(max(10.0, 0.35 * len(_improver_school_counts)), 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=_improver_school_counts, x="current_school", y="n", color="#4c72b0", ax=ax)
    ax.set_xlabel("School (at Last Poor-Attending Year)")
    ax.set_ylabel("Total Students")
    ax.set_title("Improvers by School")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "improvers_count_by_school.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote improvers_count_by_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 12 — Percentage of improvers by school
# ═══════════════════════════════════════════════════════════════════════════════

_n_trans12 = _anc_school.groupby("current_school")["student_number"].nunique().rename("n_trans")
_n_poor12  = _poor_rows_anchor_school.groupby("current_school")["student_number"].nunique().rename("n_poor")
_school_counts = pd.concat([_n_trans12, _n_poor12], axis=1).fillna({"n_trans": 0, "n_poor": 0}).reset_index()
_school_counts = _school_counts[_school_counts["n_poor"] > 0].copy()
_school_counts["n"] = _school_counts["n_trans"] / _school_counts["n_poor"]
_school_counts = _school_counts.sort_values("n", ascending=False)

if not _school_counts.empty:
    fig, ax = plt.subplots(figsize=(max(10.0, 0.38 * len(_school_counts)), 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=_school_counts, x="current_school", y="n", color="#4c72b0", ax=ax)
    ax.set_xlabel("School (at Last Poor-Attending Year)")
    ax.set_ylabel("Improvers / All Poor Attenders")
    ax.set_title("Percentage of Improvers by School")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "percentage_of_improvers_by_school.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote percentage_of_improvers_by_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 13 — Percentage of improvers by school group
# ═══════════════════════════════════════════════════════════════════════════════

_hs_names = {str(i).strip() for i in pd.read_csv(_hs_path, index_col=0).index} if _hs_path.is_file() else set()
_ms_names = {str(i).strip() for i in pd.read_csv(_ms_path, index_col=0).index} if _ms_path.is_file() else set()
_es_names = {str(i).strip() for i in pd.read_csv(_es_path, index_col=0).index} if _es_path.is_file() else set()

def _group_label(school_name: object) -> str:
    s = str(school_name).strip()
    sl = s.lower()
    if s in _es_names or "elem." in sl:
        return "Elementary"
    if s in _ms_names or "middle" in sl:
        return "Middle school"
    return "High school"

_n_trans13 = (_anc_school.assign(school_group=_anc_school["current_school"].map(_group_label))
              .groupby("school_group")["student_number"].nunique().rename("n_trans"))
_n_poor13  = (_poor_rows_anchor_school.assign(school_group=_poor_rows_anchor_school["current_school"].map(_group_label))
              .groupby("school_group")["student_number"].nunique().rename("n_poor"))
_group_counts = pd.concat([_n_trans13, _n_poor13], axis=1).fillna({"n_trans": 0, "n_poor": 0}).reset_index()
_group_counts = _group_counts[_group_counts["n_poor"] > 0].copy()
_group_counts["n"] = _group_counts["n_trans"] / _group_counts["n_poor"]

if not _group_counts.empty:
    _group_counts["school_group"] = pd.Categorical(
        _group_counts["school_group"],
        categories=["Elementary", "Middle school", "High school"],
        ordered=True,
    )
    _group_counts = _group_counts.sort_values("school_group")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=_group_counts, x="school_group", y="n", color="#55a868", ax=ax)
    ax.set_xlabel("School Group (at Last Poor-Attending Year)")
    ax.set_ylabel("Improvers / All Poor Attenders")
    ax.set_title("Percentage of Improvers by School Group")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "percentage_of_improvers_by_school_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote percentage_of_improvers_by_school_group.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 14 — Top 30 teachers by percentage of improvers
# ═══════════════════════════════════════════════════════════════════════════════

_sy14 = (
    _base_imp.groupby(["student_number", "year"], as_index=False)["percent_days_attended"].mean()
    .rename(columns={"percent_days_attended": "pct"})
)
_poor14 = _sy14[_sy14["pct"] < 90.0][["student_number", "year"]].rename(columns={"year": "poor_year"})
_good14 = _sy14[_sy14["pct"] >= 95.0][["student_number", "year"]].rename(columns={"year": "good_year"})
_oyt = _poor14.merge(_good14, on="student_number", how="inner")
_oyt = _oyt[_oyt["good_year"] == _oyt["poor_year"] + 1].drop_duplicates(subset=["student_number", "poor_year"])

if _tg_path.is_file() and not _oyt.empty:
    _tgrid = pd.read_csv(_tg_path)
    _tcols = [c for c in _tgrid.columns if c.startswith("teacher_")]
    _tgrid["student_number"] = _tgrid["student_number"].astype(str)
    _tgrid["year"] = pd.to_numeric(_tgrid["year"], errors="coerce")
    _exp = _oyt.merge(_tgrid, left_on=["student_number", "poor_year"], right_on=["student_number", "year"], how="inner")
    if not _exp.empty and _tcols:
        _n_trans14 = _exp[_tcols].sum(axis=0).astype(int)
        _poor_sy14 = _base_imp[_base_imp["percent_days_attended"] < 90.0][["student_number", "year"]].drop_duplicates()
        _merged_poor14 = _tgrid[["student_number", "year"] + _tcols].merge(_poor_sy14, on=["student_number", "year"], how="inner")
        _n_poor14 = (_merged_poor14[_tcols] > 0).sum(axis=0).astype(int)
        _enrolled_sy14 = _base_imp[["student_number", "year"]].drop_duplicates()
        _merged_enrolled14 = _tgrid[["student_number", "year"] + _tcols].merge(_enrolled_sy14, on=["student_number", "year"], how="inner")
        _n_enrolled14 = (_merged_enrolled14[_tcols] > 0).sum(axis=0).astype(int)

        _tc = pd.DataFrame({
            "teacher":    _n_trans14.index.astype(str),
            "n_trans":    _n_trans14.values,
            "n_poor":     _n_poor14.reindex(_n_trans14.index).fillna(0).astype(int).values,
            "n_enrolled": _n_enrolled14.reindex(_n_trans14.index).fillna(0).astype(int).values,
        })
        _tc = _tc[_tc["n_trans"] > 0].copy()
        _tc["ratio"] = np.where(_tc["n_poor"] > 0, _tc["n_trans"] / _tc["n_poor"], np.nan)
        if not _tc.empty:
            if len(_tc) == 1:
                _pool = _tc.sort_values(["ratio", "n_trans"], ascending=[False, False])
            else:
                _avg_load = float(_tc["n_enrolled"].mean())
                _pool = _tc[_tc["n_enrolled"] > _avg_load].sort_values(["ratio", "n_trans"], ascending=[False, False])

            if not _pool.empty:
                _affiliation: dict[str, set[str]] = {}
                for _aff_path in (_hs_path, _ms_path, _es_path):
                    if not _aff_path.is_file():
                        continue
                    _aff_grid = pd.read_csv(_aff_path, index_col=0)
                    for _sch_name, _aff_row in _aff_grid.iterrows():
                        _school14 = str(_sch_name).strip()
                        for _col in _tcols:
                            if _col in _aff_grid.columns and pd.notna(_aff_row[_col]) and float(_aff_row[_col]) >= 1:
                                _affiliation.setdefault(_col, set()).add(_school14)

                _plot14 = _pool.head(30).iloc[::-1].copy()
                _plot14["ratio_plot"] = _plot14["ratio"].fillna(0.0)
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.set_theme(style="whitegrid")
                sns.barplot(data=_plot14, x="ratio_plot", y="teacher", color="#8172b3", ax=ax)
                ax.set_xlabel("Improvers / All Poor Attenders")
                ax.set_ylabel("Teacher")
                ax.set_title("Top 30 Teachers by Percentage of Improvers")
                _max_x = float(_plot14["ratio_plot"].max()) if not _plot14.empty else 0.0
                _x_pad = max(0.02, 0.04 * _max_x) if _max_x > 0 else 0.05
                ax.set_xlim(0, min(1.0, _max_x + _x_pad) if _max_x <= 1.0 else _max_x + _x_pad)
                for _patch, _row14 in zip(ax.patches, _plot14.itertuples(index=False)):
                    _schools14 = sorted(_affiliation.get(_row14.teacher, set()))
                    _school_text = ", ".join(_schools14) if _schools14 else "Unmapped"
                    _rtxt = f"{_row14.n_trans}/{_row14.n_poor}" if _row14.n_poor > 0 else f"{_row14.n_trans}/0"
                    _label = f"{_school_text}  ({_rtxt}, N={_row14.n_enrolled})"
                    _y_mid = _patch.get_y() + (_patch.get_height() / 2.0)
                    ax.text(_patch.get_width() + _x_pad, _y_mid, _label, va="center", ha="left", fontsize=8)
                fig.tight_layout()
                fig.savefig(FIG_DIR / "top_30_teachers_by_percentage_of_improvers.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                print("Wrote top_30_teachers_by_percentage_of_improvers.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 15 — Improvers count by transition length
# ═══════════════════════════════════════════════════════════════════════════════

_span_counts = (
    _improvers.assign(years_to_good=_improvers["first_good_year"] - _improvers["first_poor_year"])
    .groupby("years_to_good", as_index=False).size()
    .rename(columns={"size": "n"})
    .sort_values("years_to_good")
)
if not _span_counts.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=_span_counts, x="years_to_good", y="n", color="#55a868", ax=ax)
    ax.set_xlabel("Transition Length (Years)")
    ax.set_ylabel("Total Students")
    ax.set_title("Improvers Count by Transition Length (Years)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "improvers_count_by_transition_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote improvers_count_by_transition_length.png")

print(f"\nDone. Figures written to: {FIG_DIR}")
