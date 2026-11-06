from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from plotnine import *

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "exploratory_data.csv"
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

# Chronic / school charts (4, 7, 8, 9, 10, 11, 12)
_cbase = df[(df["school_membership"] >= MIN_MEMBERSHIP) & df["current_grade"].between(0, 12)].copy()
_cbase["_grade"] = _cbase["current_grade"].astype(int)
_cbase["_pct_absent"] = 100.0 * _cbase["days_absent"].fillna(0) / _cbase["school_membership"]
_cbase["_chronic"] = _cbase["_pct_absent"] >= CHRONIC_ABSENT_PCT
_cbase["_school"] = (_cbase["current_school"].astype(str).str.strip()
                     .replace({"nan": np.nan, "None": np.nan, "": np.nan}))

# GPA subset of _cbase (Chart 13)
_cbase_gpa = _cbase[_cbase["overall_gpa"].notna() & (_cbase["_grade"] >= 7)].copy()

# Demographics charts (5, 6)
_dbase = df[df["school_membership"] > 119].copy()
_dbase["_pct_absent"] = 100.0 * _dbase["days_absent"].fillna(0) / _dbase["school_membership"]
_dbase["_chronic"] = _dbase["_pct_absent"] >= CHRONIC_ABSENT_PCT
_dbase["_disability_group"] = np.where(
    _dbase["scram_membership"].fillna(0) > 0, "Disability", "No disability"
)

# Improver charts base (17–22) — filtered to match 09_attendance_q2.py
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

# Anchor at first_good_year (the year the student achieved good attendance)
_anchor = _improvers.reset_index()[["student_number", "first_good_year"]].copy()
_anchor["anchor_year"] = _anchor["first_good_year"]

_sg_lookup = (_base_imp[["student_number", "year", "current_school", "current_grade"]]
              .drop_duplicates(subset=["student_number", "year"], keep="first"))
_trans_anchor = (_anchor
    .merge(_sg_lookup, left_on=["student_number", "anchor_year"], right_on=["student_number", "year"], how="inner")
    .drop_duplicates("student_number", keep="first"))

_anc_grade  = _trans_anchor.dropna(subset=["current_grade"])
_anc_school = _trans_anchor.dropna(subset=["current_school"])

# Denominator: all poor rows across all years (numerator is improvement year, so denominator spans full history)
_poor_rows_anchor = _poor_rows.drop_duplicates(subset=["student_number", "year"], keep="first")
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
fig.savefig(FIG_DIR / "attendance_distribution_pct_bucket_histogram.png", dpi=150, bbox_inches="tight")
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
_eth_cols_gpa = {
    "amerindian_alaskan": "Am. Indian/Alaska Native",
    "asian": "Asian",
    "black_african_amer": "Black/African American",
    "hawaiian_pacific_isl": "Hawaiian/Pacific Islander",
    "white": "White",
}
_pos_vals = {"1", "y", "yes", "true", "t"}
for _ecol, _elabel in _eth_cols_gpa.items():
    if _ecol in _base_gpa.columns:
        _emask = _base_gpa[_ecol].astype(str).str.strip().str.lower().isin(_pos_vals)
        _esub = _base_gpa[_emask].copy()
        if len(_esub) < MIN_N:
            continue
        _esub["att_bin"] = pd.cut(_esub["percent_days_attended"], bins=_bins_gpa, include_lowest=True)
        _egagg = (
            _esub.groupby("att_bin", observed=True)
            .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_egagg["mid_att"], _egagg["mean_gpa"], marker=".", label=_elabel)
if "ethnicity" in _base_gpa.columns:
    _lat_mask_gpa = _base_gpa["ethnicity"].astype(str).str.strip().str.lower().isin(_pos_vals)
    _lat_sub_gpa = _base_gpa[_lat_mask_gpa].copy()
    if len(_lat_sub_gpa) >= MIN_N:
        _lat_sub_gpa["att_bin"] = pd.cut(_lat_sub_gpa["percent_days_attended"], bins=_bins_gpa, include_lowest=True)
        _lgagg = (
            _lat_sub_gpa.groupby("att_bin", observed=True)
            .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_lgagg["mid_att"], _lgagg["mean_gpa"], marker=".", label="Latinx")
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean GPA")
ax.set_title("D. Mean GPA vs attendance by ethnicity")
ax.legend(title="Ethnicity", fontsize=7)

fig.suptitle(
    "Relationship Between Attendance and GPA\n"
    "Gray line = good attendance ref. (>170/180, membership 180).",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "attendance_gpa_hinge_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — ACT hinge: binned mean ACT vs attendance (overall + subgroups)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.set_theme(style="whitegrid")

ax = axes[0, 0]
if len(_base_act) < 500:
    ax.text(0.5, 0.5, "Insufficient data\n(n < 500 test-takers)", ha="center", va="center", transform=ax.transAxes)
else:
    ax.plot(_agg_act["mid_att"], _agg_act["mean_act"], marker="o", color="#2ca02c")
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8, label=GOOD_ATTENDANCE_VLINE_LABEL)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean ACT composite")
ax.set_title("A. Mean ACT by attendance (all test-takers)")
ax.legend(loc="lower right", fontsize=8)

ax = axes[0, 1]
_grades_act = sorted(_base_act["current_grade"].dropna().unique().astype(int).tolist())
_grades_act = [g for g in _grades_act if 1 <= g <= 12]
for _g in _grades_act:
    _sub = _base_act[_base_act["current_grade"] == _g].copy()
    if len(_sub) < 100:
        continue
    _sub["att_bin"] = pd.cut(_sub["percent_days_attended"], bins=_bins_act, include_lowest=True)
    _gagg = (
        _sub.groupby("att_bin", observed=True)
        .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"))
        .reset_index()
    )
    ax.plot(_gagg["mid_att"], _gagg["mean_act"], marker=".", label=f"Grade {_g}")
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean ACT composite")
ax.set_title("B. Mean ACT vs attendance by grade")
ax.legend(title="Grade", fontsize=7, ncol=2)

ax = axes[1, 0]
if "gender" in _base_act.columns:
    _gender_vals_act = _base_act["gender"].astype(str).str.strip().str.upper()
    for _gend in ["M", "F"]:
        _sub = _base_act[_gender_vals_act == _gend].copy()
        if len(_sub) < 100:
            continue
        _sub["att_bin"] = pd.cut(_sub["percent_days_attended"], bins=_bins_act, include_lowest=True)
        _gagg = (
            _sub.groupby("att_bin", observed=True)
            .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_gagg["mid_att"], _gagg["mean_act"], marker=".", label=_gend)
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean ACT composite")
ax.set_title("C. Mean ACT vs attendance by gender")
ax.legend(fontsize=8)

ax = axes[1, 1]
for _ecol, _elabel in _eth_cols_gpa.items():
    if _ecol in _base_act.columns:
        _emask = _base_act[_ecol].astype(str).str.strip().str.lower().isin(_pos_vals)
        _esub = _base_act[_emask].copy()
        if len(_esub) < MIN_N:
            continue
        _esub["att_bin"] = pd.cut(_esub["percent_days_attended"], bins=_bins_act, include_lowest=True)
        _egagg = (
            _esub.groupby("att_bin", observed=True)
            .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_egagg["mid_att"], _egagg["mean_act"], marker=".", label=_elabel)
if "ethnicity" in _base_act.columns:
    _lat_mask_act = _base_act["ethnicity"].astype(str).str.strip().str.lower().isin(_pos_vals)
    _lat_sub_act = _base_act[_lat_mask_act].copy()
    if len(_lat_sub_act) >= MIN_N:
        _lat_sub_act["att_bin"] = pd.cut(_lat_sub_act["percent_days_attended"], bins=_bins_act, include_lowest=True)
        _lgagg_act = (
            _lat_sub_act.groupby("att_bin", observed=True)
            .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_lgagg_act["mid_att"], _lgagg_act["mean_act"], marker=".", label="Latinx")
ax.axvline(GOOD_ATTENDANCE_PCT_MIN, color="gray", linestyle=":", alpha=0.8)
ax.set_xlabel("Percent days attended (bin center)")
ax.set_ylabel("Mean ACT composite")
ax.set_title("D. Mean ACT vs attendance by ethnicity")
ax.legend(title="Ethnicity", fontsize=7)

fig.suptitle(
    "Relationship Between Attendance and ACT Score\n"
    "Gray line = good attendance ref. (>170/180, membership 180).",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "attendance_act_hinge_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Chronic absenteeism rate by grade
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
fig.savefig(FIG_DIR / "chronic_absenteeism_rate_by_grade.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Chronic absenteeism rate by ethnicity
# ═══════════════════════════════════════════════════════════════════════════════

_pos = {"1", "y", "yes", "true", "t"}
_eth_col_labels = {
    "amerindian_alaskan":  "Native American",
    "asian":               "Asian",
    "black_african_amer":  "Black/African American",
    "hawaiian_pacific_isl": "Hawaiian/Pacific Islander",
    "white":               "White",
}
_eth_data5 = {}
for _col, _label in _eth_col_labels.items():
    if _col in _dbase.columns:
        _mask = _dbase[_col].astype(str).str.strip().str.lower().isin(_pos)
        _n = int(_mask.sum())
        if _n >= MIN_N:
            _eth_data5[_label] = {"rate": _dbase.loc[_mask, "_chronic"].mean(), "n": _n}
if "ethnicity" in _dbase.columns:
    _lat_mask = _dbase["ethnicity"].astype(str).str.strip().str.lower().isin(_pos)
    _lat_n = int(_lat_mask.sum())
    if _lat_n >= MIN_N:
        _eth_data5["Latinx"] = {"rate": _dbase.loc[_lat_mask, "_chronic"].mean(), "n": _lat_n}
if _eth_data5:
    _eth_df5 = pd.DataFrame(_eth_data5).T.sort_values("rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    _bars5 = ax.barh(_eth_df5.index, _eth_df5["rate"], color="#4c72b0")
    for _bar, (_idx, _row) in zip(_bars5, _eth_df5.iterrows()):
        ax.text(
            _bar.get_width() + 0.003, _bar.get_y() + _bar.get_height() / 2,
            f"n={int(_row['n']):,}", va="center", fontsize=8, color="#555555",
        )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, _eth_df5["rate"].max() * 1.25)
    ax.set_xlabel("Share chronically absent")
    ax.set_title("Chronic Absenteeism Rate by Ethnicity", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chronic_absenteeism_by_ethnicity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Chronic absenteeism by gender and disability status
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.set_theme(style="whitegrid")

ax = axes[0]
if "gender" in _dbase.columns:
    _gvals = _dbase["gender"].astype(str).str.strip().str.upper()
    _glabels = {"M": "Male", "F": "Female"}
    _grps = [g for g in ["M", "F"] if (_gvals == g).any()]
    ax.bar([_glabels[g] for g in _grps], [_dbase.loc[_gvals == g, "_chronic"].mean() for g in _grps], color="#c44e52")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylabel("Percent chronically absent")
    ax.set_title("A. Chronic absenteeism rate by gender")

ax = axes[1]
_dis_rates = (
    _dbase.groupby("_disability_group", observed=True)["_chronic"]
    .mean()
    .reindex(["No disability", "Disability"])
    .dropna()
)
if not _dis_rates.empty:
    ax.bar(_dis_rates.index, _dis_rates.values, color=["#4c72b0", "#c44e52"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylabel("Percent chronically absent")
    ax.set_title("B. Chronic absenteeism rate by disability status")

fig.suptitle("Chronic Absenteeism by Gender and Disability Status", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_absenteeism_by_gender_and_disability.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Chronic absenteeism rate by school
# ═══════════════════════════════════════════════════════════════════════════════

from matplotlib.patches import Patch as _Patch

_SCHOOL_GROUP_COLORS = {"Elementary": "#4c72b0", "Middle school": "#55a868", "High school": "#c44e52"}

def _school_level(name: str) -> str:
    if "Elem." in name:
        return "Elementary"
    if "middle" in name.lower():
        return "Middle school"
    return "High school"

_sagg = (
    _cbase[_cbase["_school"].notna()]
    .groupby("_school", dropna=True)
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_sagg["chronic_rate"] = _sagg["n_chronic"] / _sagg["n"]
_sagg = _sagg[_sagg["n"] >= MIN_N].copy()
_sagg = _sagg[~_sagg["_school"].str.lower().isin(["cache high", "other"])].sort_values("chronic_rate", ascending=True)
_sagg["_group"] = _sagg["_school"].apply(_school_level)

fig, ax = plt.subplots(figsize=(9, max(4, len(_sagg) * 0.45)))
sns.set_theme(style="whitegrid")
_sbars = ax.barh(_sagg["_school"], _sagg["chronic_rate"],
                 color=[_SCHOOL_GROUP_COLORS[g] for g in _sagg["_group"]])
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
ax.legend(
    handles=[_Patch(color=_SCHOOL_GROUP_COLORS[g], label=g)
             for g in ["Elementary", "Middle school", "High school"]],
    fontsize=8, loc="lower right",
)
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_rate_by_school.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote chronic_rate_by_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Chronic absenteeism rate by elementary school
# ═══════════════════════════════════════════════════════════════════════════════

_sagg_elem = (
    _cbase[_cbase["_school"].notna() & _cbase["_school"].str.contains("Elem.", na=False)]
    .groupby("_school", dropna=True)
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_sagg_elem["chronic_rate"] = _sagg_elem["n_chronic"] / _sagg_elem["n"]
_sagg_elem = _sagg_elem[_sagg_elem["n"] >= MIN_N].sort_values("chronic_rate", ascending=True)

fig, ax = plt.subplots(figsize=(9, max(4, len(_sagg_elem) * 0.45)))
sns.set_theme(style="whitegrid")
_sbars_elem = ax.barh(_sagg_elem["_school"], _sagg_elem["chronic_rate"],
                      color=_SCHOOL_GROUP_COLORS["Elementary"])
for _bar, _n in zip(_sbars_elem, _sagg_elem["n"]):
    ax.text(
        _bar.get_width() + 0.002, _bar.get_y() + _bar.get_height() / 2,
        f"n={_n:,}", va="center", fontsize=7, color="#555555"
    )
ax.set_xlabel("Rate of chronic absenteeism")
ax.set_title("Chronic Absenteeism Rate — Elementary Schools", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_rate_by_elementary_school.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote chronic_rate_by_elementary_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Chronic absenteeism rate by middle school
# ═══════════════════════════════════════════════════════════════════════════════

_sagg_ms = (
    _cbase[_cbase["_school"].notna() & _cbase["_school"].str.contains("Middle", na=False, case=False)]
    .groupby("_school", dropna=True)
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_sagg_ms["chronic_rate"] = _sagg_ms["n_chronic"] / _sagg_ms["n"]
_sagg_ms = _sagg_ms[_sagg_ms["n"] >= MIN_N].sort_values("chronic_rate", ascending=True)

fig, ax = plt.subplots(figsize=(9, max(4, len(_sagg_ms) * 0.45)))
sns.set_theme(style="whitegrid")
_sbars_ms = ax.barh(_sagg_ms["_school"], _sagg_ms["chronic_rate"],
                    color=_SCHOOL_GROUP_COLORS["Middle school"])
for _bar, _n in zip(_sbars_ms, _sagg_ms["n"]):
    ax.text(
        _bar.get_width() + 0.002, _bar.get_y() + _bar.get_height() / 2,
        f"n={_n:,}", va="center", fontsize=7, color="#555555"
    )
ax.set_xlabel("Rate of chronic absenteeism")
ax.set_title("Chronic Absenteeism Rate — Middle Schools", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_rate_by_middle_school.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote chronic_rate_by_middle_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 10 — Chronic absenteeism rate by high school
# ═══════════════════════════════════════════════════════════════════════════════

_sagg_hs = (
    _cbase[
        _cbase["_school"].notna() &
        ~_cbase["_school"].str.contains("Elem.", na=False) &
        ~_cbase["_school"].str.contains("Middle", na=False, case=False)
    ]
    .groupby("_school", dropna=True)
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_sagg_hs["chronic_rate"] = _sagg_hs["n_chronic"] / _sagg_hs["n"]
_sagg_hs = _sagg_hs[_sagg_hs["n"] >= MIN_N].copy()
_sagg_hs = _sagg_hs[~_sagg_hs["_school"].str.lower().isin(["cache high", "other"])].sort_values("chronic_rate", ascending=True)

fig, ax = plt.subplots(figsize=(9, max(4, len(_sagg_hs) * 0.45)))
sns.set_theme(style="whitegrid")
_sbars_hs = ax.barh(_sagg_hs["_school"], _sagg_hs["chronic_rate"],
                    color=_SCHOOL_GROUP_COLORS["High school"])
for _bar, _n in zip(_sbars_hs, _sagg_hs["n"]):
    ax.text(
        _bar.get_width() + 0.002, _bar.get_y() + _bar.get_height() / 2,
        f"n={_n:,}", va="center", fontsize=7, color="#555555"
    )
ax.set_xlabel("Rate of chronic absenteeism")
ax.set_title("Chronic Absenteeism Rate — High Schools", fontsize=11,
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_rate_by_high_school.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote chronic_rate_by_high_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 11 — Dual Language Immersion (DLI) and attendance
# ═══════════════════════════════════════════════════════════════════════════════

_dli_base = _cbase.copy()
_dli_base["_dli"] = pd.to_numeric(_dli_base["dli_ind"], errors="coerce") == 1.0
_dli_base["_year"] = pd.to_numeric(_dli_base["year"], errors="coerce")
_DLI_LABEL = {True: "DLI", False: "Non-DLI"}
_DLI_COLORS = {True: "#2ca02c", False: "#1f77b4"}

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
sns.set_theme(style="whitegrid")

# ── A: Chronic rate over time ──────────────────────────────────────────────────
ax = axes[0, 0]
_yr_agg16 = (
    _dli_base[_dli_base["_year"].between(2022, 2025)]
    .groupby(["_year", "_dli"])
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_yr_agg16 = _yr_agg16[_yr_agg16["n"] >= MIN_N].copy()
_yr_agg16["chronic_rate"] = _yr_agg16["n_chronic"] / _yr_agg16["n"]
for _dv16, _col16 in [(False, _DLI_COLORS[False]), (True, _DLI_COLORS[True])]:
    _s16 = _yr_agg16[_yr_agg16["_dli"] == _dv16]
    if not _s16.empty:
        ax.plot(_s16["_year"], _s16["chronic_rate"], marker="o", color=_col16,
                label=_DLI_LABEL[_dv16])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xlabel("Year")
ax.set_ylabel("Share chronically absent")
ax.set_title("A. Chronic absenteeism rate over time\n(all grades)")
ax.legend(fontsize=9)

# ── B: Chronic rate by grade (grades 7–8, where DLI data exists) ──────────────
ax = axes[0, 1]
_gr_agg16 = (
    _dli_base
    .groupby(["_grade", "_dli"])
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_gr_agg16["chronic_rate"] = _gr_agg16["n_chronic"] / _gr_agg16["n"]
_gr_agg16["_grade"] = _gr_agg16["_grade"].astype(int)
# Only keep grades where both DLI and non-DLI meet minimum n
_valid_grades16 = set()
for _g16, _ggrp in _gr_agg16.groupby("_grade"):
    if all(_ggrp[_ggrp["_dli"] == v]["n"].sum() >= MIN_N for v in [True, False]):
        _valid_grades16.add(_g16)
_gr_agg16 = _gr_agg16[_gr_agg16["_grade"].isin(_valid_grades16)]
_grades16 = sorted(_gr_agg16["_grade"].unique())
_x16 = np.arange(len(_grades16))
_w16 = 0.38
for _i16, _dv16 in enumerate([False, True]):
    _sub16 = _gr_agg16[_gr_agg16["_dli"] == _dv16].set_index("_grade").reindex(_grades16)
    _bars16 = ax.bar(
        _x16 + (_i16 - 0.5) * _w16,
        _sub16["chronic_rate"].fillna(0),
        _w16,
        color=_DLI_COLORS[_dv16],
        label=_DLI_LABEL[_dv16],
    )
    for _bar16, _n16 in zip(_bars16, _sub16["n"].fillna(0).astype(int)):
        if _n16 > 0:
            ax.text(
                _bar16.get_x() + _bar16.get_width() / 2,
                _bar16.get_height() + 0.003,
                f"n={_n16:,}", ha="center", fontsize=6.5, color="#555555",
            )
ax.set_xticks(_x16)
ax.set_xticklabels([f"Grade {g}" for g in _grades16])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_ylabel("Share chronically absent")
ax.set_title("B. Chronic absenteeism by grade")
ax.legend(fontsize=9)

# ── C: Attendance distribution (box plots) ────────────────────────────────────
ax = axes[1, 0]
_att16 = _dli_base[_dli_base["percent_days_attended"].between(50, 100)].copy()
_att16["_group"] = _att16["_dli"].map(_DLI_LABEL)
sns.boxplot(
    data=_att16, x="_group", y="percent_days_attended",
    hue="_group",
    order=["Non-DLI", "DLI"],
    palette={"DLI": _DLI_COLORS[True], "Non-DLI": _DLI_COLORS[False]},
    legend=False,
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("Percent days attended")
ax.set_title("C. Attendance distribution: DLI vs Non-DLI\n(all grades)")
for _gname16, _ggrp16 in _att16.groupby("_group"):
    _idx16 = ["Non-DLI", "DLI"].index(_gname16)
    ax.text(
        _idx16, _att16["percent_days_attended"].quantile(0.02) - 0.5,
        f"n={len(_ggrp16):,}", ha="center", fontsize=8, color="#555555",
    )

# ── D: Mean GPA comparison (grades 7–8) ───────────────────────────────────────
ax = axes[1, 1]
_gpa16 = (
    _dli_base[
        _dli_base["overall_gpa"].notna() & (_dli_base["overall_gpa"] > 0)
    ].copy()
)
_gpa16["_group"] = _gpa16["_dli"].map(_DLI_LABEL)
_gpa_means16 = (
    _gpa16.groupby("_group")["overall_gpa"]
    .agg(mean="mean", count="count")
    .reindex(["Non-DLI", "DLI"])
    .reset_index()
)
_bars16d = ax.bar(
    _gpa_means16["_group"], _gpa_means16["mean"],
    color=[_DLI_COLORS[v] for v in [False, True]],
)
for _bar16d, _row16d in zip(_bars16d, _gpa_means16.itertuples()):
    ax.text(
        _bar16d.get_x() + _bar16d.get_width() / 2,
        _bar16d.get_height() + 0.01,
        f"n={_row16d.count:,}", ha="center", fontsize=8, color="#555555",
    )
ax.set_xlabel("")
ax.set_ylabel("Mean cumulative GPA")
ax.set_title("D. Mean GPA: DLI vs Non-DLI (all grades)")
ax.set_ylim(0, _gpa_means16["mean"].max() * 1.15)

fig.suptitle(
    "Dual Language Immersion (DLI) and Attendance",
    fontsize=12, y=1.01,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "dli_attendance_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote dli_attendance_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 12 — Title 1 schools and attendance (elementary only)
# ═══════════════════════════════════════════════════════════════════════════════

_TITLE1_SCHOOLS = {"Birch Creek Elem.", "Canyon Elem.", "Lewiston Elem.", "Lincoln Elem.", "Summit Elem."}
_T1_COLORS = {True: "#c44e52", False: "#4c72b0"}
_T1_LABEL  = {True: "Title 1", False: "Non-Title 1"}

# Elementary-only base: "Elem." in school name, valid membership, 2022–2024 only
# (2025 is partial-year data, ~half a school year, so excluded)
_t1_base = _cbase[
    _cbase["_school"].str.contains("Elem.", na=False) &
    pd.to_numeric(_cbase["year"], errors="coerce").between(2022, 2024)
].copy()
_t1_base["_title1"] = _t1_base["_school"].isin(_TITLE1_SCHOOLS)
_t1_base["_year"]   = pd.to_numeric(_t1_base["year"], errors="coerce")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
sns.set_theme(style="whitegrid")

# ── A: Chronic rate by school (colored by Title 1 status), 2022–2024 pooled ───
ax = axes[0]
_sagg17 = (
    _t1_base.groupby("_school")
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"),
         title1=("_title1", "first"))
    .reset_index()
)
_sagg17 = _sagg17[_sagg17["n"] >= MIN_N].copy()
_sagg17["chronic_rate"] = _sagg17["n_chronic"] / _sagg17["n"]
_sagg17 = _sagg17.sort_values("chronic_rate", ascending=True)

_bars17a = ax.barh(_sagg17["_school"], _sagg17["chronic_rate"],
                   color=[_T1_COLORS[t] for t in _sagg17["title1"]])
for _bar, _row in zip(_bars17a, _sagg17.itertuples()):
    ax.text(
        _bar.get_width() + 0.002, _bar.get_y() + _bar.get_height() / 2,
        f"n={_row.n:,}", va="center", fontsize=7, color="#555555",
    )
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xlabel("Share chronically absent")
ax.set_title("A. Chronic absenteeism rate by school\n(2022–2024 pooled)")
ax.legend(handles=[_Patch(color=_T1_COLORS[True], label="Title 1"),
                   _Patch(color=_T1_COLORS[False], label="Non-Title 1")],
          fontsize=8, loc="lower right")

# ── B: Aggregate trend: Title 1 vs non-Title 1, 2022–2024 ────────────────────
ax = axes[1]
_yr_agg17 = (
    _t1_base.groupby(["_year", "_title1"])
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_yr_agg17 = _yr_agg17[_yr_agg17["n"] >= MIN_N].copy()
_yr_agg17["chronic_rate"] = _yr_agg17["n_chronic"] / _yr_agg17["n"]
for _tv in [False, True]:
    _s = _yr_agg17[_yr_agg17["_title1"] == _tv]
    if not _s.empty:
        ax.plot(_s["_year"], _s["chronic_rate"], marker="o",
                color=_T1_COLORS[_tv], label=_T1_LABEL[_tv])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xlabel("Year")
ax.set_ylabel("Share chronically absent")
ax.set_title("B. Chronic absenteeism over time\nTitle 1 vs non-Title 1 aggregate")
ax.legend(fontsize=9)
ax.set_xticks([2022, 2023, 2024])

# ── C: Per-school trend for each Title 1 school, non-Title 1 avg as reference ─
ax = axes[2]
_school_yr17 = (
    _t1_base.groupby(["_school", "_year", "_title1"])
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_school_yr17 = _school_yr17[_school_yr17["n"] >= MIN_N].copy()
_school_yr17["chronic_rate"] = _school_yr17["n_chronic"] / _school_yr17["n"]

# Non-Title 1 average reference line
_non_t1_ref = _yr_agg17[_yr_agg17["_title1"] == False]
ax.plot(_non_t1_ref["_year"], _non_t1_ref["chronic_rate"],
        color=_T1_COLORS[False], linewidth=2, linestyle="--",
        label="Non-Title 1 avg.", zorder=1)

# Individual Title 1 school lines
_t1_schools_data = _school_yr17[_school_yr17["_title1"] == True]
_t1_palette = sns.color_palette("Set2", n_colors=len(_TITLE1_SCHOOLS))
for (_sch, _grp), _col in zip(_t1_schools_data.groupby("_school"), _t1_palette):
    _short = _sch.replace(" Elem.", "")
    ax.plot(_grp["_year"], _grp["chronic_rate"], marker="o",
            color=_col, label=_short, zorder=2)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xlabel("Year")
ax.set_ylabel("Share chronically absent")
ax.set_title("C. Chronic absenteeism by Title 1 school\n(dashed = non-Title 1 avg.)")
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([2022, 2023, 2024])

fig.suptitle("Title 1 Schools and Attendance (Elementary Schools, 2022–2024)", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "title1_attendance_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote title1_attendance_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 13 — Mean GPA: chronic vs non-chronic by grade (7–12)
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
    fig.savefig(FIG_DIR / "mean_gpa_chronic_vs_not_by_grade.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 14 — Long-term impact: total years chronically absent vs high school GPA
# ═══════════════════════════════════════════════════════════════════════════════

# Chronic year count uses project-standard criteria (MIN_MEMBERSHIP, CHRONIC_ABSENT_PCT)
# to match Chart 15. The GPA display still filters to HS students with GPA.
_df8_all = df[df["school_membership"] >= MIN_MEMBERSHIP].copy()
_df8_all["_pct_absent8"] = 100.0 * _df8_all["days_absent"].fillna(0) / _df8_all["school_membership"]
_df8_all["_is_chronic_yr"] = _df8_all["_pct_absent8"] >= CHRONIC_ABSENT_PCT
_chronic_yrs8 = (
    _df8_all.groupby("student_number")["_is_chronic_yr"]
    .sum()
    .reset_index(name="years_chronic_total")
)
_df8 = df[df["school_membership"] >= MIN_MEMBERSHIP].copy()
_df8 = _df8.merge(_chronic_yrs8, on="student_number")
_df8 = _df8[(_df8["current_grade"] >= 9) & _df8["overall_gpa"].notna()]

_n_by_bucket8 = _df8.groupby("years_chronic_total").size()
_buckets8 = sorted(_n_by_bucket8.index)
_tick_labels8 = [f"{b}\n(n={_n_by_bucket8[b]:,})" for b in _buckets8]

fig, ax = plt.subplots(figsize=(12, 6))
sns.set_theme(style="whitegrid")
sns.boxplot(data=_df8, x="years_chronic_total", y="overall_gpa",
            hue="years_chronic_total", palette="viridis", legend=False, ax=ax)
ax.set_xticks(range(len(_buckets8)))
ax.set_xticklabels(_tick_labels8, fontsize=9)
ax.set_xlabel(f"Total years chronically absent (≥{CHRONIC_ABSENT_PCT:.0f}% days missed, membership ≥{MIN_MEMBERSHIP})")
ax.set_ylabel("Overall GPA")
ax.set_title("Long-term Impact: Total Years Chronically Absent vs. High School GPA", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_years_vs_hs_gpa_boxplot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote chronic_years_vs_hs_gpa_boxplot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 15 — Multi-year chronic absenteeism and disability status
# ═══════════════════════════════════════════════════════════════════════════════

_dis_base = df[df["school_membership"] >= 60].copy()
_dis_base["_pct_absent"] = 100.0 * _dis_base["days_absent"].fillna(0) / _dis_base["school_membership"]
_dis_base["_chronic"]    = _dis_base["_pct_absent"] >= CHRONIC_ABSENT_PCT
_dis_base["_disability"] = _dis_base["scram_membership"].fillna(0) > 0

# Student-level summary: chronic year count + disability (ever flagged in any year)
_stu18 = (
    _dis_base.groupby("student_number")
    .agg(
        chronic_years=("_chronic", "sum"),
        ever_disability=("_disability", "any"),
    )
    .reset_index()
)
# Cap at 4+ years to keep sample sizes meaningful (n=36 at 5 years)
_stu18["chronic_bucket"] = _stu18["chronic_years"].clip(upper=4).astype(int)
_bucket_labels = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"}
_stu18["chronic_label"] = _stu18["chronic_bucket"].map(_bucket_labels)

_rate_by_yr18 = (
    _stu18.groupby("chronic_bucket")
    .agg(n=("student_number", "count"), n_dis=("ever_disability", "sum"))
    .reset_index()
)
_rate_by_yr18 = _rate_by_yr18[_rate_by_yr18["n"] >= MIN_N].copy()
_rate_by_yr18["dis_rate"] = _rate_by_yr18["n_dis"] / _rate_by_yr18["n"]
_rate_by_yr18["label"]    = _rate_by_yr18["chronic_bucket"].map(_bucket_labels)

fig, ax = plt.subplots(figsize=(8, 5))
sns.set_theme(style="whitegrid")
_bars18 = ax.bar(_rate_by_yr18["label"], _rate_by_yr18["dis_rate"], color="#8172b3")
for _bar, _row in zip(_bars18, _rate_by_yr18.itertuples()):
    ax.text(
        _bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.003,
        f"n={_row.n:,}", ha="center", fontsize=8, color="#555555",
    )
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xlabel("Years chronically absent\n(Disability = SCRAM membership > 0)")
ax.set_ylabel("Share of students with a disability")
ax.set_title("Disability Rate by Number of Chronic Absence Years")
fig.text(
    0.5, -0.04,
    "Note: Disability is measured by SCRAM membership (health accommodation services).\n"
    "Students classified as severely disabled (IsOnePercent = Y) were excluded from this dataset.",
    ha="center", fontsize=8, color="#666666", style="italic",
    wrap=True,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "disability_chronic_absence.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote disability_chronic_absence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 16 — GPA impact of each additional year of chronic absenteeism
# ═══════════════════════════════════════════════════════════════════════════════

_gpa20_all = df[df["school_membership"] >= MIN_MEMBERSHIP].copy()
_gpa20_all["_pct_absent20"] = 100.0 * _gpa20_all["days_absent"].fillna(0) / _gpa20_all["school_membership"]
_gpa20_all["_is_chronic20"] = _gpa20_all["_pct_absent20"] >= CHRONIC_ABSENT_PCT
_chronic_yrs20 = (
    _gpa20_all.groupby("student_number")["_is_chronic20"]
    .sum()
    .reset_index(name="chronic_years")
)
_gpa20 = df[df["school_membership"] >= MIN_MEMBERSHIP].copy()
_gpa20 = _gpa20.merge(_chronic_yrs20, on="student_number")
_gpa20 = _gpa20[(_gpa20["current_grade"] >= 9) & _gpa20["overall_gpa"].notna()]

_gpa_agg20 = (
    _gpa20.groupby("chronic_years")
    .agg(n=("student_number", "count"), mean_gpa=("overall_gpa", "mean"))
    .reset_index()
)
_gpa_agg20 = _gpa_agg20[_gpa_agg20["n"] >= MIN_N].copy()
_gpa_agg20["gpa_drop"] = _gpa_agg20["mean_gpa"].diff()

fig, ax = plt.subplots(figsize=(10, 6))
sns.set_theme(style="whitegrid")

_bars20 = ax.bar(
    _gpa_agg20["chronic_years"], _gpa_agg20["mean_gpa"],
    color="#4c72b0", width=0.6,
)
# Label each bar with mean GPA and n
for _bar, _row in zip(_bars20, _gpa_agg20.itertuples()):
    ax.text(
        _bar.get_x() + _bar.get_width() / 2,
        _bar.get_height() + 0.02,
        f"{_row.mean_gpa:.2f}\n(n={_row.n:,})",
        ha="center", va="bottom", fontsize=8, color="#333333",
    )

# Annotate the drop between consecutive bars
for i, _row in _gpa_agg20.iterrows():
    if pd.isna(_row["gpa_drop"]):
        continue
    _prev = _gpa_agg20.loc[i - 1]
    _x_mid = _row["chronic_years"] - 0.5
    _y_mid  = (_row["mean_gpa"] + _prev["mean_gpa"]) / 2
    ax.annotate(
        f"−{abs(_row['gpa_drop']):.2f}",
        xy=(_x_mid, _y_mid),
        ha="center", va="center", fontsize=12, color="#c44e52", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c44e52", alpha=0.85),
    )

ax.set_xlabel("Total years chronically absent")
ax.set_ylabel("Mean GPA (high school)")
ax.set_title(
    "GPA Impact of Each Additional Year of Chronic Absenteeism\n"
    "Red labels show GPA drop from the prior chronic-year group",
    fontsize=12,
)
ax.set_xticks(_gpa_agg20["chronic_years"])
ax.set_ylim(0, _gpa_agg20["mean_gpa"].max() + 0.45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
fig.tight_layout()
fig.savefig(FIG_DIR / "gpa_impact_per_chronic_year.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote gpa_impact_per_chronic_year.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 17 — Improvers count by transition length
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


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 18 — Percentage of improvers by grade
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
    ax.set_xlabel("Grade (at Year of Improvement)")
    ax.set_ylabel("Improvers / All Poor Attenders")
    ax.set_title("Percentage of Improvers by Grade")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "percentage_of_improvers_by_grade.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote percentage_of_improvers_by_grade.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 19 — Percentage of improvers by school group
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
    ax.set_xlabel("School Group (at Year of Improvement)")
    ax.set_ylabel("Improvers / All Poor Attenders")
    ax.set_title("Percentage of Improvers by School Group")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "percentage_of_improvers_by_school_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote percentage_of_improvers_by_school_group.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 20 — Percentage of improvers by school
# ═══════════════════════════════════════════════════════════════════════════════

_n_trans12 = _anc_school.groupby("current_school")["student_number"].nunique().rename("n_trans")
_n_poor12  = _poor_rows_anchor_school.groupby("current_school")["student_number"].nunique().rename("n_poor")
_school_counts = pd.concat([_n_trans12, _n_poor12], axis=1).fillna({"n_trans": 0, "n_poor": 0}).reset_index()
_school_counts = _school_counts[_school_counts["n_poor"] > 0].copy()
_school_counts = _school_counts[_school_counts["current_school"].str.lower() != "cache high"].copy()
_school_counts["n"] = _school_counts["n_trans"] / _school_counts["n_poor"]
_school_counts["_group"] = _school_counts["current_school"].map(_group_label)
_school_counts = _school_counts.sort_values("n", ascending=False)

if not _school_counts.empty:
    fig, ax = plt.subplots(figsize=(max(10.0, 0.38 * len(_school_counts)), 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=_school_counts, x="current_school", y="n",
        palette={s: _SCHOOL_GROUP_COLORS[g] for s, g in zip(_school_counts["current_school"], _school_counts["_group"])},
        ax=ax,
    )
    ax.set_xlabel("School (at Year of Improvement)")
    ax.set_ylabel("Improvers / All Poor Attenders")
    ax.set_title("Percentage of Improvers by School")
    ax.legend(
        handles=[_Patch(color=_SCHOOL_GROUP_COLORS[g], label=g)
                 for g in ["Elementary", "Middle school", "High school"]],
        title="School group", fontsize=8, loc="upper right",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "percentage_of_improvers_by_school.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote percentage_of_improvers_by_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 21 — Top 10 middle school teachers by percentage of improvers
# CHART 22 — Top 10 high school teachers by percentage of improvers
# ═══════════════════════════════════════════════════════════════════════════════

_sy14 = (
    _base_imp.groupby(["student_number", "year"], as_index=False)["percent_days_attended"].mean()
    .rename(columns={"percent_days_attended": "pct"})
)
_poor14 = _sy14[_sy14["pct"] < 90.0][["student_number", "year"]].rename(columns={"year": "poor_year"})
_good14 = _sy14[_sy14["pct"] >= 95.0][["student_number", "year"]].rename(columns={"year": "good_year"})
_oyt = _poor14.merge(_good14, on="student_number", how="inner")
_oyt = _oyt[_oyt["good_year"] == _oyt["poor_year"] + 1].drop_duplicates(subset=["student_number", "poor_year"])
_oyt = _oyt.sort_values("poor_year").drop_duplicates(subset="student_number", keep="first")

if _tg_path.is_file() and not _oyt.empty:
    _tgrid = pd.read_csv(_tg_path)
    _tcols = [c for c in _tgrid.columns if c.startswith("teacher_")]
    _tgrid["student_number"] = _tgrid["student_number"].astype(str)
    _tgrid["year"] = pd.to_numeric(_tgrid["year"], errors="coerce")
    _exp = _oyt.merge(_tgrid, left_on=["student_number", "good_year"], right_on=["student_number", "year"], how="inner")
    if not _exp.empty and _tcols:
        _n_trans14 = (_exp[_tcols] > 0).sum(axis=0).astype(int)
        _poor14_shifted = _poor14.copy()
        _poor14_shifted["lookup_year"] = _poor14_shifted["poor_year"] + 1
        _merged_poor14 = _tgrid[["student_number", "year"] + _tcols].merge(
            _poor14_shifted[["student_number", "lookup_year"]],
            left_on=["student_number", "year"], right_on=["student_number", "lookup_year"], how="inner"
        )
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
        _tc = _tc[(_tc["n_trans"] > 0) & (_tc["n_poor"] >= 10)].copy()
        _tc["ratio"] = np.where(_tc["n_poor"] > 0, _tc["n_trans"] / _tc["n_poor"], np.nan)
        if not _tc.empty:
            if len(_tc) == 1:
                _pool = _tc.sort_values(["ratio", "n_trans"], ascending=[False, False])
            else:
                _avg_load = float(_tc["n_enrolled"].mean())
                _pool = _tc[_tc["n_enrolled"] > _avg_load].sort_values(["ratio", "n_trans"], ascending=[False, False])

            if not _pool.empty:
                # Step 1: build static affiliations from curated CSV files
                _affiliation_static: dict[str, set[str]] = {}
                for _aff_path in (_hs_path, _ms_path, _es_path):
                    if not _aff_path.is_file():
                        continue
                    _aff_grid = pd.read_csv(_aff_path, index_col=0)
                    for _sch_name, _aff_row in _aff_grid.iterrows():
                        _school14 = str(_sch_name).strip()
                        for _col in _tcols:
                            if _col in _aff_grid.columns and pd.notna(_aff_row[_col]) and float(_aff_row[_col]) >= 1:
                                _affiliation_static.setdefault(_col, set()).add(_school14)

                # Step 2: find which schools each teacher has students at in the 2022+ data
                _base_sn = _base_imp[["student_number", "year", "current_school"]].drop_duplicates(subset=["student_number", "year"], keep="first").copy()
                _base_sn["student_number"] = _base_sn["student_number"].astype(str)
                _tgrid_with_school = _tgrid.merge(_base_sn, on=["student_number", "year"], how="inner")
                _recent_schools: dict[str, set[str]] = {}
                for _col in _tcols:
                    for _s in _tgrid_with_school.loc[_tgrid_with_school[_col] > 0, "current_school"].dropna().unique():
                        _s_str = str(_s).strip()
                        if _s_str and _s_str != "nan":
                            _recent_schools.setdefault(_col, set()).add(_s_str)

                # Step 3: intersect — keep only static affiliations confirmed by recent data
                _affiliation: dict[str, set[str]] = {
                    _col: _affiliation_static[_col] & _recent_schools.get(_col, set())
                    for _col in _affiliation_static
                    if _affiliation_static[_col] & _recent_schools.get(_col, set())
                }

                _ms_teacher_set = {
                    t for t, schools in _affiliation.items()
                    if any(_school_level(s) == "Middle school" for s in schools)
                }
                _hs_teacher_set = {
                    t for t, schools in _affiliation.items()
                    if any(_school_level(s) == "High school" for s in schools)
                }

                def _plot_top_teachers(pool, teacher_set, n, title, filename, color):
                    sub = pool[pool["teacher"].isin(teacher_set)].head(n).iloc[::-1].copy()
                    if sub.empty:
                        return
                    sub["ratio_plot"] = sub["ratio"].fillna(0.0)
                    fig, ax = plt.subplots(figsize=(14, max(4, len(sub) * 0.7)))
                    sns.set_theme(style="whitegrid")
                    sns.barplot(data=sub, x="ratio_plot", y="teacher", color=color, ax=ax)
                    ax.set_xlabel("Improved / Previously Poor-Attending Students in Class")
                    ax.set_ylabel("Teacher")
                    ax.set_title(title)
                    _mx = float(sub["ratio_plot"].max()) if not sub.empty else 0.0
                    _xp = max(0.02, 0.04 * _mx) if _mx > 0 else 0.05
                    ax.set_xlim(0, min(1.0, _mx + _xp) if _mx <= 1.0 else _mx + _xp)
                    for _patch, _row in zip(ax.patches, sub.itertuples(index=False)):
                        _sch_list = sorted(_affiliation.get(_row.teacher, set()))
                        _sch_txt = ", ".join(_sch_list) if _sch_list else "Unmapped"
                        _rtxt = f"{_row.n_trans}/{_row.n_poor}" if _row.n_poor > 0 else f"{_row.n_trans}/0"
                        ax.text(
                            _patch.get_width() + _xp,
                            _patch.get_y() + _patch.get_height() / 2.0,
                            f"{_sch_txt}  ({_rtxt}, N={_row.n_enrolled})",
                            va="center", ha="left", fontsize=8,
                        )
                    fig.tight_layout()
                    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Wrote {filename}")

                # ── CHART 21 ──────────────────────────────────────────────────
                _plot_top_teachers(
                    _pool, _ms_teacher_set, 10,
                    "Top 10 Middle School Teachers: 1-Year Attendance Turnaround Rate",
                    "top_10_ms_teachers_by_percentage_of_improvers.png",
                    _SCHOOL_GROUP_COLORS["Middle school"],
                )

                # ── CHART 22 ──────────────────────────────────────────────────
                _plot_top_teachers(
                    _pool, _hs_teacher_set, 10,
                    "Top 10 High School Teachers: 1-Year Attendance Turnaround Rate",
                    "top_10_hs_teachers_by_percentage_of_improvers.png",
                    _SCHOOL_GROUP_COLORS["High school"],
                )



# ═══════════════════════════════════════════════════════════════════════════════
# CHART 23 — Sky View policy impact: mean attendance trend (2022–2025)
# Policy: teacher-driven engagement implemented 2024-25 school year
# ═══════════════════════════════════════════════════════════════════════════════
_sv_schools  = ["Green Canyon", "Mountain Crest", "Ridgeline", "Sky View"]
_sv_colors   = {"Sky View": "#d62728", "Green Canyon": "#aaaaaa",
                "Mountain Crest": "#aaaaaa", "Ridgeline": "#aaaaaa"}
_sv_policy_year = 2025  # 2024-25 school year (end-year convention)

_sv_hs = df[
    df["current_school"].isin(_sv_schools) &
    (df["school_membership"] >= MIN_MEMBERSHIP) &
    df["percent_days_attended"].notna() &
    df["year"].notna() &
    (df["year"] >= 2022)
].copy()

_sv_agg = (
    _sv_hs.groupby(["current_school", "year"])
    .agg(mean_att=("percent_days_attended", "mean"))
    .reset_index()
)
_sv_agg["year"] = _sv_agg["year"].astype(int)
_sv_years = sorted(_sv_agg["year"].unique())

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_theme(style="whitegrid")
for _sv_school in _sv_schools:
    _sv_sub = _sv_agg[_sv_agg["current_school"] == _sv_school].sort_values("year")
    _sv_clr = _sv_colors[_sv_school]
    _sv_lw  = 2.5 if _sv_school == "Sky View" else 1.2
    _sv_zo  = 3   if _sv_school == "Sky View" else 1
    _sv_ms  = 6   if _sv_school == "Sky View" else 4
    ax.plot(_sv_sub["year"], _sv_sub["mean_att"],
            color=_sv_clr, lw=_sv_lw, marker="o", ms=_sv_ms, zorder=_sv_zo,
            label=_sv_school)

_sv_ylims  = ax.get_ylim()
_sv_yrange = _sv_ylims[1] - _sv_ylims[0]
ax.axvline(_sv_policy_year - 0.5, color="black", ls=":", lw=1.2, alpha=0.6)
ax.text(_sv_policy_year - 0.42, _sv_ylims[1] - _sv_yrange * 0.02,
        "Policy implemented\n(2024–25 →)", fontsize=8, va="top", ha="left")
ax.set_xticks(_sv_years)
ax.set_xticklabels(_sv_years)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_xlabel("Year (end of school year)")
ax.set_ylabel("Mean Attendance Rate (% days attended)")
ax.set_title("Mean Attendance Rate by High School (2022–2025)\nSky View policy implemented 2024–25")
ax.legend(title="School", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "skyview_policy_mean_attendance_trend.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote skyview_policy_mean_attendance_trend.png")


print(f"\nDone. Figures written to: {FIG_DIR}")


import geopandas as gpd
import io

# ── Geo map paths & helpers ────────────────────────────────────────────────────
_HS_POLY = ROOT / "data" / "High Schools Boundary Maps" / "High Schools-POLYGON.shp"
_MS_DIR  = ROOT / "data" / "Middle Schools Boundary Map"
_ES_POLY = ROOT / "data" / "Elementary Boundary Maps" / "Elementary Boundary.shp"

# Convert polygons to long-form coordinate df for geom_polygon (from 09_geo-example.py)
def _gdf_to_coords(gdf, id_col="att_school"):
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            for x, y in zip(xs, ys):
                rows.append({id_col: row[id_col], "lon": x, "lat": y})
    return pd.DataFrame(rows)

def _geo_school_metrics(schools):
    rows = []
    for s in schools:
        sub = _cbase[_cbase["current_school"] == s]
        if len(sub) < 10:
            rows.append({"school": s, "mean_att": np.nan, "chronic_rate": np.nan, "improver_rate": np.nan})
            continue
        n_poor  = _poor_rows_anchor_school[_poor_rows_anchor_school["current_school"] == s]["student_number"].nunique()
        n_trans = _anc_school[_anc_school["current_school"] == s]["student_number"].nunique()
        rows.append({
            "school":        s,
            "mean_att":      sub["percent_days_attended"].mean(),
            "chronic_rate":  sub["_chronic"].mean() * 100,
            "improver_rate": (n_trans / n_poor * 100) if n_poor > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def _prep_geo(gdf, label_map, metrics):
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[["att_school", "geometry"]].dissolve(by="att_school", as_index=False)
    gdf = gdf.merge(metrics.rename(columns={"school": "att_school"}), on="att_school", how="left")
    gdf_proj = gdf.to_crs("EPSG:3857")
    centroids = gdf_proj.geometry.centroid.to_crs("EPSG:4326")
    labels_df = pd.DataFrame({
        "att_school": gdf["att_school"],
        "label": gdf["att_school"].map(label_map).fillna(gdf["att_school"]),
        "lon": centroids.x,
        "lat": centroids.y,
    })
    coords = _gdf_to_coords(gdf)
    coords = coords.merge(
        gdf[["att_school", "mean_att", "chronic_rate", "improver_rate"]],
        on="att_school", how="left",
    )
    return coords, labels_df

def _make_geo_plot(coords, labels_df, fill_col, fill_label, palette, title):
    return (
        ggplot(coords, aes(x="lon", y="lat", group="att_school", fill=fill_col))
        + geom_polygon(color="white", size=0.5, alpha=0.85)
        + geom_text(
            data=labels_df,
            mapping=aes(x="lon", y="lat", label="label", group="att_school"),
            size=7,
            color="#1a1a1a",
            fontweight="bold",
            inherit_aes=False,
        )
        + scale_fill_distiller(palette=palette, direction=1, name=fill_label)
        + coord_fixed()
        + labs(title=title, x=None, y=None)
        + theme_void()
        + theme(
            plot_title=element_text(size=12, face="bold", ha="center"),
            figure_size=(9, 8),
        )
    )

def _save_combined_geo(plots, suptitle, filename):
    arrays = []
    for p in plots:
        buf = io.BytesIO()
        p.save(buf, format="png", dpi=150, verbose=False)
        buf.seek(0)
        arrays.append(plt.imread(buf))
    fig, axes = plt.subplots(1, 3, figsize=(28, 9))
    for ax, arr in zip(axes, arrays):
        ax.imshow(arr)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {filename}")

# Name maps: shapefile Name (after \xa0 fix) → (short display label, attendance data key)
_HS_MAP = {
    "Sky View High School":     ("Sky View",      "Sky View"),
    "Green Canyon High School": ("Green Canyon",  "Green Canyon"),
    "Ridgeline High School":    ("Ridgeline",     "Ridgeline"),
    "Mt. Crest High School":    ("Mt. Crest",     "Mountain Crest"),
}
_MS_MAP = {
    "North Cache Middle School":  ("North Cache",  "North Cache Middle"),
    "South Cache Middle School":  ("South Cache",  "South Cache Middle"),
    "Spring Creek Middle School": ("Spring Creek", "Spring Creek Middle"),
}
# Wellesville is a typo in the attendance CSV; the shapefile has Wellsville.
_ES_MAP = {
    "Birch Creek Elementary":   ("Birch Creek",   "Birch Creek Elem."),
    "Canyon Elementary":        ("Canyon",         "Canyon Elem."),
    "Cedar Ridge Elementary":   ("Cedar Ridge",    "Cedar Ridge Elem."),
    "Greenville Elementary":    ("Greenville",     "Greenville Elem."),
    "Heritage Elementary":      ("Heritage",       "Heritage Elem."),
    "Lewiston Elementary":      ("Lewiston",       "Lewiston Elem."),
    "Lincoln Elementary":       ("Lincoln",        "Lincoln Elem."),
    "Millville Elementary":     ("Millville",      "Millville Elem."),
    "Mountainside Elementary":  ("Mountainside",   "Mountainside Elem."),
    "Nibley Elementary":        ("Nibley",         "Nibley Elem."),
    "North Park Elementary":    ("North Park",     "North Park Elem."),
    "Providence Elementary":    ("Providence",     "Providence Elem."),
    "River Heights Elementary": ("River Heights",  "River Heights Elem."),
    "Summit Elementary":        ("Summit",         "Summit Elem."),
    "Sunrise Elementary":       ("Sunrise",        "Sunrise Elem."),
    "Wellsville Elementary":    ("Wellsville",     "Wellesville Elem."),
    "White Pine Elementary":    ("White Pine",     "White Pine Elem."),
}

# Prepare GeoDataFrames and coord/label tables once per level
_gdf_hs = gpd.read_file(_HS_POLY)
_gdf_hs["Name"]       = _gdf_hs["Name"].str.replace("\xa0", " ", regex=False)
_gdf_hs["att_school"] = _gdf_hs["Name"].map({k: v[1] for k, v in _HS_MAP.items()})
_hs_metrics            = _geo_school_metrics([v[1] for v in _HS_MAP.values()])
_hs_coords, _hs_labels = _prep_geo(_gdf_hs, {v[1]: v[0] for v in _HS_MAP.values()}, _hs_metrics)

_ms_parts = []
for _shp in sorted(_MS_DIR.glob("*-POLYGON.shp")):
    _g = gpd.read_file(_shp)
    _g["Name"]       = _g["Name"].str.replace("\xa0", " ", regex=False)
    _g["att_school"] = _g["Name"].map({k: v[1] for k, v in _MS_MAP.items()})
    _ms_parts.append(_g)
_gdf_ms = gpd.GeoDataFrame(pd.concat(_ms_parts, ignore_index=True), geometry="geometry")
_ms_metrics            = _geo_school_metrics([v[1] for v in _MS_MAP.values()])
_ms_coords, _ms_labels = _prep_geo(_gdf_ms, {v[1]: v[0] for v in _MS_MAP.values()}, _ms_metrics)

_gdf_es = gpd.read_file(_ES_POLY)
_gdf_es["Name"]       = _gdf_es["Name"].str.replace("\xa0", " ", regex=False)
_gdf_es["att_school"] = _gdf_es["Name"].map({k: v[1] for k, v in _ES_MAP.items()})
_es_metrics            = _geo_school_metrics([v[1] for v in _ES_MAP.values()])
_es_coords, _es_labels = _prep_geo(_gdf_es, {v[1]: v[0] for v in _ES_MAP.values()}, _es_metrics)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 24 — Geographic: mean attendance rate by school boundary (all levels)
# ═══════════════════════════════════════════════════════════════════════════════

_save_combined_geo(
    plots=[
        _make_geo_plot(_hs_coords, _hs_labels, "mean_att", "Mean Attendance (%)", "Blues", "High Schools"),
        _make_geo_plot(_ms_coords, _ms_labels, "mean_att", "Mean Attendance (%)", "Blues", "Middle Schools"),
        _make_geo_plot(_es_coords, _es_labels, "mean_att", "Mean Attendance (%)", "Blues", "Elementary Schools"),
    ],
    suptitle="Mean Attendance Rate (%) by School Attendance Boundary",
    filename="geo_mean_att.png",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 25 — Geographic: chronic absenteeism rate by school boundary (all levels)
# ═══════════════════════════════════════════════════════════════════════════════

_save_combined_geo(
    plots=[
        _make_geo_plot(_hs_coords, _hs_labels, "chronic_rate", "Chronic Absenteeism (%)", "Reds", "High Schools"),
        _make_geo_plot(_ms_coords, _ms_labels, "chronic_rate", "Chronic Absenteeism (%)", "Reds", "Middle Schools"),
        _make_geo_plot(_es_coords, _es_labels, "chronic_rate", "Chronic Absenteeism (%)", "Reds", "Elementary Schools"),
    ],
    suptitle="Chronic Absenteeism Rate (%) by School Attendance Boundary",
    filename="geo_chronic.png",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 26 — Geographic: improver rate by school boundary (all levels)
# ═══════════════════════════════════════════════════════════════════════════════

_save_combined_geo(
    plots=[
        _make_geo_plot(_hs_coords, _hs_labels, "improver_rate", "Improver Rate (%, 2022–2025)", "Greens", "High Schools"),
        _make_geo_plot(_ms_coords, _ms_labels, "improver_rate", "Improver Rate (%, 2022–2025)", "Greens", "Middle Schools"),
        _make_geo_plot(_es_coords, _es_labels, "improver_rate", "Improver Rate (%, 2022–2025)", "Greens", "Elementary Schools"),
    ],
    suptitle="Improver Rate (%, 2022–2025) by School Attendance Boundary",
    filename="geo_improvers.png",
)
