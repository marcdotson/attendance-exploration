from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "exploratory_data_all_grades.csv"
FIG_DIR   = ROOT / "figures"

MIN_N              = 30
MIN_MEMBERSHIP     = 60
CHRONIC_ABSENT_PCT = 10.0
# Improver definition (all charts 9–15): any student with at least one poor year (< 90%)
# and a later good year (>= 95%), allowing intermediate years in [90, 95).
IMPROVER_POOR_MAX_PCT = 90.0
IMPROVER_GOOD_MIN_PCT = 95.0

# School tier labels (must match _group_label return values and chart categoricals)
_SCHOOL_GROUP_ORDER: tuple[str, ...] = ("Elementary", "Middle School", "High School")

_hs_path = ROOT / "data" / "hs_teacher_data.csv"
_ms_path = ROOT / "data" / "ms_teacher_data.csv"
_es_path = ROOT / "data" / "es_teacher_data.csv"
_tg_path = ROOT / "data" / "all_teachers_grid_by_year.csv"

_hs_names = {str(i).strip() for i in pd.read_csv(_hs_path, index_col=0).index} if _hs_path.is_file() else set()
_ms_names = {str(i).strip() for i in pd.read_csv(_ms_path, index_col=0).index} if _ms_path.is_file() else set()
_es_names = {str(i).strip() for i in pd.read_csv(_es_path, index_col=0).index} if _es_path.is_file() else set()


def _group_label(school_name: object) -> str:
    s = str(school_name).strip()
    sl = s.lower()
    if s in _es_names or "elem." in sl:
        return "Elementary"
    if s in _ms_names or "middle" in sl:
        return "Middle School"
    return "High School"


_SCHOOL_GROUP_COLORS: dict[str, str] = {
    "Elementary": "#55a868",
    "Middle School": "#4c72b0",
    "High School": "#c44e52",
}
_SCHOOL_EXCLUDE = {"cache high", "other"}

# ── Load & coerce ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
for _col in ["percent_days_attended", "school_membership", "overall_gpa", "composite_score",
             "current_grade", "days_absent", "scram_membership", "year"]:
    if _col in df.columns:
        df[_col] = pd.to_numeric(df[_col], errors="coerce")
df["student_number"] = df["student_number"].astype(str)

# Canonical ethnicity (same logic as 08_attendance_eda._prep) for GPA hinge and demographics
ETHNICITY_DUMMY_COLUMNS: tuple[str, ...] = (
    "amerindian_alaskan",
    "asian",
    "black_african_amer",
    "hawaiian_pacific_isl",
    "white",
)
LATINX_INDICATOR_COLUMN = "ethnicity"
CHART_ETHNICITY_ORDER: tuple[str, ...] = (
    "white",
    "latinx",
    "asian",
    "black_african_amer",
    "hawaiian_pacific_isl",
    "amerindian_alaskan",
)
CHART_ETHNICITY_LABEL: dict[str, str] = {
    "white": "White",
    "latinx": "Latinx",
    "asian": "Asian",
    "black_african_amer": "Black",
    "hawaiian_pacific_isl": "Hawaiian/Pacific",
    "amerindian_alaskan": "Native American",
}


def _row_ethnicity_positive(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value == 1
    return str(value).strip().lower() in {"1", "y", "yes", "true", "t"}


def _canonical_ethnicity_row(row: pd.Series) -> str:
    if LATINX_INDICATOR_COLUMN in row.index and _row_ethnicity_positive(
        row.get(LATINX_INDICATOR_COLUMN)
    ):
        return "latinx"
    for c in ETHNICITY_DUMMY_COLUMNS:
        if c in row.index and _row_ethnicity_positive(row.get(c)):
            return c
    raw = row.get(LATINX_INDICATOR_COLUMN, np.nan)
    if pd.isna(raw):
        return "unknown"
    raw_str = str(raw).strip()
    return raw_str if raw_str else "unknown"


if LATINX_INDICATOR_COLUMN in df.columns or any(c in df.columns for c in ETHNICITY_DUMMY_COLUMNS):
    df["ethnicity"] = df.apply(_canonical_ethnicity_row, axis=1)


def _chronic_rate(sub: pd.DataFrame) -> float:
    return float(sub["_chronic"].mean()) if len(sub) else float("nan")


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
    _dbase["scram_membership"].fillna(0) > 0, "Disability", "No Disability"
)

# Improver charts base (9–15) — 09-style filters, then improvers are students with
# any later good year after at least one poor year (see _improvers below).
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

_poor_rows = _base_imp[_base_imp["percent_days_attended"] < IMPROVER_POOR_MAX_PCT].copy()

# Improver candidate: first observed poor year, and first observed good year after that poor year.
_first_poor_yr = _poor_rows.groupby("student_number")["year"].min().rename("first_poor_year")
_good_rows = (
    _base_imp[_base_imp["percent_days_attended"] >= IMPROVER_GOOD_MIN_PCT][["student_number", "year"]]
    .join(_first_poor_yr, on="student_number")
)
_good_rows = _good_rows[_good_rows["year"] > _good_rows["first_poor_year"]]
_first_good_yr = _good_rows.groupby("student_number")["year"].min().rename("first_good_year")
_cand = pd.concat([_first_poor_yr, _first_good_yr], axis=1).dropna()
_improvers = _cand.copy()

# Anchor at first_good_year - 1 (for denominator: pool of poor student-years in the last school year before good)
_anchor = _improvers.reset_index()[["student_number", "first_good_year"]].copy()
_anchor["anchor_year"] = _anchor["first_good_year"] - 1

_sg_lookup = (_base_imp[["student_number", "year", "current_school", "current_grade"]]
              .drop_duplicates(subset=["student_number", "year"], keep="first"))
# Improver rows tagged with school/grade in the first good year (transition year: good attendance)
_trans_first_good = (
    _improvers.reset_index()[["student_number", "first_good_year"]]
    .merge(
        _sg_lookup,
        left_on=["student_number", "first_good_year"],
        right_on=["student_number", "year"],
        how="inner",
    )
    .drop_duplicates("student_number", keep="first")
)
_anc_grade  = _trans_first_good.dropna(subset=["current_grade"])
_anc_school = _trans_first_good.dropna(subset=["current_school"])

# Denominator: poor rows in anchor years (same window as 09); re-label with first-good
# school/grade for improvers so rates align with the improver “transition” year.
_poor_rows_anchor = (
    _poor_rows[_poor_rows["year"].isin(_anchor["anchor_year"].unique())]
    .drop_duplicates(subset=["student_number", "year"], keep="first")
)
_fg_from_good = _trans_first_good[["student_number", "current_grade", "current_school"]].rename(
    columns={"current_grade": "fg_grade", "current_school": "fg_school"},
)
_poor_rows_disp = _poor_rows_anchor.merge(_fg_from_good, on="student_number", how="left")
_poor_rows_disp["disp_grade"] = _poor_rows_disp["fg_grade"].fillna(_poor_rows_disp["current_grade"])
_poor_rows_disp["disp_school"] = _poor_rows_disp["fg_school"].fillna(_poor_rows_disp["current_school"])
_poor_rows_disp_school = _poor_rows_disp.dropna(subset=["disp_school"])

# Per improver: grade in anchor year (last year before first good) vs grade in first good year.
# Used for proportion_of_improvers_by_grade: bar G = (G−1 → G) transitions vs all poor at G−1 in anchor.
_impr_pair = (
    _anchor.merge(
        _sg_lookup,
        left_on=["student_number", "anchor_year"],
        right_on=["student_number", "year"],
        how="inner",
    )
    .drop_duplicates("student_number", keep="first")[
        ["student_number", "anchor_year", "first_good_year", "current_grade"]
    ]
    .rename(columns={"current_grade": "anchor_grade"})
    .merge(
        _trans_first_good[["student_number", "current_grade"]].rename(
            columns={"current_grade": "fg_grade"}
        ),
        on="student_number",
        how="inner",
    )
)
_impr_pair["anchor_grade"] = pd.to_numeric(_impr_pair["anchor_grade"], errors="coerce")
_impr_pair["fg_grade"] = pd.to_numeric(_impr_pair["fg_grade"], errors="coerce")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Overall attendance bins through the district
# ═══════════════════════════════════════════════════════════════════════════════

_base1 = df[(df["school_membership"] > 119) & df["percent_days_attended"].between(0, 100)].copy()
_bins1 = np.arange(0, 105, 5)
_labels1 = [f"{i}-{i + 5}" for i in range(0, 100, 5)]
_base1["att_bucket"] = pd.cut(_base1["percent_days_attended"], bins=_bins1, labels=_labels1, include_lowest=True, right=True)
_counts1 = _base1.groupby("att_bucket", observed=True).size().reset_index(name="n")

fig, ax = plt.subplots(figsize=(11, 5))
sns.set_theme(style="whitegrid")
sns.barplot(data=_counts1, x="att_bucket", y="n", color="#4c72b0", ax=ax)
ax.set_xlabel("Percent Days Attended (5-Point Buckets)")
ax.set_ylabel("Students")
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

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.set_theme(style="whitegrid")

ax = axes[0, 0]
ax.plot(_agg_gpa["mid_att"], _agg_gpa["mean_gpa"], marker="o", color="#1f77b4")
ax.set_xlabel("Percent Days Attended")
ax.set_ylabel("Mean GPA")
ax.set_title("A. Mean GPA by Attendance")

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
ax.set_xlabel("Percent Days Attended")
ax.set_ylabel("Mean GPA")
ax.set_title("B. Mean GPA vs Attendance by Grade")
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
ax.set_xlabel("Percent Days Attended")
ax.set_ylabel("Mean GPA")
ax.set_title("C. Mean GPA vs Attendance by Gender")
ax.legend(fontsize=8)

ax = axes[1, 1]
if "ethnicity" in _base_gpa.columns:
    for _eth, _lbl in [("white", "White"), ("latinx", "Latinx")]:
        _sub = _base_gpa[_base_gpa["ethnicity"] == _eth].copy()
        if len(_sub) < 50:
            continue
        _sub["att_bin"] = pd.cut(_sub["percent_days_attended"], bins=_bins_gpa, include_lowest=True)
        _eagg = (
            _sub.groupby("att_bin", observed=True)
            .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_eagg["mid_att"], _eagg["mean_gpa"], marker=".", label=_lbl)
ax.set_xlabel("Percent Days Attended")
ax.set_ylabel("Mean GPA")
ax.set_title("D. Mean GPA vs Attendance by Ethnicity")
ax.legend(fontsize=7)

fig.suptitle("Relationship between Attendance and GPA", fontsize=18, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "attendance_gpa_hinge.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2b — ACT hinge: binned mean ACT composite score vs attendance (overall + subgroups)
# ═══════════════════════════════════════════════════════════════════════════════

_base_act = df[
    (df["school_membership"] > 119) &
    df["composite_score"].notna() & (df["composite_score"] > 0) &
    (df["current_grade"] >= 7)
].copy()
_base_act = _base_act[_base_act["percent_days_attended"].between(50, 100)]
_bins_act = np.linspace(_base_act["percent_days_attended"].min(), 100, 21)
_base_act["att_bin"] = pd.cut(_base_act["percent_days_attended"], bins=_bins_act, include_lowest=True)
_agg_act = (
    _base_act.groupby("att_bin", observed=True)
    .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"), n=("student_number", "count"))
    .reset_index()
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.set_theme(style="whitegrid")

ax = axes[0, 0]
ax.plot(_agg_act["mid_att"], _agg_act["mean_act"], marker="o", color="#1f77b4")
ax.set_xlabel("Percent Days Attended")
ax.set_ylabel("Mean ACT Composite Score")
ax.set_title("A. Mean ACT Score by Attendance")

ax = axes[0, 1]
_grades_act = sorted(_base_act["current_grade"].dropna().unique().astype(int).tolist())
_grades_act = [g for g in _grades_act if 1 <= g <= 12]
for _g in _grades_act[:8]:
    _sub = _base_act[_base_act["current_grade"] == _g].copy()
    if len(_sub) < 200:
        continue
    _sub["att_bin"] = pd.cut(_sub["percent_days_attended"], bins=_bins_act, include_lowest=True)
    _gagg = (
        _sub.groupby("att_bin", observed=True)
        .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"))
        .reset_index()
    )
    ax.plot(_gagg["mid_att"], _gagg["mean_act"], marker=".", label=f"Grade {_g}")
ax.set_xlabel("Percent Days Attended")
ax.set_ylabel("Mean ACT Composite Score")
ax.set_title("B. Mean ACT Score vs Attendance by Grade")
ax.legend(title="Grade", fontsize=7, ncol=2)

ax = axes[1, 0]
if "gender" in _base_act.columns:
    _gender_vals_act = _base_act["gender"].astype(str).str.strip().str.upper()
    for _gend in ["M", "F"]:
        _sub = _base_act[_gender_vals_act == _gend].copy()
        if len(_sub) < 300:
            continue
        _sub["att_bin"] = pd.cut(_sub["percent_days_attended"], bins=_bins_act, include_lowest=True)
        _gagg = (
            _sub.groupby("att_bin", observed=True)
            .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_gagg["mid_att"], _gagg["mean_act"], marker=".", label=_gend)
ax.set_xlabel("Percent Days Attended")
ax.set_ylabel("Mean ACT Composite Score")
ax.set_title("C. Mean ACT Score vs Attendance by Gender")
ax.legend(fontsize=8)

ax = axes[1, 1]
if "ethnicity" in _base_act.columns:
    for _eth, _lbl in [("white", "White"), ("latinx", "Latinx")]:
        _sub = _base_act[_base_act["ethnicity"] == _eth].copy()
        if len(_sub) < 50:
            continue
        _sub["att_bin"] = pd.cut(_sub["percent_days_attended"], bins=_bins_act, include_lowest=True)
        _eagg = (
            _sub.groupby("att_bin", observed=True)
            .agg(mean_act=("composite_score", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(_eagg["mid_att"], _eagg["mean_act"], marker=".", label=_lbl)
ax.set_xlabel("Percent Days Attended")
ax.set_ylabel("Mean ACT Composite Score")
ax.set_title("D. Mean ACT Score vs Attendance by Ethnicity")
ax.legend(fontsize=7)

fig.suptitle("Relationship between Attendance and ACT Score", fontsize=18, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "attendance_act_hinge.png", dpi=150, bbox_inches="tight")
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
ax.set_ylabel("Percentage of eligible student-years chronically absent")
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
# CHART 5a — Chronic absenteeism by ethnicity (with n sizes)
# ═══════════════════════════════════════════════════════════════════════════════

if "ethnicity" in _dbase.columns:
    _eth_agg5a = (
        _dbase[_dbase["ethnicity"].isin(CHART_ETHNICITY_ORDER)]
        .groupby("ethnicity", observed=True)
        .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
        .reindex(list(CHART_ETHNICITY_ORDER))
        .dropna()
        .reset_index()
    )
    _eth_agg5a["chronic_rate"] = _eth_agg5a["n_chronic"] / _eth_agg5a["n"]
    _eth_labels5a = [CHART_ETHNICITY_LABEL[k] for k in _eth_agg5a["ethnicity"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    _bars5a = ax.barh(_eth_labels5a, _eth_agg5a["chronic_rate"], color="#4c72b0")
    _max_rate5a = float(_eth_agg5a["chronic_rate"].max()) if not _eth_agg5a.empty else 0.1
    for _bar, _n in zip(_bars5a, _eth_agg5a["n"]):
        ax.text(
            _bar.get_width() + 0.002, _bar.get_y() + _bar.get_height() / 2,
            f"n={int(_n):,}", va="center", fontsize=8, color="#555555"
        )
    ax.set_xlabel("Percent Chronically Absent")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, _max_rate5a * 1.25)
    ax.set_title("Chronic Absenteeism Rate by Ethnicity", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chronic_absenteeism_by_ethnicity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote chronic_absenteeism_by_ethnicity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5b — Chronic absenteeism by gender and disability status
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.set_theme(style="whitegrid")

ax = axes[0]
if "gender" in _dbase.columns:
    _gvals5b = _dbase["gender"].astype(str).str.strip().str.upper()
    _glabels5b = {"M": "Male", "F": "Female"}
    _grps5b = [g for g in ["M", "F"] if (_gvals5b == g).any()]
    ax.bar(
        [_glabels5b[g] for g in _grps5b],
        [_chronic_rate(_dbase[_gvals5b == g]) for g in _grps5b],
        color="#c44e52",
    )
    ax.set_ylabel("Percent Chronically Absent")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("A. Chronic Absenteeism Rate by Gender")
    ax.tick_params(axis="x", rotation=20)
else:
    ax.set_visible(False)

ax = axes[1]
if "scram_membership" in _dbase.columns:
    _ord_dis = ["No Disability", "Disability"]
    _dis_rates = (
        _dbase.groupby("_disability_group", observed=True)["_chronic"]
        .mean()
        .reindex(_ord_dis)
        .dropna()
    )
    if not _dis_rates.empty:
        ax.bar(_dis_rates.index.astype(str), _dis_rates.values, color=["#4c72b0", "#c44e52"])
        ax.set_ylabel("Percent Chronically Absent")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_title("B. Chronic Absenteeism Rate by Disability Status")
        ax.tick_params(axis="x", rotation=15)
    else:
        ax.set_visible(False)
else:
    ax.text(0.5, 0.5, "Missing scram_membership", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

fig.suptitle(
    "Chronic Absenteeism by Gender and Disability Status",
    fontsize=16,
    y=1.02,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_absenteeism_by_gender_disability.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote chronic_absenteeism_by_gender_disability.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Chronic absenteeism rate by school (color-coded by school group)
# ═══════════════════════════════════════════════════════════════════════════════

_sagg = (
    _cbase[_cbase["_school"].notna()]
    .groupby("_school", dropna=True)
    .agg(n=("student_number", "count"), n_chronic=("_chronic", "sum"))
    .reset_index()
)
_sagg["chronic_rate"] = _sagg["n_chronic"] / _sagg["n"]
_sagg = _sagg[_sagg["n"] >= MIN_N]
_sagg = _sagg[~_sagg["_school"].str.strip().str.lower().isin(_SCHOOL_EXCLUDE)]
_sagg["school_group"] = _sagg["_school"].map(_group_label)
_sagg = _sagg.sort_values("chronic_rate", ascending=True)

fig, ax = plt.subplots(figsize=(9, max(4, len(_sagg) * 0.45)))
sns.set_theme(style="whitegrid")
_sbars = ax.barh(
    _sagg["_school"],
    _sagg["chronic_rate"],
    color=_sagg["school_group"].map(_SCHOOL_GROUP_COLORS),
)
for _bar, _n in zip(_sbars, _sagg["n"]):
    ax.text(
        _bar.get_width() + 0.002, _bar.get_y() + _bar.get_height() / 2,
        f"n={_n:,}", va="center", fontsize=7, color="#555555"
    )
ax.set_xlabel("Percentage of eligible student-years chronically absent")
ax.set_title(
    "Chronic absenteeism rate by school\n"
    f"(eligible = school_membership ≥ {MIN_MEMBERSHIP}; chronic = ≥{CHRONIC_ABSENT_PCT:.0f}% days missed)",
    fontsize=11,
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
_legend_patches = [
    mpatches.Patch(color=_SCHOOL_GROUP_COLORS[g], label=g) for g in _SCHOOL_GROUP_ORDER
]
ax.legend(handles=_legend_patches, title="School Group", fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "chronic_rate_by_school.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote chronic_rate_by_school.png")


# ── Drill-down: chronic absenteeism by school, one chart per school group ─────

for _tier6 in _SCHOOL_GROUP_ORDER:
    _sub6 = _sagg[_sagg["school_group"] == _tier6].sort_values("chronic_rate", ascending=True)
    if _sub6.empty:
        continue
    _slug6 = _tier6.lower().replace(" ", "_")
    fig, ax = plt.subplots(figsize=(9, max(4, len(_sub6) * 0.55)))
    sns.set_theme(style="whitegrid")
    _sbars6 = ax.barh(_sub6["_school"], _sub6["chronic_rate"], color=_SCHOOL_GROUP_COLORS[_tier6])
    for _bar, _n in zip(_sbars6, _sub6["n"]):
        ax.text(
            _bar.get_width() + 0.002, _bar.get_y() + _bar.get_height() / 2,
            f"n={_n:,}", va="center", fontsize=8, color="#555555"
        )
    ax.set_xlabel("Percentage of eligible student-years chronically absent")
    ax.set_title(f"Chronic Absenteeism Rate — {_tier6} Schools", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"chronic_rate_by_school_{_slug6}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote chronic_rate_by_school_{_slug6}.png")


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

_n_by_yrs8 = _df8.groupby("years_chronic_total")["student_number"].nunique()

fig, ax = plt.subplots(figsize=(12, 6))
sns.set_theme(style="whitegrid")
sns.boxplot(data=_df8, x="years_chronic_total", y="overall_gpa", palette="viridis", ax=ax)
ax.set_xticklabels(
    [f"{int(t.get_text())}\nn={_n_by_yrs8.get(int(t.get_text()), 0):,}" for t in ax.get_xticklabels()],
    fontsize=8,
)
ax.set_xlabel("Total Years Student was Chronically Absent")
ax.set_ylabel("Overall GPA")
ax.set_title("Total Years Chronically Absent vs. High School GPA", fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "Chronic Absenteeism.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote Chronic Absenteeism.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Improvers by first good-attendance grade (grade in year of first good attendance)
# ═══════════════════════════════════════════════════════════════════════════════

_m9g = _anc_grade[_anc_grade["current_grade"].between(1, 12)]
if not _m9g.empty:
    _counts9_imp = (
        _m9g.groupby("current_grade", as_index=False)
        .size()
        .rename(columns={"size": "n_improvers"})
        .sort_values("current_grade")
    )
    _poor_grade_num = pd.to_numeric(_poor_rows["current_grade"], errors="coerce")
    _poor_for_chart9 = _poor_rows.loc[_poor_grade_num.notna(), ["student_number"]].copy()
    _poor_for_chart9["grade"] = _poor_grade_num[_poor_grade_num.notna()].astype(int).values
    _counts9_poor = (
        _poor_for_chart9.groupby("grade", as_index=False)["student_number"]
        .nunique()
        .rename(columns={"student_number": "n_poor"})
    )
    _counts9 = (
        _counts9_imp.rename(columns={"current_grade": "grade"})
        .merge(_counts9_poor, on="grade", how="outer")
        .fillna(0)
        .sort_values("grade")
    )
    _counts9_long = _counts9.melt(
        id_vars="grade",
        value_vars=["n_improvers", "n_poor"],
        var_name="cohort",
        value_name="n",
    )
    _counts9_long["cohort"] = _counts9_long["cohort"].map(
        {
            "n_improvers": "Improvers",
            "n_poor": "Poor Attenders",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=_counts9_long,
        x="grade",
        y="n",
        hue="cohort",
        hue_order=["Improvers", "Poor Attenders"],
        palette=["#4c72b0", "#c44e52"],
        ax=ax,
    )
    ax.set_xlabel("Grade")
    ax.set_ylabel("Total Students")
    ax.set_title(
        "Improvers vs Poor Attenders by Grade"
    )
    ax.legend(title=None, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "improvers_count_by_grade.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote improvers_count_by_grade.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 10 — Proportion of improvers to poor attenders by grade
#   Bar at grade G:  (improvers with first-good grade G)
#                    / (all students with poor attendance in grade G).
# ═══════════════════════════════════════════════════════════════════════════════

_pc = pd.to_numeric(_poor_rows["current_grade"], errors="coerce")
_rows10: list[dict[str, float | int]] = []
for _g in range(2, 13):
    _n_poor_g = _poor_rows[_pc == _g]["student_number"].nunique()
    if _n_poor_g <= 0:
        continue
    _n_tr = _impr_pair[_impr_pair["fg_grade"] == _g]["student_number"].nunique()
    _rows10.append(
        {"grade": _g, "n": _n_tr / _n_poor_g, "n_trans": _n_tr, "n_poor": _n_poor_g}
    )
_grade_counts = pd.DataFrame(_rows10).sort_values("grade") if _rows10 else pd.DataFrame()

if not _grade_counts.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=_grade_counts, x="grade", y="n", color="#8172b3", ax=ax)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Improvers / Poor Attenders")
    ax.set_title("Proportion of Improvers to Poor Attenders by Grade")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "proportion_of_improvers_to_poor_attenders_by_grade.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote proportion_of_improvers_to_poor_attenders_by_grade.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 11 — Improvers by school (first good-attendance year)
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
    ax.set_xlabel("School")
    ax.set_ylabel("Total Improvers")
    ax.set_title(
        "Improvers by School"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "improvers_count_by_school.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote improvers_count_by_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 12 — Proportion of improvers by school (first good-attendance year)
# ═══════════════════════════════════════════════════════════════════════════════

_t12 = _anc_school.groupby("current_school", as_index=False)["student_number"].nunique().rename(
    columns={"student_number": "n_trans"},
)
_p12 = (
    _poor_rows_disp_school.groupby("disp_school", as_index=False)["student_number"]
    .nunique()
    .rename(columns={"disp_school": "current_school", "student_number": "n_poor"})
)
_school_counts = _t12.merge(_p12, on="current_school", how="outer").fillna(0)
_school_counts = _school_counts[_school_counts["n_poor"] > 0].copy()
_school_counts["n"] = _school_counts["n_trans"] / _school_counts["n_poor"]
_school_counts = _school_counts.sort_values("n", ascending=False)

if not _school_counts.empty:
    fig, ax = plt.subplots(figsize=(max(10.0, 0.38 * len(_school_counts)), 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=_school_counts, x="current_school", y="n", color="#4c72b0", ax=ax)
    ax.set_xlabel("School")
    ax.set_ylabel("Improvers / All Poor Attenders")
    ax.set_title("Proportion of Improvers by School")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "proportion_of_improvers_by_school.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote proportion_of_improvers_by_school.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 13 — Proportion of improvers by school group (first good-attendance year)
# ═══════════════════════════════════════════════════════════════════════════════

_n_trans13 = (
    _anc_school.assign(school_group=_anc_school["current_school"].map(_group_label))
    .groupby("school_group")["student_number"]
    .nunique()
    .rename("n_trans")
)
_n_poor13 = (
    _poor_rows_disp_school.assign(school_group=_poor_rows_disp_school["disp_school"].map(_group_label))
    .groupby("school_group")["student_number"]
    .nunique()
    .rename("n_poor")
)
_group_counts = pd.concat([_n_trans13, _n_poor13], axis=1).fillna({"n_trans": 0, "n_poor": 0}).reset_index()
_group_counts = _group_counts[_group_counts["n_poor"] > 0].copy()
_group_counts["n"] = _group_counts["n_trans"] / _group_counts["n_poor"]

if not _group_counts.empty:
    _group_counts["school_group"] = pd.Categorical(
        _group_counts["school_group"],
        categories=list(_SCHOOL_GROUP_ORDER),
        ordered=True,
    )
    _group_counts = _group_counts.sort_values("school_group")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=_group_counts, x="school_group", y="n", color="#55a868", ax=ax)
    ax.set_xlabel("School Group")
    ax.set_ylabel("Improvers / All Poor Attenders")
    ax.set_title(
        "Proportion of Improvers by School Group"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "proportion_of_improvers_by_school_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote proportion_of_improvers_by_school_group.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 14 — Top 10 teachers by improver percentage, split by school level (ES / MS / HS)
#   All poor→good improvers (any years-to-good), teacher links in *anchor* year
#   (first_good_year − 1). Poor pool: same anchor years, poor (<90) student-years
#   with a link. Supplement: mean % attendance for improvers in the *first good* year vs
#   all student–years on that teacher in the grid.
# ═══════════════════════════════════════════════════════════════════════════════

_TIER_RANK = {"Elementary": 0, "Middle School": 1, "High School": 2}
_TIER_SLUGS = {
    "Elementary": "elementary",
    "Middle School": "middle_school",
    "High School": "high_school",
}
_TEACHER_TOP_N = 10


def _teacher_school_tier(teacher_id: str, aff: dict[str, set[str]]) -> str:
    """Assign teacher to highest school tier (HS over MS over ES) from roster grids."""
    schs = aff.get(teacher_id, set())
    if not schs:
        return "High School"
    return max({_group_label(s) for s in schs}, key=lambda t: _TIER_RANK.get(t, 0))


if _tg_path.is_file() and not _anchor.empty:
    _tgrid = pd.read_csv(_tg_path)
    _tcols = [c for c in _tgrid.columns if c.startswith("teacher_")]
    _tgrid["student_number"] = _tgrid["student_number"].astype(str)
    _tgrid["year"] = pd.to_numeric(_tgrid["year"], errors="coerce")
    # Improvers with teacher–student links in anchor year (year before first good; any transition length)
    _exp = (
        _anchor.merge(
            _tgrid,
            left_on=["student_number", "anchor_year"],
            right_on=["student_number", "year"],
            how="inner",
        )
        .drop_duplicates(subset=["student_number"], keep="first")
    )
    if not _exp.empty and _tcols:
        _n_trans14 = _exp[_tcols].sum(axis=0).astype(int)
        # Poor attender pool in the same *anchor* calendar years (aligns with other improver charts)
        _poor_anch_t = _poor_rows[_poor_rows["year"].isin(_anchor["anchor_year"].unique())].drop_duplicates(
            subset=["student_number", "year"], keep="first"
        )[["student_number", "year"]]
        _merged_poor14 = _tgrid[["student_number", "year"] + _tcols].merge(
            _poor_anch_t, on=["student_number", "year"], how="inner"
        )
        _tc_poor = _merged_poor14[_tcols].apply(pd.to_numeric, errors="coerce")
        _n_poor14 = (_tc_poor.fillna(0) > 0).sum(axis=0).astype(int)
        _enrolled_sy14 = _base_imp[["student_number", "year"]].drop_duplicates()
        _merged_enrolled14 = _tgrid[["student_number", "year"] + _tcols].merge(_enrolled_sy14, on=["student_number", "year"], how="inner")
        _n_enrolled14 = (_merged_enrolled14[_tcols] > 0).sum(axis=0).astype(int)

        # Mean attendance: good-attendance student-years on teacher roster (>=95%) vs improvers
        # in *first good* year (teacher link is still at anchor; improver attendance uses first good year)
        _tgrid_att = _tgrid.merge(
            _base_imp[["student_number", "year", "percent_days_attended"]].drop_duplicates(
                subset=["student_number", "year"]
            ),
            on=["student_number", "year"],
            how="left",
        )
        _att_sy = _base_imp[["student_number", "year", "percent_days_attended"]].drop_duplicates(
            subset=["student_number", "year"], keep="first"
        )
        _exp_att = _exp.merge(
            _att_sy,
            left_on=["student_number", "first_good_year"],
            right_on=["student_number", "year"],
            how="left",
        )
        _exp_att = _exp_att.drop(columns=["year"], errors="ignore")

        def _class_good_mean_att(_col: str) -> float:
            _d = _tgrid_att[pd.to_numeric(_tgrid_att[_col], errors="coerce") >= 1.0]
            _d = _d[_d["percent_days_attended"] >= IMPROVER_GOOD_MIN_PCT]
            if _d.empty or _d["percent_days_attended"].isna().all():
                return float("nan")
            return float(_d["percent_days_attended"].mean())

        def _improver_mean_att(_col: str) -> float:
            _d = _exp_att[pd.to_numeric(_exp_att[_col], errors="coerce") >= 1.0]
            if _d.empty or _d["percent_days_attended"].isna().all():
                return float("nan")
            return float(_d["percent_days_attended"].mean())

        _tc = pd.DataFrame({
            "teacher":    _n_trans14.index.astype(str),
            "n_trans":    _n_trans14.values,
            "n_poor":     _n_poor14.reindex(_n_trans14.index).fillna(0).astype(int).values,
            "n_enrolled": _n_enrolled14.reindex(_n_trans14.index).fillna(0).astype(int).values,
        })
        _tc = _tc[_tc["n_trans"] > 0].copy()
        _tc["ratio"] = np.where(_tc["n_poor"] > 0, _tc["n_trans"] / _tc["n_poor"], np.nan)
        if not _tc.empty:
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

            _tc["school_tier"] = _tc["teacher"].map(
                lambda t: _teacher_school_tier(t, _affiliation)
            )

            for _tier in _SCHOOL_GROUP_ORDER:
                _sub = _tc[_tc["school_tier"] == _tier].copy()
                if _sub.empty:
                    continue
                if len(_sub) == 1:
                    _pool_t = _sub.sort_values(["ratio", "n_trans"], ascending=[False, False])
                else:
                    _avg_t = float(_sub["n_enrolled"].mean())
                    _pool_t = _sub[_sub["n_enrolled"] > _avg_t].copy()
                    if _pool_t.empty:
                        _pool_t = _sub.sort_values(["ratio", "n_trans"], ascending=[False, False])
                    else:
                        _pool_t = _pool_t.sort_values(["ratio", "n_trans"], ascending=[False, False])

                if _pool_t.empty:
                    continue

                _plot_t = _pool_t.head(_TEACHER_TOP_N).iloc[::-1].copy()
                _plot_t["ratio_plot"] = _plot_t["ratio"].fillna(0.0)
                _slug = _TIER_SLUGS[_tier]

                fig, ax = plt.subplots(figsize=(12, max(4.0, 0.45 * len(_plot_t))))
                sns.set_theme(style="whitegrid")
                sns.barplot(data=_plot_t, x="ratio_plot", y="teacher", color="#8172b3", ax=ax)
                ax.set_xlabel("Improvers / All Poor Attenders")
                ax.set_ylabel("Teacher")
                ax.set_title(
                    f"Top {_TEACHER_TOP_N} Teachers by Percentage of Improvers — {_tier}"
                )
                _max_x = float(_plot_t["ratio_plot"].max()) if not _plot_t.empty else 0.0
                _x_pad = max(0.02, 0.04 * _max_x) if _max_x > 0 else 0.05
                ax.set_xlim(0, min(1.0, _max_x + _x_pad) if _max_x <= 1.0 else _max_x + _x_pad)
                for _patch, _row14 in zip(ax.patches, _plot_t.itertuples(index=False)):
                    _schools14 = sorted(_affiliation.get(_row14.teacher, set()))
                    _school_text = ", ".join(_schools14) if _schools14 else "Unmapped"
                    _rtxt = f"{_row14.n_trans}/{_row14.n_poor}" if _row14.n_poor > 0 else f"{_row14.n_trans}/0"
                    _label = f"{_school_text}  ({_rtxt}, N={_row14.n_enrolled})"
                    _y_mid = _patch.get_y() + (_patch.get_height() / 2.0)
                    ax.text(_patch.get_width() + _x_pad, _y_mid, _label, va="center", ha="left", fontsize=7)
                fig.tight_layout()
                fig.savefig(
                    FIG_DIR / f"top_{_TEACHER_TOP_N}_teachers_by_percentage_of_improvers_{_slug}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
                print(f"Wrote top_{_TEACHER_TOP_N}_teachers_by_percentage_of_improvers_{_slug}.png")

                _supp_rows: list[dict[str, str | float]] = []
                for _row in _plot_t.itertuples(index=False):
                    _tcol = str(_row.teacher)
                    if _tcol not in _tcols:
                        continue
                    _mi = _improver_mean_att(_tcol)
                    _mc = _class_good_mean_att(_tcol)
                    if pd.isna(_mi) and pd.isna(_mc):
                        continue
                    _supp_rows.append({"teacher": _tcol, "cohort": "Improvers", "mean_att": _mi})
                    _supp_rows.append(
                        {
                            "teacher": _tcol,
                            "cohort": f"Good-Attendance Students with Teacher (>= {IMPROVER_GOOD_MIN_PCT:.0f}%)",
                            "mean_att": _mc,
                        }
                    )
                if _supp_rows:
                    _sdf = pd.DataFrame(_supp_rows)
                    _sdf = _sdf[_sdf["mean_att"].notna()].copy()
                    if not _sdf.empty:
                        _sdf["teacher"] = pd.Categorical(
                            _sdf["teacher"],
                            categories=_plot_t["teacher"].tolist(),
                            ordered=True,
                        )
                        _fig2, _ax2 = plt.subplots(figsize=(11, max(4.0, 0.5 * _plot_t["teacher"].nunique())))
                        sns.set_theme(style="whitegrid")
                        sns.barplot(
                            data=_sdf,
                            y="teacher",
                            x="mean_att",
                            hue="cohort",
                            palette=["#c44e52", "#4c72b0"],
                            ax=_ax2,
                        )
                        _ax2.set_xlabel("Mean Percent Days Attended")
                        _ax2.set_ylabel("Teacher")
                        _ax2.set_title(
                            f"Mean Attendance: Improvers vs Good-Attendance Students with Teacher — {_tier}"
                        )
                        _ax2.set_xlim(0, 105)
                        _ax2.legend(title=None, loc="lower right", fontsize=8)
                        _fig2.tight_layout()
                        _fig2.savefig(
                            FIG_DIR / f"teacher_improvers_vs_class_attendance_{_slug}.png",
                            dpi=150,
                            bbox_inches="tight",
                        )
                        plt.close(_fig2)
                        print(f"Wrote teacher_improvers_vs_class_attendance_{_slug}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 15 — Improvers count by years from first poor to first good year
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
    ax.set_title(
        "Improvers Count by Transition Length"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "improvers_count_by_transition_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote improvers_count_by_transition_length.png")

print(f"\nDone. Figures written to: {FIG_DIR}")
