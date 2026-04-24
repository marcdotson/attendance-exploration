# Exploratory charts: attendance–GPA hinge, 5-point attendance histogram, chronic absenteeism by demographics.
# Requires data/exploratory_data_all_grades.csv (run after upstream table scripts).

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "exploratory_data_all_grades.csv"
FIG_DIR = ROOT / "figures"

# ACT scale (composite and section scores); invalid / placeholder values outside this range → NaN
ACT_SCORE_COLUMNS = (
    "composite_score",
    "english_score",
    "math_score",
    "reading_score",
    "science_score",
    "writing_score",
)

ETHNICITY_DUMMY_COLUMNS = (
    "amerindian_alaskan",
    "asian",
    "black_african_amer",
    "hawaiian_pacific_isl",
    "white",
)
LATINX_INDICATOR_COLUMN = "ethnicity"

# Six groups from _ethnicity_label (dummy columns + latinx); used for demographic charts
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


def _valid_gpa_mask(d: pd.DataFrame) -> pd.Series:
    """Rows with usable GPA for GPA-based analyses."""
    gpa = pd.to_numeric(d["overall_gpa"], errors="coerce")
    grade = pd.to_numeric(d["current_grade"], errors="coerce")
    return gpa.notna() & (gpa > 0) & (grade >= 7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize types and derived fields used across analyses."""
    out = df.copy()
    out["student_number"] = out["student_number"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    for c in (
        "percent_days_attended",
        "overall_gpa",
        "current_grade",
        "days_attended",
        "days_absent",
        "school_membership",
        "composite_score",
        "english_score",
        "math_score",
        "reading_score",
        "science_score",
        "writing_score",
        "ac_ind",
        "ac_count",
        "extracurricular_ind",
        "school_count",
        "scram_membership",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ACT_SCORE_COLUMNS:
        if c not in out.columns:
            continue
        bad = ~out[c].between(0.0, 36.0, inclusive="both")
        out.loc[bad, c] = np.nan

    # Valid enrollment rows (avoid divide-by-zero artifacts from upstream fill rules)
    sm = out["school_membership"].fillna(0)
    out["_valid_enroll"] = sm > 119

    out["_pct_absent"] = np.where(
        sm > 0, 100.0 * out["days_absent"].fillna(0) / sm, np.nan
    )
    # K–12 chronic absenteeism: missing >= 10% of enrolled days
    out["_chronic"] = (out["_valid_enroll"]) & (out["_pct_absent"] >= 10.0)

    def _is_positive(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return value == 1
        return str(value).strip().lower() in {"1", "y", "yes", "true", "t"}

    def _ethnicity_label(row: pd.Series) -> str:
        # If upstream ethnicity is a Latinx indicator, prioritize that mapping.
        if LATINX_INDICATOR_COLUMN in row.index and _is_positive(row.get(LATINX_INDICATOR_COLUMN)):
            return "latinx"
        for c in ETHNICITY_DUMMY_COLUMNS:
            if c in row.index and _is_positive(row.get(c)):
                return c

        raw = row.get(LATINX_INDICATOR_COLUMN, np.nan)
        if pd.isna(raw):
            return "unknown"
        raw_str = str(raw).strip()
        return raw_str if raw_str else "unknown"

    if LATINX_INDICATOR_COLUMN in out.columns or any(c in out.columns for c in ETHNICITY_DUMMY_COLUMNS):
        out["ethnicity"] = out.apply(_ethnicity_label, axis=1)

    return out


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Q1 — Hinge points: attendance vs achievement (overall + subgroups)
# ---------------------------------------------------------------------------


def chart_q1_hinge(df: pd.DataFrame) -> None:
    d = _prep(df)
    base = d[d["_valid_enroll"] & _valid_gpa_mask(d)].copy()
    base = base[(base["percent_days_attended"] >= 50) & (base["percent_days_attended"] <= 100)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.set_theme(style="whitegrid")

    # A) Binned mean GPA vs attendance
    n_bins = 20
    bins = np.linspace(base["percent_days_attended"].min(), 100, n_bins + 1)
    base["att_bin"] = pd.cut(base["percent_days_attended"], bins=bins, include_lowest=True)
    agg = (
        base.groupby("att_bin", observed=True)
        .agg(
            mean_gpa=("overall_gpa", "mean"),
            mid_att=("percent_days_attended", "mean"),
            n=("student_number", "count"),
        )
        .reset_index()
    )
    ax = axes[0, 0]
    ax.plot(agg["mid_att"], agg["mean_gpa"], marker="o", color="#1f77b4")
    ax.set_xlabel("Percent Days Attended")
    ax.set_ylabel("Mean GPA")
    ax.set_title("A. Mean GPA by Attendance")

    # B) By grade (current_grade)
    ax = axes[0, 1]
    grades = sorted(base["current_grade"].dropna().unique().astype(int).tolist())
    grades = [g for g in grades if 1 <= g <= 12]
    for g in grades[:8]:
        sub = base[base["current_grade"] == g]
        if len(sub) < 200:
            continue
        sub = sub.copy()
        sub["att_bin"] = pd.cut(sub["percent_days_attended"], bins=bins, include_lowest=True)
        gagg = (
            sub.groupby("att_bin", observed=True)
            .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
            .reset_index()
        )
        ax.plot(gagg["mid_att"], gagg["mean_gpa"], marker=".", label=f"Grade {g}")
    ax.set_xlabel("Percent Days Attended")
    ax.set_ylabel("Mean GPA")
    ax.set_title("B. Mean GPA vs Attendance by Grade")
    ax.legend(title="Grade", fontsize=7, ncol=2)

    # C) Gender (if present)
    ax = axes[1, 0]
    if "gender" in base.columns:
        gender_vals = base["gender"].astype(str).str.strip().str.upper()
        for gend in ["M", "F"]:
            sub = base[gender_vals == gend]
            if len(sub) < 300:
                continue
            sub = sub.copy()
            sub["att_bin"] = pd.cut(sub["percent_days_attended"], bins=bins, include_lowest=True)
            gagg = (
                sub.groupby("att_bin", observed=True)
                .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
                .reset_index()
            )
            ax.plot(gagg["mid_att"], gagg["mean_gpa"], marker=".", label=gend)
    ax.set_xlabel("Percent Days Attended")
    ax.set_ylabel("Mean GPA")
    ax.set_title("C. Mean GPA vs Attendance by Gender")
    ax.legend(fontsize=8)

    # D) Ethnicity — White and Latinx only
    ax = axes[1, 1]
    if "ethnicity" in base.columns:
        eth_labels = {"white": "White", "latinx": "Latinx"}
        for eth in ["white", "latinx"]:
            sub = base[base["ethnicity"] == eth]
            if len(sub) < 50:
                continue
            sub = sub.copy()
            sub["att_bin"] = pd.cut(sub["percent_days_attended"], bins=bins, include_lowest=True)
            gagg = (
                sub.groupby("att_bin", observed=True)
                .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
                .reset_index()
            )
            ax.plot(gagg["mid_att"], gagg["mean_gpa"], marker=".", label=eth_labels[eth])
    ax.set_xlabel("Percent Days Attended")
    ax.set_ylabel("Mean GPA")
    ax.set_title("D. Mean GPA vs Attendance by Ethnicity")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Relationship between Attendance and GPA",
        fontsize=18,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "attendance_gpa_hinge.png")


def chart_q1_attendance_histogram(df: pd.DataFrame) -> None:
    """Q1 supplement: count rows by 5-point percent_days_attended buckets."""
    d = _prep(df)
    base = d[d["_valid_enroll"]].copy()
    base = base.dropna(subset=["percent_days_attended"])
    base = base[(base["percent_days_attended"] >= 0) & (base["percent_days_attended"] <= 100)]
    if base.empty:
        print("Skipping Q1 attendance histogram (no valid percent_days_attended rows).")
        return

    # Fixed 5% buckets from 0 to 100.
    bins = np.arange(0, 105, 5)
    labels = [f"{i}-{i + 5}" for i in range(0, 100, 5)]
    base["att_bucket"] = pd.cut(
        base["percent_days_attended"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    counts = (
        base.groupby("att_bucket", observed=True)
        .size()
        .reset_index(name="n")
    )
    if counts.empty:
        print("Skipping Q1 attendance histogram (no rows after bucketing).")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=counts, x="att_bucket", y="n", color="#4c72b0", ax=ax)
    ax.set_xlabel("Percent Days Attended (5-Point Buckets)")
    ax.set_ylabel("Students")
    ax.set_title("Student Count by Attendance Bucket")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, "attendance_histogram.png")

# ---------------------------------------------------------------------------
# Q3 — Chronic absenteeism by demographics
# ---------------------------------------------------------------------------


def chart_q3_demographics(df: pd.DataFrame) -> None:
    """Chronic Absenteeism by Demographics"""
    d = _prep(df)
    base = d[d["_valid_enroll"]].copy()

    def chronic_rate(sub: pd.DataFrame) -> float:
        return float(sub["_chronic"].mean()) if len(sub) else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # A) Chronic rate by ethnicity (fixed six groups; labels for display only)
    ax = axes[0]
    if "ethnicity" in base.columns:
        rates = [chronic_rate(base[base["ethnicity"] == k]) for k in CHART_ETHNICITY_ORDER]
        labels = [CHART_ETHNICITY_LABEL[k] for k in CHART_ETHNICITY_ORDER]
        ax.barh(labels, rates, color="#4c72b0")
        ax.set_xlabel("Percent Chronically Absent")
        ax.set_title("A. Chronic Absenteeism Rate by Ethnicity")
    else:
        ax.set_visible(False)

    # B) Chronic rate by gender
    ax = axes[1]
    if "gender" in base.columns:
        gender_vals = base["gender"].astype(str).str.strip().str.upper()
        gender_labels = {"M": "Male", "F": "Female"}
        grp = [g for g in ["M", "F"] if (gender_vals == g).any()]
        rates = [chronic_rate(base[gender_vals == g]) for g in grp]
        ax.bar([gender_labels[g] for g in grp], rates, color="#c44e52")
        ax.set_ylabel("Percent Chronically Absent")
        ax.set_title("B. Chronic Absenteeism Rate by Gender")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.set_visible(False)

    # C) Chronic rate by disability status
    ax = axes[2]
    if "scram_membership" in base.columns:
        base = base.copy()
        base["_disability_group"] = np.where(
            base["scram_membership"].fillna(0) > 0,
            "Disability",
            "No Disability",
        )
        order = ["No Disability", "Disability"]
        chronic_rates = (
            base.groupby("_disability_group", observed=True)["_chronic"]
            .mean()
            .reindex(order)
            .dropna()
        )
        if not chronic_rates.empty:
            ax.bar(chronic_rates.index.astype(str), chronic_rates.values, color=["#4c72b0", "#c44e52"])
            ax.set_ylabel("Percent Chronically Absent")
            ax.set_title("C. Chronic Absenteeism Rate by Disability Status")
            ax.tick_params(axis="x", rotation=15)
        else:
            ax.set_visible(False)
    else:
        ax.text(0.5, 0.5, "Missing scram_membership", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(
        "Chronic Absenteeism by Demographics",
        fontsize=18,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "chronic_absenteeism_by_demographics.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not DATA_PATH.is_file():
        print(
            f"Missing {DATA_PATH}. Run upstream table scripts and the all-grades combine step first.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows from {DATA_PATH}")

    chart_q1_hinge(df)
    chart_q1_attendance_histogram(df)
    chart_q3_demographics(df)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
