# Q2: aggregate (all-schools) charts for attendance transitions.
# Includes counts by current_grade, counts by years from first poor to first good year,
# and counts of poor-attendance students by grade.
# Input: data/exploratory_data_all_grades.csv (expects student_number, year, current_school,
# percent_days_attended, school_membership, current_grade or grade).
# Rows with school_membership < 60 are excluded. Only students with a poor year (<90% attendance)
# and a later good year (>=95% attendance) are kept.

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "exploratory_data_all_grades.csv"
FIG_DIR = ROOT / "figures" / "Q2_Schools"

# Poor / good thresholds (percent_days_attended); good year must be after first poor year.
POOR_PCT_LT = 90.0
GOOD_PCT_GE = 95.0


def _grade_column(df: pd.DataFrame) -> str:
    if "current_grade" in df.columns:
        return "current_grade"
    if "grade" in df.columns:
        return "grade"
    raise KeyError(
        "Need a grade column: 'current_grade' or 'grade'. "
        f"Columns found: {list(df.columns)}"
    )


def _school_slug(name: object) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(name).strip().strip('"'), flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")[:80]
    return s or "unknown"


def _expected_grade_bounds(school_name: str) -> tuple[int, int]:
    """Return inclusive (low, high) grade for charting by school name."""
    s = str(school_name).lower()
    if "elem." in s:
        return (0, 6)
    if "middle" in s:
        return (7, 8)
    return (9, 12)


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _improver_transition_years(df: pd.DataFrame) -> pd.DataFrame:
    """One row per improver: first_poor_year, first_good_year (membership >= 60 rows only)."""
    d = df[["student_number", "year", "percent_days_attended", "school_membership"]].copy()
    d["student_number"] = d["student_number"].astype(str)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["percent_days_attended"] = pd.to_numeric(d["percent_days_attended"], errors="coerce")
    d["school_membership"] = pd.to_numeric(d["school_membership"], errors="coerce")
    d = d[d["school_membership"] >= 60]
    d = d.dropna(subset=["year", "percent_days_attended"])
    poor = d["percent_days_attended"] < POOR_PCT_LT
    good = d["percent_days_attended"] >= GOOD_PCT_GE
    first_poor = d.loc[poor].groupby("student_number")["year"].min()
    first_good = d.loc[good].groupby("student_number")["year"].min()
    trans = pd.DataFrame({"first_poor_year": first_poor, "first_good_year": first_good}).dropna()
    trans = trans[trans["first_good_year"] > trans["first_poor_year"]]
    trans.index = trans.index.astype(str)
    return trans


def _improver_student_ids(df: pd.DataFrame) -> set[str]:
    """Students with at least one year with pct < POOR_PCT_LT and a later year with pct >= GOOD_PCT_GE.

    Only school years with school_membership >= 60 are used to detect poor/good (same rule as the charts).
    """
    return set(_improver_transition_years(df).index)


def _improver_first_good_year_details(df: pd.DataFrame) -> pd.DataFrame:
    """Per improver: first year reaching good attendance, grade and school on that year-row."""
    trans = _improver_transition_years(df)
    if trans.empty:
        return pd.DataFrame(
            columns=["student_number", "first_good_year", "current_grade", "current_school"]
        )

    gcol = _grade_column(df)
    d = df[
        ["student_number", "year", "percent_days_attended", "school_membership", "current_school", gcol]
    ].copy()
    d["student_number"] = d["student_number"].astype(str)
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d[gcol] = pd.to_numeric(d[gcol], errors="coerce")
    d["school_membership"] = pd.to_numeric(d["school_membership"], errors="coerce")
    d["current_school"] = d["current_school"].astype(str).str.strip()
    d = d[d["school_membership"] >= 60]
    d = d.dropna(subset=["year"])

    m = trans.reset_index().merge(d, on="student_number", how="inner")
    m = m[m["year"] == m["first_good_year"]]
    m = m.sort_values(["student_number", "year"]).drop_duplicates("student_number", keep="first")
    out = m[["student_number", "first_good_year", gcol, "current_school"]].rename(
        columns={gcol: "current_grade"}
    )
    return out


def _improver_transition_span_details(df: pd.DataFrame) -> pd.DataFrame:
    """Per improver: years from first poor year to first good year; school at first good year."""
    trans = _improver_transition_years(df)
    if trans.empty:
        return pd.DataFrame(columns=["student_number", "years_to_good", "current_school"])

    details = _improver_first_good_year_details(df)
    if details.empty:
        return pd.DataFrame(columns=["student_number", "years_to_good", "current_school"])

    t = trans.reset_index()[["student_number", "first_poor_year", "first_good_year"]]
    # Avoid duplicate first_good_year columns from merge suffixes
    merged = details.drop(columns=["first_good_year"]).merge(t, on="student_number", how="inner")
    merged["years_to_good"] = merged["first_good_year"] - merged["first_poor_year"]
    return merged[["student_number", "years_to_good", "current_school"]]


def chart_mean_attendance_by_grade_per_school(df: pd.DataFrame) -> None:
    need = {"student_number", "year", "current_school", "percent_days_attended", "school_membership"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {missing}. Have: {list(df.columns)}")

    improvers = _improver_student_ids(df)
    if not improvers:
        print("No students with poor (<90%) then later good (>=95%) attendance; no charts written.", file=sys.stderr)
        return

    gcol = _grade_column(df)
    d = df[list(need | {gcol})].copy()
    d["student_number"] = d["student_number"].astype(str)
    d = d[d["student_number"].isin(improvers)]
    d["percent_days_attended"] = pd.to_numeric(d["percent_days_attended"], errors="coerce")
    d["school_membership"] = pd.to_numeric(d["school_membership"], errors="coerce")
    d[gcol] = pd.to_numeric(d[gcol], errors="coerce")
    d["current_school"] = d["current_school"].astype(str).str.strip()
    d = d[d["current_school"].notna() & (d["current_school"] != "") & (d["current_school"] != "nan")]
    d = d[d["school_membership"] >= 60]
    d = d.dropna(subset=[gcol, "percent_days_attended"])

    schools = sorted(d["current_school"].unique())
    if not schools:
        print("No rows with valid current_school and grade; no charts written.", file=sys.stderr)
        return

    sns.set_theme(style="whitegrid")

    for school in schools:
        sub = d[d["current_school"] == school]
        lo, hi = _expected_grade_bounds(school)
        sub = sub[(sub[gcol] >= lo) & (sub[gcol] <= hi)]
        agg = (
            sub.groupby(gcol, dropna=True)["percent_days_attended"]
            .mean()
            .reset_index()
            .rename(columns={gcol: "grade", "percent_days_attended": "mean_pct"})
        )
        agg = agg.sort_values("grade")
        if agg.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=agg, x="grade", y="mean_pct", color="#4c72b0", ax=ax)
        ax.set_xlabel("Grade")
        ax.set_ylabel("Mean % days attended")
        ax.set_title(
            f"Mean attendance by grade (poor→good improvers) — {school}\n"
            f"(< {POOR_PCT_LT:g}% then ≥ {GOOD_PCT_GE:g}% on a later year)"
        )
        ax.set_ylim(0, 105)
        slug = _school_slug(school)
        _save(fig, f"eda_q2_improvers_mean_pct_by_grade_{slug}.png")


def _plot_improver_count_by_current_grade(
    counts: pd.DataFrame, title: str, outfile: str
) -> None:
    """Bar chart: x = current_grade, y = n students (one row per grade)."""
    if counts.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="current_grade", y="n", color="#4c72b0", ax=ax)
    ax.set_xlabel("current_grade")
    ax.set_ylabel("Students (count)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, outfile)


def chart_improver_count_by_current_grade_per_school(df: pd.DataFrame) -> None:
    """Per school: total improvers by current_grade (grade in first good year), all years combined."""
    details = _improver_first_good_year_details(df)
    if details.empty:
        print(
            "No students with poor (<90%) then later good (≥95%) attendance; "
            "no improver-count-by-grade charts written.",
            file=sys.stderr,
        )
        return

    details = details.copy()
    details = details[
        details["current_school"].notna()
        & (details["current_school"] != "")
        & (details["current_school"] != "nan")
    ]
    details = details.dropna(subset=["current_grade"])
    if details.empty:
        print("No improver rows with current_grade on first good year; no charts.", file=sys.stderr)
        return

    sns.set_theme(style="whitegrid")

    for school in sorted(details["current_school"].unique()):
        sub = details[details["current_school"] == school]
        lo, hi = _expected_grade_bounds(school)
        sub = sub[(sub["current_grade"] >= lo) & (sub["current_grade"] <= hi)]
        if sub.empty:
            continue

        counts = (
            sub.groupby("current_grade", dropna=True)
            .size()
            .reset_index(name="n")
            .sort_values("current_grade")
        )
        slug = _school_slug(school)
        _plot_improver_count_by_current_grade(
            counts,
            title=(
                f"Improvers by current_grade (first good year) — {school}\n"
                f"(total across all years; membership ≥ 60; < {POOR_PCT_LT:g}% then ≥ {GOOD_PCT_GE:g}%)"
            ),
            outfile=f"eda_q2_improvers_count_by_current_grade_{slug}.png",
        )


def chart_improver_count_by_current_grade_all_schools(df: pd.DataFrame) -> None:
    """District-wide: same as per-school, but all students summed across schools (one chart)."""
    details = _improver_first_good_year_details(df)
    if details.empty:
        return

    details = details.copy()
    details = details[
        details["current_school"].notna()
        & (details["current_school"] != "")
        & (details["current_school"] != "nan")
    ]
    details = details.dropna(subset=["current_grade"])
    if details.empty:
        return

    details = details[(details["current_grade"] >= 0) & (details["current_grade"] <= 12)]
    counts = (
        details.groupby("current_grade", dropna=True)
        .size()
        .reset_index(name="n")
        .sort_values("current_grade")
    )
    _plot_improver_count_by_current_grade(
        counts,
        title=(
            "Improvers by current_grade (first good year) — all schools\n"
            f"(total across all years; membership ≥ 60; < {POOR_PCT_LT:g}% then ≥ {GOOD_PCT_GE:g}%)"
        ),
        outfile="eda_q2_improvers_count_by_current_grade_all_schools.png",
    )


def _plot_improver_count_by_transition_years(
    counts: pd.DataFrame, title: str, outfile: str
) -> None:
    """Bar chart: x = years between first poor and first good year, y = student count."""
    if counts.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    order = sorted(counts["years_to_good"].unique())
    sns.barplot(data=counts, x="years_to_good", y="n", color="#55a868", order=order, ax=ax)
    ax.set_xlabel("Years from first poor year to first good year (first occurrence of each)")
    ax.set_ylabel("Students (count)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, outfile)


def chart_improver_count_by_transition_years_per_school(df: pd.DataFrame) -> None:
    """Per school: histogram of transition length (calendar years) using first poor vs first good year."""
    details = _improver_transition_span_details(df)
    if details.empty:
        print(
            "No improver transition spans; no transition-years charts written.",
            file=sys.stderr,
        )
        return

    details = details.copy()
    details = details[
        details["current_school"].notna()
        & (details["current_school"] != "")
        & (details["current_school"] != "nan")
    ]
    details = details.dropna(subset=["years_to_good"])
    if details.empty:
        return

    for school in sorted(details["current_school"].unique()):
        sub = details[details["current_school"] == school]
        if sub.empty:
            continue
        counts = (
            sub.groupby("years_to_good", dropna=True)
            .size()
            .reset_index(name="n")
            .sort_values("years_to_good")
        )
        slug = _school_slug(school)
        _plot_improver_count_by_transition_years(
            counts,
            title=(
                f"Improvers by transition length — {school}\n"
                f"(first good year − first poor year; membership ≥ 60; "
                f"< {POOR_PCT_LT:g}% then ≥ {GOOD_PCT_GE:g}%)"
            ),
            outfile=f"eda_q2_improvers_count_by_transition_years_{slug}.png",
        )


def chart_improver_count_by_transition_years_all_schools(df: pd.DataFrame) -> None:
    """District-wide: same transition-length distribution, all schools combined."""
    details = _improver_transition_span_details(df)
    if details.empty:
        return

    details = details.copy()
    details = details[
        details["current_school"].notna()
        & (details["current_school"] != "")
        & (details["current_school"] != "nan")
    ]
    details = details.dropna(subset=["years_to_good"])
    if details.empty:
        return

    counts = (
        details.groupby("years_to_good", dropna=True)
        .size()
        .reset_index(name="n")
        .sort_values("years_to_good")
    )
    _plot_improver_count_by_transition_years(
        counts,
        title=(
            "Improvers by transition length — all schools\n"
            f"(first good year − first poor year; membership ≥ 60; "
            f"< {POOR_PCT_LT:g}% then ≥ {GOOD_PCT_GE:g}%)"
        ),
        outfile="eda_q2_improvers_count_by_transition_years_all_schools.png",
    )


def chart_poor_attendance_count_by_grade_all_schools(df: pd.DataFrame) -> None:
    """All schools: count students with poor attendance (<90%) by grade."""
    gcol = _grade_column(df)
    need = {"student_number", "percent_days_attended", "school_membership", gcol}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {missing}. Have: {list(df.columns)}")

    d = df[list(need)].copy()
    d["student_number"] = d["student_number"].astype(str)
    d["percent_days_attended"] = pd.to_numeric(d["percent_days_attended"], errors="coerce")
    d["school_membership"] = pd.to_numeric(d["school_membership"], errors="coerce")
    d[gcol] = pd.to_numeric(d[gcol], errors="coerce")
    d = d[d["school_membership"] >= 60]
    d = d.dropna(subset=[gcol, "percent_days_attended"])
    d = d[d["percent_days_attended"] < POOR_PCT_LT]
    if d.empty:
        return

    # Count rows within each grade bucket across all years/schools.
    counts = (
        d.groupby(gcol, dropna=True)
        .size()
        .reset_index(name="n")
        .rename(columns={gcol: "current_grade"})
        .sort_values("current_grade")
    )
    _plot_improver_count_by_current_grade(
        counts,
        title=(
            "Students with poor attendance by grade — all schools\n"
            f"(percent_days_attended < {POOR_PCT_LT:g}%; membership ≥ 60)"
        ),
        outfile="eda_q2_poor_attendance_count_by_current_grade_all_schools.png",
    )


def chart_poor_attendance_share_by_grade_all_schools(df: pd.DataFrame) -> None:
    """All schools: poor-attendance row share by grade (poor rows / total rows in grade)."""
    gcol = _grade_column(df)
    need = {"percent_days_attended", "school_membership", gcol}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {missing}. Have: {list(df.columns)}")

    d = df[list(need)].copy()
    d["percent_days_attended"] = pd.to_numeric(d["percent_days_attended"], errors="coerce")
    d["school_membership"] = pd.to_numeric(d["school_membership"], errors="coerce")
    d[gcol] = pd.to_numeric(d[gcol], errors="coerce")
    d = d[d["school_membership"] >= 60]
    d = d.dropna(subset=[gcol, "percent_days_attended"])
    if d.empty:
        return

    total_by_grade = d.groupby(gcol, dropna=True).size().rename("total_n")
    poor_by_grade = d[d["percent_days_attended"] < POOR_PCT_LT].groupby(gcol, dropna=True).size().rename("poor_n")
    shares = (
        total_by_grade.to_frame()
        .join(poor_by_grade, how="left")
        .fillna({"poor_n": 0})
        .reset_index()
        .rename(columns={gcol: "current_grade"})
        .sort_values("current_grade")
    )
    shares["poor_pct"] = 100.0 * shares["poor_n"] / shares["total_n"]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=shares, x="current_grade", y="poor_pct", color="#c44e52", ax=ax)
    ax.set_xlabel("current_grade")
    ax.set_ylabel("Poor-attendance rows (%)")
    ax.set_title(
        "Share of poor-attendance rows by grade — all schools\n"
        f"(poor if percent_days_attended < {POOR_PCT_LT:g}%; membership ≥ 60)"
    )
    fig.tight_layout()
    _save(fig, "eda_q2_poor_attendance_share_by_current_grade_all_schools.png")


def main() -> None:
    if not DATA_PATH.is_file():
        print(f"Expected data file not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    chart_improver_count_by_current_grade_all_schools(df)
    chart_improver_count_by_transition_years_all_schools(df)
    chart_poor_attendance_count_by_grade_all_schools(df)
    chart_poor_attendance_share_by_grade_all_schools(df)


if __name__ == "__main__":
    main()
