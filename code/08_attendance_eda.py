# Exploratory charts for attendance research questions.
# Requires data/exploratory_data.csv from 07_combine_data-table.py (run after upstream table scripts).

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "exploratory_data.csv"
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
    out["_valid_enroll"] = sm > 30

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


def _hinge_indices(x_centers: np.ndarray, y: np.ndarray) -> list[int]:
    """Return bin indices where discrete slope of y vs x is largest (simple hinge heuristic)."""
    x_centers = np.asarray(x_centers, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x_centers) & np.isfinite(y)
    x_centers, y = x_centers[ok], y[ok]
    if len(y) < 4:
        return []
    dy = np.diff(y)
    dx = np.diff(x_centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = dy / np.where(dx == 0, np.nan, dx)
    # Largest negative slope magnitude = steepest drop in achievement with worsening attendance
    idx = int(np.nanargmin(slope)) if np.any(np.isfinite(slope)) else -1
    out = []
    if 0 <= idx < len(slope):
        out.append(idx)
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
    base = d[d["_valid_enroll"] & d["overall_gpa"].notna()].copy()
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
    hp = _hinge_indices(agg["mid_att"].values, agg["mean_gpa"].values)
    for hi in hp:
        if hi + 1 < len(agg):
            ax.axvline(agg["mid_att"].iloc[hi + 1], color="crimson", linestyle="--", alpha=0.7)
    ax.axvline(90, color="gray", linestyle=":", alpha=0.8, label="90% attendance (~10% absent)")
    ax.set_xlabel("Percent days attended (bin center)")
    ax.set_ylabel("Mean cumulative GPA")
    ax.set_title("A. Mean GPA by attendance (20 equal-width bins)")
    ax.legend(loc="lower right", fontsize=8)

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
    ax.axvline(90, color="gray", linestyle=":", alpha=0.8)
    ax.set_xlabel("Percent days attended (bin center)")
    ax.set_ylabel("Mean GPA")
    ax.set_title("B. Mean GPA vs attendance by grade")
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
    ax.axvline(90, color="gray", linestyle=":", alpha=0.8)
    ax.set_xlabel("Percent days attended (bin center)")
    ax.set_ylabel("Mean GPA")
    ax.set_title("C. Mean GPA vs attendance by gender")
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
    ax.axvline(90, color="gray", linestyle=":", alpha=0.8)
    ax.set_xlabel("Percent days attended (bin center)")
    ax.set_ylabel("Mean GPA")
    ax.set_title("D. Mean GPA vs attendance by ethnicity (White and Latinx)")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Q1 — Non-linear relationship between attendance and GPA (hinge ≈ steepest slope segment)\n"
        "Red dashed line marks a data-driven elbow on the overall curve; gray line = 90% attendance (~10% absent).",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "eda_q1_attendance_gpa_hinge.png")


def chart_q1_act(df: pd.DataFrame) -> None:
    d = _prep(df)
    base = d[d["_valid_enroll"]].copy()
    base = base[(base["percent_days_attended"] >= 50) & (base["percent_days_attended"] <= 100)]
    # ACT composite: 0 often means no test in this pipeline
    base = base[base["composite_score"] > 1]

    if len(base) < 500:
        print("Skipping ACT hinge chart (too few rows with composite_score > 1).")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = 16
    bins = np.linspace(base["percent_days_attended"].min(), 100, n_bins + 1)
    base["att_bin"] = pd.cut(base["percent_days_attended"], bins=bins, include_lowest=True)
    agg = (
        base.groupby("att_bin", observed=True)
        .agg(
            mean_act=("composite_score", "mean"),
            mid_att=("percent_days_attended", "mean"),
        )
        .reset_index()
    )
    ax.plot(agg["mid_att"], agg["mean_act"], marker="o", color="#2ca02c")
    ax.axvline(90, color="gray", linestyle=":", alpha=0.8)
    ax.set_xlabel("Percent days attended (bin center)")
    ax.set_ylabel("Mean ACT composite")
    ax.set_title("Q1 (supplement) — Mean ACT composite vs attendance (test-takers only)")
    fig.tight_layout()
    _save(fig, "eda_q1_attendance_act_hinge.png")


# ---------------------------------------------------------------------------
# Q2 — Poor → good attendance: correlates (school change, grade, advanced courses)
# ---------------------------------------------------------------------------


def _q2_school_changed_between_rows(r0: pd.Series, r1: pd.Series) -> bool:
    """True if current_school differs between two year-rows (same rule as Q2 improver logic)."""
    return (
        str(r0.get("current_school", "")) != str(r1.get("current_school", ""))
        if pd.notna(r0.get("current_school")) or pd.notna(r1.get("current_school"))
        else False
    )


def chart_q2_transitions(df: pd.DataFrame) -> None:
    d = _prep(df)
    d = d[d["_valid_enroll"]].copy()
    d = d.sort_values(["student_number", "year"])

    poor = d["percent_days_attended"] < 90
    good = d["percent_days_attended"] >= 95

    # First year meeting each condition per student
    first_poor = d.loc[poor].groupby("student_number")["year"].min()
    first_good = d.loc[good].groupby("student_number")["year"].min()
    trans = pd.DataFrame({"first_poor_year": first_poor, "first_good_year": first_good}).dropna()
    trans = trans[trans["first_good_year"] > trans["first_poor_year"]]
    trans.index = trans.index.astype(str)

    rows = []
    for sid in trans.index:
        sub = d[d["student_number"] == sid].sort_values("year")
        r0 = sub[sub["year"] == trans.at[sid, "first_poor_year"]].iloc[0]
        r1 = sub[sub["year"] == trans.at[sid, "first_good_year"]].iloc[0]
        school_changed = _q2_school_changed_between_rows(r0, r1)
        grade_delta = float(r1.get("current_grade", np.nan)) - float(
            r0.get("current_grade", np.nan)
        )
        rows.append(
            {
                "student_number": sid,
                "school_changed": school_changed,
                "grade_delta": grade_delta,
                "current_school": r1.get("current_school"),
            }
        )

    if not rows:
        return

    tdf = pd.DataFrame(rows)

    pct_improver_changed = 100.0 * tdf["school_changed"].mean()

    # All students (≥2 years): school at first observed year vs last observed year
    all_changed_flags: list[bool] = []
    for _, sub in d.groupby("student_number", sort=False):
        sub = sub.sort_values("year")
        if len(sub) < 2:
            continue
        all_changed_flags.append(_q2_school_changed_between_rows(sub.iloc[0], sub.iloc[-1]))
    pct_all_changed = 100.0 * float(np.mean(all_changed_flags)) if all_changed_flags else float("nan")

    # Unique students per school (denominator for Q2 panel B ratios)
    students_per_school = (
        d.loc[d["current_school"].notna()]
        .assign(
            current_school=lambda x: x["current_school"].astype(str).str.strip().replace("", np.nan)
        )
        .dropna(subset=["current_school"])
        .groupby("current_school")["student_number"]
        .nunique()
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # A — Grade progression (was panel B)
    ax = axes[0]
    ax.hist(tdf["grade_delta"].dropna(), bins=range(-3, 5), color="#55a868", edgecolor="white")
    ax.set_xlabel("Grade level change")
    ax.set_ylabel("Count")
    ax.set_title("A. Grade progression across transition")

    # B — Improvers who did not change school: improvers per school / students at school
    ax = axes[1]
    same_school = tdf.loc[~tdf["school_changed"]].copy()
    same_school = same_school[same_school["current_school"].notna()]
    if len(same_school):
        vc_school = same_school["current_school"].astype(str).str.strip()
        vc_school = vc_school.replace("", np.nan).dropna()
        counts = vc_school.value_counts()
        if len(counts) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid current_school values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            if len(counts) > 20:
                counts = counts.head(20)
            denom = pd.Series(
                [students_per_school.get(s, np.nan) for s in counts.index],
                index=counts.index,
            )
            ratio_s = counts / denom
            ratio_s = ratio_s[(denom > 0) & ratio_s.notna() & np.isfinite(ratio_s)]
            if ratio_s.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Could not match schools to enrollment counts",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            else:
                y_pos = np.arange(len(ratio_s))
                ax.barh(y_pos, ratio_s.values, color="#8172b3")
                ax.set_yticks(y_pos)
                labels = [str(s)[:40] + ("…" if len(str(s)) > 40 else "") for s in ratio_s.index]
                ax.set_yticklabels(labels, fontsize=7)
                ax.invert_yaxis()
                ax.set_xlabel("Improvers / unique students at school")
    else:
        ax.text(0.5, 0.5, "No improvers with same school", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("B. Improvers: share by school")

    # C — School change: improvers vs all (was panel A)
    ax = axes[2]
    ax.bar(
        ["Improvers\n(poor→good)", "All students\n(≥2 years)"],
        [
            pct_improver_changed / 100.0,
            pct_all_changed / 100.0 if np.isfinite(pct_all_changed) else 0.0,
        ],
        color=["#c44e52", "#4c72b0"],
    )
    ax.set_ylabel("Share with school change")
    ax.set_title("C. School change: improvers vs all students")
    ax.set_ylim(0, 0.2)
    ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Q2 — Students who moved from poor (<90%) to good (≥95%) attendance on different years\n",
        y=1.05,
    )
    fig.tight_layout()
    _save(fig, "eda_q2_poor_to_good_correlates.png")


# ---------------------------------------------------------------------------
# Q3 — Demographics: chronic absence rate and achievement gap
# ---------------------------------------------------------------------------


def chart_q3_demographics(df: pd.DataFrame) -> None:
    d = _prep(df)
    base = d[d["_valid_enroll"]].copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def chronic_rate(sub: pd.DataFrame) -> float:
        return sub["_chronic"].mean() if len(sub) else np.nan

    # A) Chronic rate by ethnicity
    ax = axes[0, 0]
    if "ethnicity" in base.columns:
        eth = base["ethnicity"].value_counts().head(8).index
        rates = [chronic_rate(base[base["ethnicity"] == e]) for e in eth]
        ax.barh([str(e)[:22] for e in eth], rates, color="#4c72b0")
        ax.set_xlabel("Share chronically absent (≥10% days missed)")
        ax.set_title("A. Chronic absence rate by ethnicity (top groups)")
    else:
        ax.set_visible(False)

    # B) Chronic rate by gender
    ax = axes[0, 1]
    if "gender" in base.columns:
        gender_vals = base["gender"].astype(str).str.strip().str.upper()
        grp = [g for g in ["M", "F"] if (gender_vals == g).any()]
        rates = [chronic_rate(base[gender_vals == g]) for g in grp]
        ax.bar([str(g)[:18] for g in grp], rates, color="#c44e52")
        ax.set_xlabel("Share chronically absent")
        ax.set_title("B. Chronic rate by gender")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.set_visible(False)

    # C) GPA gap by ethnicity: non-chronic mean minus chronic mean
    ax = axes[1, 0]
    if "ethnicity" in base.columns:
        eth = base["ethnicity"].value_counts().head(6).index
        labels = []
        gaps = []
        for e in eth:
            sub = base[base["ethnicity"] == e]
            mean_nc = sub.loc[~sub["_chronic"], "overall_gpa"].mean()
            mean_ch = sub.loc[sub["_chronic"], "overall_gpa"].mean()
            gap = mean_nc - mean_ch
            if pd.notna(gap):
                labels.append(str(e)[:12])
                gaps.append(gap)
        ax.bar(labels, gaps, color="#4c72b0")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.8)
        ax.set_ylabel("GPA gap (non-chronic − chronic)")
        ax.set_title("C. Mean GPA gap by ethnicity")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.set_visible(False)

    # D) GPA gap by gender: non-chronic mean minus chronic mean
    ax = axes[1, 1]
    if "gender" in base.columns:
        gender_vals = base["gender"].astype(str).str.strip().str.upper()
        grp = [g for g in ["M", "F"] if (gender_vals == g).any()]
        labels = []
        gaps = []
        for g in grp:
            sub = base[gender_vals == g]
            mean_nc = sub.loc[~sub["_chronic"], "overall_gpa"].mean()
            mean_ch = sub.loc[sub["_chronic"], "overall_gpa"].mean()
            gap = mean_nc - mean_ch
            if pd.notna(gap):
                labels.append(g)
                gaps.append(gap)
        ax.bar(labels, gaps, color="#55a868")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.8)
        ax.set_ylabel("GPA gap (non-chronic − chronic)")
        ax.set_title("D. Mean GPA gap by gender")
    else:
        ax.set_visible(False)

    fig.suptitle(
        "Q3 — Who is most impacted by chronic absence, and is GPA impact similar?\n"
        "Chronic = ≥10% of enrolled days missed.",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "eda_q3_demographics_chronic_gpa.png")


def chart_q3_act_gap(df: pd.DataFrame) -> None:
    d = _prep(df)
    base = d[d["_valid_enroll"] & (d["composite_score"] > 1)].copy()

    if len(base) < 200:
        print("Skipping Q3 ACT gap (insufficient ACT rows).")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if "ethnicity" not in base.columns:
        return
    eth = base["ethnicity"].value_counts().head(6).index
    x = np.arange(len(eth))
    w = 0.35
    for i, e in enumerate(eth):
        sub = base[base["ethnicity"] == e]
        ax.bar(
            i - w / 2,
            sub.loc[~sub["_chronic"], "composite_score"].mean(),
            w,
            color="#8c8c8c",
        )
        ax.bar(
            i + w / 2,
            sub.loc[sub["_chronic"], "composite_score"].mean(),
            w,
            color="#d62728",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(e)[:14] for e in eth], rotation=20, ha="right")
    ax.set_ylabel("Mean ACT composite")
    ax.set_title("Q3 (supplement) — ACT composite: chronic vs not by ethnicity (test-takers)")

    ax.legend(
        handles=[Patch(facecolor="#8c8c8c", label="Not chronic"), Patch(facecolor="#d62728", label="Chronic")],
    )
    fig.tight_layout()
    _save(fig, "eda_q3_act_gap_ethnicity.png")


def chart_q3_disability(df: pd.DataFrame) -> None:
    d = _prep(df)
    base = d[d["_valid_enroll"]].copy()

    if "scram_membership" not in base.columns:
        print("Skipping Q3 disability chart (missing scram_membership).")
        return

    base["_disability_group"] = np.where(
        base["scram_membership"].fillna(0) > 0,
        "Disability (scram_membership > 0)",
        "No disability (scram_membership <= 0)",
    )

    order = [
        "No disability (scram_membership <= 0)",
        "Disability (scram_membership > 0)",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # A) Chronic rate by disability status
    ax = axes[0]
    chronic_rates = (
        base.groupby("_disability_group", observed=True)["_chronic"].mean().reindex(order).dropna()
    )
    ax.bar(chronic_rates.index, chronic_rates.values, color=["#4c72b0", "#c44e52"])
    ax.set_ylabel("Share chronically absent")
    ax.set_title("A. Chronic rate by disability status")
    ax.tick_params(axis="x", rotation=15)

    # B) GPA gap by disability status (non-chronic − chronic)
    ax = axes[1]
    labels: list[str] = []
    gaps: list[float] = []
    for grp in order:
        sub = base[base["_disability_group"] == grp]
        if sub.empty:
            continue
        mean_nc = sub.loc[~sub["_chronic"], "overall_gpa"].mean()
        mean_ch = sub.loc[sub["_chronic"], "overall_gpa"].mean()
        gap = mean_nc - mean_ch
        if pd.notna(gap):
            labels.append(grp)
            gaps.append(gap)
    ax.bar(labels, gaps, color=["#55a868", "#8172b3"][: len(labels)])
    ax.axhline(0, color="gray", linestyle=":", alpha=0.8)
    ax.set_ylabel("GPA gap (non-chronic − chronic)")
    ax.set_title("B. Mean GPA gap by disability status")
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Q3 (supplement) — Disability status, chronic absence, and GPA gap\n"
        "Disability defined as scram_membership > 0.",
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, "eda_q3_disability_chronic_gpa_gap.png")


# ---------------------------------------------------------------------------
# Q4 — Long-run chronic burden vs outcomes (proxies: no graduation/credit in this CSV)
# ---------------------------------------------------------------------------


def chart_q4_longrun(df: pd.DataFrame) -> None:
    d = _prep(df)
    d = d[d["_valid_enroll"]].copy()

    by_stu = d.groupby("student_number", sort=False).agg(
        years=("year", "nunique"),
        chronic_years=("_chronic", "sum"),
        frac_chronic=("_chronic", "mean"),
        ac_ind_max=("ac_ind", "max"),
        ac_count_max=("ac_count", "max"),
        extracurricular_max=("extracurricular_ind", "max"),
        school_count_max=("school_count", "max"),
    )
    last_obs = d.sort_values(["student_number", "year"]).groupby("student_number", sort=False).last()
    st = by_stu.copy()
    st["gpa_last"] = last_obs["overall_gpa"]
    st["composite"] = last_obs["composite_score"]

    # Burden = count of chronic years per student (one row per student in st)
    cy_raw = st["chronic_years"].fillna(0).astype(int).clip(lower=0)
    st["burden"] = np.select(
        [cy_raw == 0, cy_raw == 1, cy_raw == 2, cy_raw == 3, cy_raw >= 4],
        ["No Burden", "1 Year", "2 Years", "3 Years", "4 Years"],
        default="4 Years",
    )
    order = ["No Burden", "1 Year", "2 Years", "3 Years", "4 Years"]
    plot_order = [b for b in order if b in st["burden"].values]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # A) Student counts by chronic years — omit "No Burden" (only 1–4+ years)
    ax = axes[0, 0]
    burden_years = cy_raw
    burden_binned = burden_years.where(burden_years <= 4, 4)
    burden_counts = burden_binned.value_counts().reindex([1, 2, 3, 4], fill_value=0)
    ax.bar(["1 Year", "2 Years", "3 Years", "4 Years"], burden_counts.values, color="#4c72b0")
    ax.set_ylabel("Student count")
    ax.set_title("A. Students by chronic burden years")

    # B) Advanced course share (swapped with former GPA panel)
    ax = axes[0, 1]
    rate = st.groupby("burden")["ac_ind_max"].apply(lambda s: (s > 0).mean()).reindex(plot_order).dropna()
    rate.plot(kind="bar", ax=ax, color="#4c72b0")
    ax.set_ylabel("Share with any advanced course (ac_ind)")
    ax.set_title("B. Ever took advanced/AP/CE-style course")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    # C) GPA by burden (swapped with former advanced panel)
    ax = axes[1, 0]
    sns.boxplot(
        data=st,
        x="burden",
        y="gpa_last",
        order=plot_order,
        ax=ax,
        hue="burden",
        palette="pastel",
        legend=False,
    )
    ax.set_xlabel("Chronic burden (years chronically absent)")
    ax.set_ylabel("Last observed cumulative GPA")
    ax.set_title("C. GPA distribution by chronic burden")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    if (st["composite"] > 1).sum() > 50:
        st_act = st[st["composite"] > 1]
        ord_act = [b for b in order if b in st_act["burden"].values]
        sns.boxplot(
            data=st_act,
            x="burden",
            y="composite",
            order=ord_act,
            ax=ax,
            hue="burden",
            palette="muted",
            legend=False,
        )
        ax.set_xlabel("Chronic burden (years chronically absent)")
        ax.set_title("D. Best ACT composite (ACT takers)")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.set_visible(False)

    fig.suptitle(
        "Q4 — Long-term chronic absenteeism burden vs outcomes\n"
        "Graduation and credits are not in exploratory_data.csv; "
        "AP/CE (ac_ind) and ACT are proxies.",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "eda_q4_chronic_burden_longrun.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"Missing {DATA_PATH}. Run upstream table scripts and 07_combine_data-table.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df):,} rows from {DATA_PATH}")

    chart_q1_hinge(df)
    chart_q1_act(df)
    chart_q2_transitions(df)
    chart_q3_demographics(df)
    chart_q3_act_gap(df)
    chart_q3_disability(df)
    chart_q4_longrun(df)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
