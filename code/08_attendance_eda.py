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
        for gend in base["gender"].dropna().unique()[:6]:
            sub = base[base["gender"] == gend]
            if len(sub) < 300:
                continue
            sub = sub.copy()
            sub["att_bin"] = pd.cut(sub["percent_days_attended"], bins=bins, include_lowest=True)
            gagg = (
                sub.groupby("att_bin", observed=True)
                .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
                .reset_index()
            )
            ax.plot(gagg["mid_att"], gagg["mean_gpa"], marker=".", label=str(gend))
    ax.axvline(90, color="gray", linestyle=":", alpha=0.8)
    ax.set_xlabel("Percent days attended (bin center)")
    ax.set_ylabel("Mean GPA")
    ax.set_title("C. Mean GPA vs attendance by gender")
    ax.legend(fontsize=8)

    # D) Ethnicity — top groups by count
    ax = axes[1, 1]
    if "ethnicity" in base.columns:
        top_eth = base["ethnicity"].value_counts().head(6).index.tolist()
        for eth in top_eth:
            sub = base[base["ethnicity"] == eth]
            if len(sub) < 300:
                continue
            sub = sub.copy()
            sub["att_bin"] = pd.cut(sub["percent_days_attended"], bins=bins, include_lowest=True)
            gagg = (
                sub.groupby("att_bin", observed=True)
                .agg(mean_gpa=("overall_gpa", "mean"), mid_att=("percent_days_attended", "mean"))
                .reset_index()
            )
            ax.plot(gagg["mid_att"], gagg["mean_gpa"], marker=".", label=str(eth)[:20])
    ax.axvline(90, color="gray", linestyle=":", alpha=0.8)
    ax.set_xlabel("Percent days attended (bin center)")
    ax.set_ylabel("Mean GPA")
    ax.set_title("D. Mean GPA vs attendance by ethnicity (top groups)")
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
        school_changed = (
            str(r0.get("current_school", "")) != str(r1.get("current_school", ""))
            if pd.notna(r0.get("current_school")) or pd.notna(r1.get("current_school"))
            else False
        )
        grade_delta = float(r1.get("current_grade", np.nan)) - float(
            r0.get("current_grade", np.nan)
        )
        ac_jump = int(
            (pd.notna(r1.get("ac_ind")) and pd.notna(r0.get("ac_ind")))
            and (r1.get("ac_ind", 0) > r0.get("ac_ind", 0))
        )
        rows.append(
            {
                "student_number": sid,
                "school_changed": school_changed,
                "grade_delta": grade_delta,
                "ac_started": ac_jump,
            }
        )

    if not rows:
        print("No poor→good transitions found; skipping Q2 charts.")
        return

    tdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    vc = tdf["school_changed"].value_counts(normalize=True).reindex([True, False], fill_value=0)
    ax.bar(["Changed school", "Same school"], [vc.get(True, 0), vc.get(False, 0)], color=["#c44e52", "#4c72b0"])
    ax.set_ylabel("Share of improvers")
    ax.set_title("A. School at first poor vs first good year")
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.hist(tdf["grade_delta"].dropna(), bins=range(-3, 5), color="#55a868", edgecolor="white")
    ax.set_xlabel("Grade level change (good year − poor year)")
    ax.set_ylabel("Count")
    ax.set_title("B. Grade progression across transition")

    ax = axes[2]
    ax.bar(
        ["Advanced course flag increased", "No increase"],
        [
            tdf["ac_started"].mean(),
            1 - tdf["ac_started"].mean(),
        ],
        color=["#8172b3", "#ccb974"],
    )
    ax.set_ylabel("Share of improvers")
    ax.set_title("C. AP/CE/concurrent participation (ac_ind)")
    ax.set_ylim(0, 1)

    fig.suptitle(
        "Q2 — Students who moved from poor (<90%) to good (≥95%) attendance on different years\n"
        "Teacher-level correlates are not in exploratory_data.csv (teacher merge disabled in pipeline).",
        fontsize=10,
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

    # B) ell_disability_group
    ax = axes[0, 1]
    if "ell_disability_group" in base.columns:
        grp = base["ell_disability_group"].value_counts().head(8).index
        rates = [chronic_rate(base[base["ell_disability_group"] == g]) for g in grp]
        ax.barh([str(g)[:28] for g in grp], rates, color="#c44e52")
        ax.set_xlabel("Share chronically absent")
        ax.set_title("B. Chronic rate by ELL × disability group")
    else:
        ax.set_visible(False)

    # C) Gender
    ax = axes[1, 0]
    if "gender" in base.columns:
        labels: list[str] = []
        rates_g: list[float] = []
        for g in base["gender"].dropna().unique()[:6]:
            sub = base[base["gender"] == g]
            if len(sub) < 100:
                continue
            labels.append(str(g))
            rates_g.append(chronic_rate(sub))
        if labels:
            ax.bar(labels, rates_g, color="#55a868")
        ax.set_ylabel("Share chronically absent")
        ax.set_title("C. Chronic rate by gender")
    else:
        ax.set_visible(False)

    # D) GPA: chronic vs not by ethnicity (top groups)
    ax = axes[1, 1]
    if "ethnicity" in base.columns:
        eth = base["ethnicity"].value_counts().head(6).index
        x = np.arange(len(eth))
        w = 0.35
        means_ch = []
        means_nc = []
        for e in eth:
            sub = base[base["ethnicity"] == e]
            means_ch.append(sub.loc[sub["_chronic"], "overall_gpa"].mean())
            means_nc.append(sub.loc[~sub["_chronic"], "overall_gpa"].mean())
        ax.bar(x - w / 2, means_nc, w, label="Not chronic", color="#8c8c8c")
        ax.bar(x + w / 2, means_ch, w, label="Chronic", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels([str(e)[:12] for e in eth], rotation=25, ha="right")
        ax.set_ylabel("Mean cumulative GPA")
        ax.set_title("D. Mean GPA: chronic vs not (by ethnicity)")
        ax.legend()

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

    # Burden categories
    def bucket(row: pd.Series) -> str:
        if row["years"] < 2:
            return "Single year only"
        fc = row["frac_chronic"]
        if fc == 0:
            return "Never chronic"
        if fc < 0.34:
            return "Low burden"
        if fc < 0.67:
            return "Moderate burden"
        return "High burden (most years)"

    st["burden"] = st.apply(bucket, axis=1)
    order = [
        "Never chronic",
        "Single year only",
        "Low burden",
        "Moderate burden",
        "High burden (most years)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    plot_order = [b for b in order if b in st["burden"].values]
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
    ax.set_xlabel("Chronic burden across observed years")
    ax.set_ylabel("Last observed cumulative GPA")
    ax.set_title("A. GPA distribution by chronic burden")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    rate = st.groupby("burden")["ac_ind_max"].apply(lambda s: (s > 0).mean()).reindex(plot_order).dropna()
    rate.plot(kind="bar", ax=ax, color="#4c72b0")
    ax.set_ylabel("Share with any advanced course (ac_ind)")
    ax.set_title("B. Ever took advanced/AP/CE-style course")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    ax = axes[1, 0]
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
        ax.set_title("C. Best ACT composite (ACT takers)")
    else:
        ax.set_visible(False)

    ax = axes[1, 1]
    sns.boxplot(
        data=st,
        x="burden",
        y="school_count_max",
        order=plot_order,
        ax=ax,
        hue="burden",
        palette="Set2",
        legend=False,
    )
    ax.set_title("D. School mobility (max school_count across years)")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Q4 — Long-term chronic absenteeism burden vs outcomes\n"
        "Graduation and credits are not in exploratory_data.csv; "
        "AP/CE (ac_ind), ACT, extracurriculars, and mobility are proxies.",
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
    chart_q4_longrun(df)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
