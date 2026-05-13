from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "exploratory_data_all_grades.csv"
TEACHER_GRID_BY_YEAR_PATH = ROOT / "data" / "all_teachers_grid_by_year.csv"
HS_TEACHER_GRID_PATH = ROOT / "data" / "hs_teacher_data.csv"
MS_TEACHER_GRID_PATH = ROOT / "data" / "ms_teacher_data.csv"
ES_TEACHER_GRID_PATH = ROOT / "data" / "es_teacher_data.csv"
FIG_DIR = ROOT / "figures"

POOR_PCT_LT = 90.0
GOOD_PCT_GE = 95.0


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    if not DATA_PATH.is_file():
        print(f"Expected data file not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATA_PATH)

    # Data cleaning
    if "exit_code" in df.columns:
        exit_num = pd.to_numeric(df["exit_code"], errors="coerce")
        df = df[exit_num.isna() | (exit_num == 0)].copy()
    df = df[df["current_school"].astype(str).str.strip().str.lower() != "other"].copy()
    # df = df[df["year"] >= 2022].copy()
    df = df[df["school_membership"] >= 60].copy()
    df["student_number"] = df["student_number"].astype(str)
    df["current_school"] = df["current_school"].astype(str).str.strip()
    df = df[df["current_school"].notna() & (df["current_school"] != "") & (df["current_school"] != "nan")].copy()

    grade_col = "current_grade" if "current_grade" in df.columns else "grade"

    # Build improver table (first poor year, then first later good year).
    poor_rows = df[df["percent_days_attended"] < POOR_PCT_LT][["student_number", "year"]]
    first_poor = poor_rows.groupby("student_number", as_index=False)["year"].min().rename(
        columns={"year": "first_poor_year"}
    )
    first_good_any = df[df["percent_days_attended"] >= GOOD_PCT_GE][["student_number", "year"]].rename(
        columns={"year": "good_year"}
    )
    improver_pairs = first_poor.merge(first_good_any, on="student_number", how="inner")
    improver_pairs = improver_pairs[improver_pairs["good_year"] > improver_pairs["first_poor_year"]].copy()
    first_good = (
        improver_pairs.groupby(["student_number", "first_poor_year"], as_index=False)["good_year"]
        .min()
        .rename(columns={"good_year": "first_good_year"})
    )
    if first_good.empty:
        print("No poor->good improvers found after filters; no charts written.", file=sys.stderr)
        return

    first_good_details = first_good.merge(
        df[["student_number", "year", "current_school", grade_col]],
        left_on=["student_number", "first_good_year"],
        right_on=["student_number", "year"],
        how="inner",
    ).drop_duplicates(subset=["student_number"], keep="first")
    first_good_details = first_good_details.rename(columns={grade_col: "current_grade"})

    first_poor_details = first_good.merge(
        df[["student_number", "year", "current_school", grade_col]],
        left_on=["student_number", "first_poor_year"],
        right_on=["student_number", "year"],
        how="inner",
    ).drop_duplicates(subset=["student_number"], keep="first")
    first_poor_details = first_poor_details.rename(columns={grade_col: "current_grade"})

    # Chart 1: Improvers count by transition years (all schools).
    span_counts = (
        first_good.assign(years_to_good=first_good["first_good_year"] - first_good["first_poor_year"])
        .groupby("years_to_good", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("years_to_good")
    )
    if not span_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=span_counts, x="years_to_good", y="n", color="#55a868", ax=ax)
        ax.set_xlabel("Transition Length (Years)")
        ax.set_ylabel("Total Students")
        ax.set_title(
            "Improvers Count by Transition Length (Years)"
        )
        fig.tight_layout()
        save_fig(fig, "improvers_count_by_transition_length.png")

    # Chart 2: Poor attendance percentage by current grade (all schools).
    total_by_grade = df.groupby(grade_col).size().rename("total_n")
    poor_by_grade = df[df["percent_days_attended"] < POOR_PCT_LT].groupby(grade_col).size().rename("poor_n")
    poor_percentage = total_by_grade.to_frame().join(poor_by_grade, how="left").fillna({"poor_n": 0}).reset_index()
    poor_percentage["poor_pct"] = 100.0 * poor_percentage["poor_n"] / poor_percentage["total_n"]
    poor_percentage = poor_percentage.rename(columns={grade_col: "current_grade"}).sort_values("current_grade")
    if not poor_percentage.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=poor_percentage, x="current_grade", y="poor_pct", color="#c44e52", ax=ax)
        ax.set_xlabel("Grade")
        ax.set_ylabel("Poor Attenders / Total Students")
        ax.set_title(
            "Percentage of Poor Attenders by Grade"
        )
        fig.tight_layout()
        save_fig(fig, "percentage_of_poor_attenders_by_grade.png")

    # Chart 3: Teacher-student interactions by school group before one-year poor->good transition.
    student_year = (
        df.groupby(["student_number", "year"], as_index=False)["percent_days_attended"]
        .mean()
        .rename(columns={"percent_days_attended": "pct"})
    )
    one_year_poor = student_year[student_year["pct"] < POOR_PCT_LT][["student_number", "year"]].rename(
        columns={"year": "poor_year"}
    )
    one_year_good = student_year[student_year["pct"] >= GOOD_PCT_GE][["student_number", "year"]].rename(
        columns={"year": "good_year"}
    )
    one_year_transitions = one_year_poor.merge(one_year_good, on="student_number", how="inner")
    one_year_transitions = one_year_transitions[
        one_year_transitions["good_year"] == one_year_transitions["poor_year"] + 1
    ].drop_duplicates(subset=["student_number", "poor_year"])

    anchor = first_good[["student_number", "first_good_year"]].copy()
    anchor["anchor_year"] = anchor["first_good_year"] - 1
    school_grade_lookup = df[["student_number", "year", "current_school", grade_col]].drop_duplicates(
        subset=["student_number", "year"], keep="first"
    )
    trans_anchor = anchor.merge(
        school_grade_lookup,
        left_on=["student_number", "anchor_year"],
        right_on=["student_number", "year"],
        how="inner",
    )
    trans_anchor = trans_anchor.rename(columns={grade_col: "anchor_grade"})

    poor_rows_for_anchor = df[df["percent_days_attended"] < POOR_PCT_LT].copy()
    poor_rows_for_anchor = poor_rows_for_anchor[
        poor_rows_for_anchor["year"].isin(trans_anchor["anchor_year"].unique())
    ].drop_duplicates(subset=["student_number", "year"], keep="first")

    hs_names = set()
    ms_names = set()
    es_names = set()
    if HS_TEACHER_GRID_PATH.is_file():
        hs_names = {str(i).strip() for i in pd.read_csv(HS_TEACHER_GRID_PATH, index_col=0).index}
    if MS_TEACHER_GRID_PATH.is_file():
        ms_names = {str(i).strip() for i in pd.read_csv(MS_TEACHER_GRID_PATH, index_col=0).index}
    if ES_TEACHER_GRID_PATH.is_file():
        es_names = {str(i).strip() for i in pd.read_csv(ES_TEACHER_GRID_PATH, index_col=0).index}

    def group_label(school_name: object) -> str:
        s = str(school_name).strip()
        sl = s.lower()
        if s in es_names or "elem." in sl:
            return "Elementary"
        if s in ms_names or "middle" in sl:
            return "Middle school"
        return "High school"

    q_grouped = trans_anchor.assign(school_group=trans_anchor["current_school"].map(group_label))
    p_grouped = poor_rows_for_anchor.assign(school_group=poor_rows_for_anchor["current_school"].map(group_label))
    n_trans_grp = q_grouped.groupby("school_group")["student_number"].nunique().rename("n_trans")
    n_poor_grp = p_grouped.groupby("school_group")["student_number"].nunique().rename("n_poor")
    group_counts = pd.concat([n_trans_grp, n_poor_grp], axis=1).fillna({"n_trans": 0, "n_poor": 0}).reset_index()
    group_counts = group_counts[group_counts["n_poor"] > 0].copy()
    group_counts["n"] = group_counts["n_trans"] / group_counts["n_poor"]
    if not group_counts.empty:
        group_counts["school_group"] = pd.Categorical(
            group_counts["school_group"],
            categories=["Elementary", "Middle school", "High school"],
            ordered=True,
        )
        group_counts = group_counts.sort_values("school_group")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=group_counts, x="school_group", y="n", color="#55a868", ax=ax)
        ax.set_xlabel("School Group (Year Before Improvement)")
        ax.set_ylabel("Improvers / All Poor Attenders")
        ax.set_title(
            "Percentage of Improvers by School Group"
        )
        fig.tight_layout()
        save_fig(
            fig,
            "percentage_of_improvers_by_school_group.png",
        )

    # Chart 4: Improvers count by current grade (all schools).
    improver_grade_counts = (
        first_good_details.groupby("current_grade", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("current_grade")
    )
    if not improver_grade_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=improver_grade_counts, x="current_grade", y="n", color="#4c72b0", ax=ax)
        ax.set_xlabel("First Good-Attendance Grade")
        ax.set_ylabel("Total Students")
        ax.set_title(
            "Improvers by First Good-Attendance Grade"
        )
        fig.tight_layout()
        save_fig(fig, "improvers_count_by_first_good_attendance_grade.png")

    # Chart 5: Improvers count by school (all schools).
    improver_school_counts = (
        first_poor_details.groupby("current_school", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
    )
    if not improver_school_counts.empty:
        fig, ax = plt.subplots(figsize=(max(10.0, 0.35 * len(improver_school_counts)), 5))
        sns.barplot(data=improver_school_counts, x="current_school", y="n", color="#4c72b0", ax=ax)
        ax.set_xlabel("School (Year Before Improvement)")
        ax.set_ylabel("Total Students")
        ax.set_title(
            "Improvers by School"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        save_fig(fig, "improvers_count_by_school.png")

    # Chart 6: Teacher count before one-year poor->good transition.
    if TEACHER_GRID_BY_YEAR_PATH.is_file() and not one_year_transitions.empty:
        teacher_grid = pd.read_csv(TEACHER_GRID_BY_YEAR_PATH)
        teacher_cols = [c for c in teacher_grid.columns if c.startswith("teacher_")]
        teacher_grid["student_number"] = teacher_grid["student_number"].astype(str)
        exposure = one_year_transitions.merge(
            teacher_grid,
            left_on=["student_number", "poor_year"],
            right_on=["student_number", "year"],
            how="inner",
        )
        if not exposure.empty and teacher_cols:
            n_trans = exposure[teacher_cols].sum(axis=0).astype(int)
            poor_sy = df[df["percent_days_attended"] < POOR_PCT_LT][["student_number", "year"]].drop_duplicates(
                subset=["student_number", "year"], keep="first"
            )
            merged_poor = teacher_grid[["student_number", "year"] + teacher_cols].merge(
                poor_sy, on=["student_number", "year"], how="inner"
            )
            n_poor = (merged_poor[teacher_cols] > 0).sum(axis=0).astype(int)
            enrolled_sy = df[["student_number", "year"]].drop_duplicates(subset=["student_number", "year"])
            merged_enrolled = teacher_grid[["student_number", "year"] + teacher_cols].merge(
                enrolled_sy, on=["student_number", "year"], how="inner"
            )
            n_enrolled = (merged_enrolled[teacher_cols] > 0).sum(axis=0).astype(int)

            teacher_counts = pd.DataFrame(
                {
                    "teacher": n_trans.index.astype(str),
                    "n_trans": n_trans.values,
                    "n_poor": n_poor.reindex(n_trans.index).fillna(0).astype(int).values,
                    "n_enrolled": n_enrolled.reindex(n_trans.index).fillna(0).astype(int).values,
                }
            )
            teacher_counts = teacher_counts[teacher_counts["n_trans"] > 0].copy()
            teacher_counts["ratio"] = np.where(
                teacher_counts["n_poor"] > 0,
                teacher_counts["n_trans"] / teacher_counts["n_poor"],
                np.nan,
            )
            if not teacher_counts.empty:
                if len(teacher_counts) == 1:
                    pool = teacher_counts.sort_values(["ratio", "n_trans"], ascending=[False, False])
                    load_note = "Only one teacher with transition links; mean-load filter skipped."
                else:
                    avg_load = float(teacher_counts["n_enrolled"].mean())
                    pool = teacher_counts[teacher_counts["n_enrolled"] > avg_load].copy()
                    pool = pool.sort_values(["ratio", "n_trans"], ascending=[False, False])
                    load_note = f"Teachers with enrolled-linked student-years > mean ({avg_load:.1f}); top 30."

                if not pool.empty:
                    affiliation: dict[str, set[str]] = {}
                    for path in (HS_TEACHER_GRID_PATH, MS_TEACHER_GRID_PATH, ES_TEACHER_GRID_PATH):
                        if not path.is_file():
                            continue
                        aff_grid = pd.read_csv(path, index_col=0)
                        for sch_name, row in aff_grid.iterrows():
                            school = str(sch_name).strip()
                            for col in teacher_cols:
                                if col in aff_grid.columns and pd.notna(row[col]) and float(row[col]) >= 1:
                                    affiliation.setdefault(col, set()).add(school)

                    plot_df = pool.head(30).iloc[::-1].copy()
                    plot_df["ratio_plot"] = plot_df["ratio"].fillna(0.0)
                    fig, ax = plt.subplots(figsize=(14, 8))
                    sns.barplot(data=plot_df, x="ratio_plot", y="teacher", color="#8172b3", ax=ax)
                    ax.set_xlabel("Improvers / All Poor Attenders")
                    ax.set_ylabel("Teacher")
                    ax.set_title(
                        "Top 30 Teachers by Percentage of Improvers"
                    )
                    max_x = float(plot_df["ratio_plot"].max()) if not plot_df.empty else 0.0
                    x_pad = max(0.02, 0.04 * max_x) if max_x > 0 else 0.05
                    ax.set_xlim(0, min(1.0, max_x + x_pad) if max_x <= 1.0 else max_x + x_pad)
                    for patch, row in zip(ax.patches, plot_df.itertuples(index=False)):
                        schools = sorted(affiliation.get(row.teacher, set()))
                        school_text = ", ".join(schools) if schools else "Unmapped"
                        rtxt = f"{row.n_trans}/{row.n_poor}" if row.n_poor > 0 else f"{row.n_trans}/0"
                        label = f"{school_text}  ({rtxt}, N={row.n_enrolled})"
                        y_mid = patch.get_y() + (patch.get_height() / 2.0)
                        ax.text(patch.get_width() + x_pad, y_mid, label, va="center", ha="left", fontsize=8)
                    fig.tight_layout()
                    save_fig(fig, "top_30_teachers_by_percentage_of_improvers.png")

    # Chart 7: Teacher-student interactions by grade before one-year poor->good transition.
    if not trans_anchor.empty and not poor_rows_for_anchor.empty:
        n_trans_grade = trans_anchor.groupby("anchor_grade")["student_number"].nunique().rename("n_trans")
        n_poor_grade = poor_rows_for_anchor.groupby(grade_col)["student_number"].nunique().rename("n_poor")
        grade_counts = pd.concat([n_trans_grade, n_poor_grade], axis=1).fillna({"n_trans": 0, "n_poor": 0})
        grade_counts = grade_counts.reset_index()
        grade_counts = grade_counts.rename(columns={grade_counts.columns[0]: "grade"})
        grade_counts = grade_counts[grade_counts["n_poor"] > 0].copy()
        grade_counts["n"] = grade_counts["n_trans"] / grade_counts["n_poor"]
        grade_counts = grade_counts.sort_values("grade")
        if not grade_counts.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(data=grade_counts, x="grade", y="n", color="#8172b3", ax=ax)
            ax.set_xlabel("Grade (Year Before Improvement)")
            ax.set_ylabel("Improvers / All Poor Attenders")
            ax.set_title("Percentage of Improvers by Grade")
            fig.tight_layout()
            save_fig(
                fig,
                "percentage_of_improvers_by_grade.png",
            )

    # Chart 8: Teacher-student interactions by school before one-year poor->good transition.
    if not trans_anchor.empty and not poor_rows_for_anchor.empty:
        n_trans_school = trans_anchor.groupby("current_school")["student_number"].nunique().rename("n_trans")
        n_poor_school = poor_rows_for_anchor.groupby("current_school")["student_number"].nunique().rename("n_poor")
        school_counts = pd.concat([n_trans_school, n_poor_school], axis=1).fillna({"n_trans": 0, "n_poor": 0})
        school_counts = school_counts.reset_index()
        school_counts = school_counts[school_counts["n_poor"] > 0].copy()
        school_counts["n"] = school_counts["n_trans"] / school_counts["n_poor"]
        school_counts = school_counts.sort_values("n", ascending=False)
        if not school_counts.empty:
            fig, ax = plt.subplots(figsize=(max(10.0, 0.38 * len(school_counts)), 5))
            sns.barplot(data=school_counts, x="current_school", y="n", color="#4c72b0", ax=ax)
            ax.set_xlabel("School (Year Before Improvement)")
            ax.set_ylabel("Improvers / All Poor Attenders")
            ax.set_title("Percentage of Improvers by School")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            save_fig(
                fig,
                "percentage_of_improvers_by_school.png",
            )

if __name__ == "__main__":
    main()