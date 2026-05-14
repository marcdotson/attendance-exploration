from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from plotnine import *

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "exploratory_data.csv"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HS_POLY = ROOT / "data" / "High Schools Boundary Maps" / "High Schools-POLYGON.shp"
MS_DIR  = ROOT / "data" / "Middle Schools Boundary Map"
ES_POLY = ROOT / "data" / "Elementary Boundary Maps" / "Elementary Boundary.shp"

MIN_MEMBERSHIP     = 60
CHRONIC_ABSENT_PCT = 10.0

# ── Load & coerce ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
for _col in ["percent_days_attended", "school_membership", "days_absent", "year"]:
    if _col in df.columns:
        df[_col] = pd.to_numeric(df[_col], errors="coerce")
df["student_number"] = df["student_number"].astype(str)
df["current_school"] = df["current_school"].astype(str).str.strip()

# ── Chronic base (all years) ───────────────────────────────────────────────────
_cbase = df[df["school_membership"] >= MIN_MEMBERSHIP].copy()
_cbase["_pct_absent"] = 100.0 * _cbase["days_absent"].fillna(0) / _cbase["school_membership"]
_cbase["_chronic"]    = _cbase["_pct_absent"] >= CHRONIC_ABSENT_PCT

# ── Improver base (2022+, matching 08_attendance.py logic) ────────────────────
_base_imp = df[
    (df["school_membership"] >= MIN_MEMBERSHIP) &
    df["year"].notna() &
    df["percent_days_attended"].notna()
].copy()
if "exit_code" in _base_imp.columns:
    _exit_num = pd.to_numeric(_base_imp["exit_code"], errors="coerce")
    _base_imp = _base_imp[_exit_num.isna() | (_exit_num == 0)].copy()
_base_imp = _base_imp[_base_imp["current_school"].str.lower() != "other"].copy()
_base_imp = _base_imp[_base_imp["year"] >= 2022].copy()
_base_imp["current_school"] = _base_imp["current_school"].astype(str).str.strip()
_base_imp = _base_imp[
    (_base_imp["current_school"] != "") & (_base_imp["current_school"] != "nan")
].copy()

_poor_rows = _base_imp[_base_imp["percent_days_attended"] < 90.0].copy()
_first_poor_yr = _poor_rows.groupby("student_number")["year"].min().rename("first_poor_year")
_good_rows = (
    _base_imp[_base_imp["percent_days_attended"] >= 95.0][["student_number", "year"]]
    .join(_first_poor_yr, on="student_number")
)
_good_rows = _good_rows[_good_rows["year"] > _good_rows["first_poor_year"]]
_first_good_yr = _good_rows.groupby("student_number")["year"].min().rename("first_good_year")
_improvers = pd.concat([_first_poor_yr, _first_good_yr], axis=1).dropna()

_sg_lookup = (
    _base_imp[["student_number", "year", "current_school"]]
    .drop_duplicates(subset=["student_number", "year"], keep="first")
)
_anchor = _improvers.reset_index()[["student_number", "first_good_year"]].copy()
_anchor["anchor_year"] = _anchor["first_good_year"]
_trans_anchor = (
    _anchor
    .merge(_sg_lookup,
           left_on=["student_number", "anchor_year"],
           right_on=["student_number", "year"],
           how="inner")
    .drop_duplicates("student_number", keep="first")
)
_anc_school           = _trans_anchor.dropna(subset=["current_school"])
_poor_rows_anchor     = _poor_rows.drop_duplicates(subset=["student_number", "year"], keep="first")
_poor_rows_anc_school = _poor_rows_anchor.dropna(subset=["current_school"])


def _school_metrics(schools: list[str]) -> pd.DataFrame:
    rows = []
    for s in schools:
        sub = _cbase[_cbase["current_school"] == s]
        if len(sub) < 10:
            rows.append({"school": s, "mean_att": np.nan, "chronic_rate": np.nan, "improver_rate": np.nan})
            continue
        n_poor  = _poor_rows_anc_school[_poor_rows_anc_school["current_school"] == s]["student_number"].nunique()
        n_trans = _anc_school[_anc_school["current_school"] == s]["student_number"].nunique()
        rows.append({
            "school":        s,
            "mean_att":      sub["percent_days_attended"].mean(),
            "chronic_rate":  sub["_chronic"].mean() * 100,
            "improver_rate": (n_trans / n_poor * 100) if n_poor > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def _clean(s: str) -> str:
    return s.replace("\xa0", " ").strip()


# Convert polygons to long-form coordinate df for geom_polygon (from 09_geo-example.py)
def gdf_to_coords(gdf, id_col="att_school"):
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            for x, y in zip(xs, ys):
                rows.append({id_col: row[id_col], "lon": x, "lat": y})
    return pd.DataFrame(rows)


def make_geo_plots(
    gdf: gpd.GeoDataFrame,
    metrics: pd.DataFrame,
    label_map: dict[str, str],
    level_name: str,
    file_prefix: str,
) -> None:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Dissolve non-contiguous zones into one geometry per school
    gdf = gdf[["att_school", "geometry"]].dissolve(by="att_school", as_index=False)
    gdf = gdf.merge(metrics.rename(columns={"school": "att_school"}), on="att_school", how="left")

    # Centroids for labels (project for accuracy, like 09_geo-example.py)
    gdf_proj = gdf.to_crs("EPSG:3857")
    centroids = gdf_proj.geometry.centroid.to_crs("EPSG:4326")
    labels_df = pd.DataFrame({
        "att_school": gdf["att_school"],
        "label": gdf["att_school"].map(label_map).fillna(gdf["att_school"]),
        "lon": centroids.x,
        "lat": centroids.y,
    })

    # Convert polygons to long-form coordinate df, then join metric values
    coords = gdf_to_coords(gdf)
    coords = coords.merge(
        gdf[["att_school", "mean_att", "chronic_rate", "improver_rate"]],
        on="att_school", how="left",
    )

    panels = [
        ("mean_att",      "Mean Attendance (%)",          "Blues",  f"{file_prefix}_mean_att.png"),
        ("chronic_rate",  "Chronic Absenteeism (%)",      "Reds",   f"{file_prefix}_chronic.png"),
        ("improver_rate", "Improver Rate (%, 2022–2025)", "Greens", f"{file_prefix}_improvers.png"),
    ]

    for fill_col, fill_label, palette, filename in panels:
        p = (
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
            + labs(
                title=f"{level_name} — {fill_label}",
                x=None,
                y=None,
            )
            + theme_void()
            + theme(
                plot_title=element_text(size=14, face="bold", ha="center"),
                figure_size=(10, 8),
            )
        )
        p.save(str(FIG_DIR / filename), dpi=150, verbose=False)
        print(f"Wrote {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# HIGH SCHOOLS
# ══════════════════════════════════════════════════════════════════════════════

HS_MAP = {
    "Sky View High School":    ("Sky View",      "Sky View"),
    "Green Canyon High School": ("Green Canyon", "Green Canyon"),
    "Ridgeline High School":   ("Ridgeline",     "Ridgeline"),
    "Mt. Crest High School":   ("Mt. Crest",     "Mountain Crest"),
}

gdf_hs = gpd.read_file(HS_POLY)
gdf_hs["Name"]       = gdf_hs["Name"].str.replace("\xa0", " ", regex=False)
gdf_hs["att_school"] = gdf_hs["Name"].map({k: v[1] for k, v in HS_MAP.items()})

make_geo_plots(
    gdf_hs,
    metrics   = _school_metrics([v[1] for v in HS_MAP.values()]),
    label_map = {v[1]: v[0] for v in HS_MAP.values()},
    level_name = "High Schools",
    file_prefix = "geo_hs",
)


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLE SCHOOLS
# ══════════════════════════════════════════════════════════════════════════════

MS_MAP = {
    "North Cache Middle School":  ("North Cache",  "North Cache Middle"),
    "South Cache Middle School":  ("South Cache",  "South Cache Middle"),
    "Spring Creek Middle School": ("Spring Creek", "Spring Creek Middle"),
}

_ms_parts = []
for _shp in sorted(MS_DIR.glob("*-POLYGON.shp")):
    _g = gpd.read_file(_shp)
    _g["Name"]       = _g["Name"].str.replace("\xa0", " ", regex=False)
    _g["att_school"] = _g["Name"].map({k: v[1] for k, v in MS_MAP.items()})
    _ms_parts.append(_g)
gdf_ms = gpd.GeoDataFrame(pd.concat(_ms_parts, ignore_index=True), geometry="geometry")

make_geo_plots(
    gdf_ms,
    metrics    = _school_metrics([v[1] for v in MS_MAP.values()]),
    label_map  = {v[1]: v[0] for v in MS_MAP.values()},
    level_name  = "Middle Schools",
    file_prefix = "geo_ms",
)


# ══════════════════════════════════════════════════════════════════════════════
# ELEMENTARY SCHOOLS
# ══════════════════════════════════════════════════════════════════════════════

# Wellesville is a typo in the attendance CSV; the shapefile has Wellsville.
ES_MAP = {
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

gdf_es = gpd.read_file(ES_POLY)
gdf_es["Name"]       = gdf_es["Name"].str.replace("\xa0", " ", regex=False)
gdf_es["att_school"] = gdf_es["Name"].map({k: v[1] for k, v in ES_MAP.items()})

make_geo_plots(
    gdf_es,
    metrics    = _school_metrics([v[1] for v in ES_MAP.values()]),
    label_map  = {v[1]: v[0] for v in ES_MAP.values()},
    level_name  = "Elementary Schools",
    file_prefix = "geo_es",
)

print(f"\nDone. Figures written to: {FIG_DIR}")
