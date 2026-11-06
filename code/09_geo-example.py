import geopandas as gpd
import pandas as pd
from plotnine import *

poly_path = "private/High Schools Boundary Map/High Schools-POLYGON.shp"
gdf_poly = gpd.read_file(poly_path)

# Clean up non-breaking spaces in names
gdf_poly["Name"] = gdf_poly["Name"].str.replace("\xa0", " ", regex=False)

# Convert polygons to long-form coordinate df for geom_polygon
def gdf_to_coords(gdf, id_col="Name"):
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            for x, y in zip(xs, ys):
                rows.append({id_col: row[id_col], "lon": x, "lat": y})
    return pd.DataFrame(rows)

coords = gdf_to_coords(gdf_poly)

# Centroids for labels (project for accuracy)
gdf_proj = gdf_poly.to_crs("EPSG:3857")
centroids = gdf_proj.geometry.centroid.to_crs("EPSG:4326")
labels_df = pd.DataFrame({
    "Name": gdf_poly["Name"],
    "lon": centroids.x,
    "lat": centroids.y,
})

p = (
    ggplot(coords, aes(x="lon", y="lat", group="Name", fill="Name"))
    + geom_polygon(color="white", size=0.5, alpha=0.85)
    + geom_text(
        data=labels_df,
        mapping=aes(x="lon", y="lat", label="Name", group="Name"),
        size=7,
        color="#1a1a1a",
        fontweight="bold",
        inherit_aes=False,
    )
    + scale_fill_brewer(type="qual", palette="Pastel1", name="School")
    + coord_fixed()
    + labs(
        title="High School Attendance Boundaries",
        x=None,
        y=None,
    )
    + theme_void()
    + theme(
        plot_title=element_text(size=14, face="bold", ha="center"),
        figure_size=(10, 8),
    )
)

p
