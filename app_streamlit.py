
import os
import io
import sys
import zipfile
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import networkx as nx
import folium
from streamlit_folium import st_folium

import osmnx as ox
ox.settings.log_console = False
ox.settings.use_cache = True

# Optional Planetary Computer (Microsoft Building Footprints)
try:
    from pystac_client import Client
    import planetary_computer as pc
    HAS_PC = True
except Exception:
    HAS_PC = False

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Isochrone Map • Roads + Buildings", layout="wide")

st.title("Distance/Time Isochrones with OSM Roads + Buildings")
st.caption("Upload a shapefile (.zip) or GeoJSON, choose time/distance bands, and compute isochrones along OSM roads. "
           "Buildings are fetched from OSM (default) or Microsoft Global ML Building Footprints via Planetary Computer (experimental).")

# -------------------- Helpers --------------------
def to_utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    north = lat >= 0
    return 32600 + zone if north else 32700 + zone

def project_gdf(gdf: gpd.GeoDataFrame, epsg: int=None) -> gpd.GeoDataFrame:
    if epsg is None:
        c = gdf.to_crs(4326).geometry.unary_union.centroid
        epsg = to_utm_epsg(c.y, c.x)
    return gdf.to_crs(epsg=epsg)

def build_study_area(gdf: gpd.GeoDataFrame, buffer_m: float) -> Polygon:
    gdf_wgs = gdf.to_crs(4326)
    uni = gdf_wgs.geometry.unary_union
    c = uni.centroid
    utm_epsg = to_utm_epsg(c.y, c.x)
    gdf_utm = gdf_wgs.to_crs(epsg=utm_epsg)
    buffered = gdf_utm.buffer(buffer_m)
    merged = unary_union(buffered)
    study_area = gpd.GeoSeries([merged], crs=f"EPSG:{utm_epsg}").to_crs(4326).iloc[0]
    return study_area

def ensure_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.geometry.iloc[0].geom_type in ("Polygon","MultiPolygon","LineString","MultiLineString"):
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.centroid
    return gdf

def read_uploaded_geometry(upload) -> gpd.GeoDataFrame:
    """Read uploaded GeoJSON or Shapefile .zip and return GeoDataFrame (WGS84)."""
    suffix = Path(upload.name).suffix.lower()
    if suffix in (".geojson", ".json", ".gpkg", ".kml", ".gml"):
        gdf = gpd.read_file(upload)
    elif suffix == ".zip":
        # Treat as zipped shapefile
        tmpdir = tempfile.mkdtemp()
        zpath = Path(tmpdir) / "uploaded.zip"
        with open(zpath, "wb") as f:
            f.write(upload.read())
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmpdir)
        # Find a .shp
        shp = None
        for p in Path(tmpdir).rglob("*.shp"):
            shp = p
            break
        if shp is None:
            raise ValueError("No .shp found inside the ZIP.")
        gdf = gpd.read_file(shp)
    else:
        raise ValueError("Unsupported file type. Please upload a .geojson/.json/.gpkg/.kml or a zipped shapefile (.zip).")

    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(4326)
    return gdf

def nearest_nodes_for_points(G, gdf_points: gpd.GeoDataFrame) -> List[int]:
    xs = gdf_points.geometry.x.to_list()
    ys = gdf_points.geometry.y.to_list()
    return ox.distance.nearest_nodes(G, X=xs, Y=ys)

def make_iso_polys(G, nodes_gdf, thresholds) -> gpd.GeoDataFrame:
    polys = []
    for thr in thresholds:
        subnodes = nodes_gdf[nodes_gdf['dist'] <= thr]
        if len(subnodes) == 0:
            continue
        subgraph = G.subgraph(subnodes.index)
        _, subedges = ox.graph_to_gdfs(subgraph)
        if len(subedges) == 0:
            continue
        poly = subedges.buffer(50).unary_union  # 50m buffer around edges
        polys.append({'threshold': thr, 'geometry': poly})
    return gpd.GeoDataFrame(polys, crs=nodes_gdf.crs)

@st.cache_data(show_spinner=False)
def fetch_osm_buildings(polygon) -> gpd.GeoDataFrame:
    tags = {"building": True}
    try:
        g = ox.geometries_from_polygon(polygon, tags)
        if g.empty:
            return g
        return g[g.geometry.geom_type.isin(["Polygon","MultiPolygon"])].copy()
    except Exception as e:
        st.warning(f"OSM building fetch failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

@st.cache_data(show_spinner=False)
def fetch_ms_buildings_pc(polygon) -> gpd.GeoDataFrame:
    if not HAS_PC:
        raise RuntimeError("Planetary Computer libraries not installed. See README for enabling Microsoft buildings.")
    try:
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(collections=["ms-buildings"], intersects=polygon.__geo_interface__)
        items = list(search.get_items())
        if len(items) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        frames = []
        for it in items:
            signed = pc.sign(it)
            # Prefer Parquet if present; otherwise try GeoJSON
            asset = signed.assets.get("data") or list(signed.assets.values())[0]
            href = asset.href
            if href.endswith(".parquet"):
                df = gpd.read_parquet(href)
            else:
                df = gpd.read_file(href)
            if df.crs is None:
                df.set_crs(epsg=4326, inplace=True)
            df = df.to_crs(4326)
            # Keep only polygons
            df = df[df.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
            if not df.empty:
                frames.append(df[["geometry"]].copy())
        if len(frames) == 0:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        out = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(out, crs="EPSG:4326")
    except Exception as e:
        st.warning(f"Microsoft PC buildings fetch failed, will fall back to OSM if selected: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def summarize_buildings_by_band(buildings_gdf: gpd.GeoDataFrame, iso_gdf: gpd.GeoDataFrame, centroid) -> pd.DataFrame:
    if buildings_gdf is None or buildings_gdf.empty or iso_gdf is None or iso_gdf.empty:
        return pd.DataFrame(columns=["band","buildings","area_m2_sum","area_m2_mean"])
    # compute area in UTM for accuracy
    utm_epsg = to_utm_epsg(centroid.y, centroid.x)
    bld = buildings_gdf.to_crs(epsg=utm_epsg).copy()
    bld["area_m2"] = bld.geometry.area
    bld_pts = bld.copy()
    bld_pts["geometry"] = bld_pts.geometry.representative_point()

    iso_bands = iso_gdf.copy()
    iso_bands["band"] = iso_bands["label"]
    joined = gpd.sjoin(bld_pts, iso_bands[["band","geometry"]], how="left", predicate="within")
    stats = joined.groupby("band").agg(
        buildings=("area_m2","count"),
        area_m2_sum=("area_m2","sum"),
        area_m2_mean=("area_m2","mean")
    ).reset_index().sort_values("band")
    return stats

def gdf_to_download_bytes(gdf: gpd.GeoDataFrame, driver="GeoJSON") -> bytes:
    buf = io.BytesIO()
    if driver.lower() == "geojson":
        gdf.to_file(buf, driver="GeoJSON")
        return buf.getvalue()
    # default to GeoPackage inside memory file
    tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
    tmp.close()
    gdf.to_file(tmp.name, driver="GPKG")
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data

# -------------------- Sidebar UI --------------------
with st.sidebar:
    st.header("1) Upload Area")
    up = st.file_uploader("Upload a GeoJSON or a zipped Shapefile (.zip)", type=["geojson","json","zip","gpkg","kml","gml"])

    st.header("2) Parameters")
    net_type = st.selectbox("Network type", ["drive","walk"], index=0)
    mode = st.selectbox("Isochrone mode", ["distance (km)","time (min)"], index=0)
    thr_str = st.text_input("Thresholds (comma separated)", value="2,5,8" if mode.startswith("distance") else "10,20,30")
    buf_km = st.slider("Study area buffer (km)", 2, 30, 12)
    bsrc = st.selectbox("Buildings source", ["OSM (default)","Microsoft (Planetary Computer)"], index=0,
                        help="Microsoft may require internet access and the Planetary Computer libraries.")

    run = st.button("Compute")

# -------------------- Main flow --------------------
if not up:
    st.info("Upload your area file to begin (point(s) or polygon(s) are fine).")
    st.stop()

# Parse thresholds
try:
    vals = [v.strip() for v in thr_str.split(",") if v.strip()]
    if mode.startswith("distance"):
        thresholds = [float(v)*1000 for v in vals]  # km -> meters
        label_from_thr = lambda t: f"{t/1000:.1f} km"
        weight_key = "length"
    else:
        thresholds = [float(v)*60 for v in vals]    # minutes -> seconds
        label_from_thr = lambda t: f"{int(round(t/60))} min"
        weight_key = "travel_time"
    thresholds = sorted(thresholds)
except Exception:
    st.error("Could not parse thresholds. Use comma-separated numbers (e.g., 2,5,8).")
    st.stop()

# Read geometry
try:
    gdf = read_uploaded_geometry(up)
    gdf_points = ensure_points(gdf)
    centroid = gdf_points.geometry.unary_union.centroid
except Exception as e:
    st.error(f"Failed to read geometry: {e}")
    st.stop()

if run:
    with st.spinner("Building study area..."):
        study_poly = build_study_area(gdf_points, buf_km*1000)
        study_gdf = gpd.GeoDataFrame({"id":[1]}, geometry=[study_poly], crs="EPSG:4326")

    with st.spinner("Downloading OSM road network... (can take a minute)"):
        try:
            G = ox.graph_from_polygon(study_poly, network_type=net_type, simplify=True)
            # Project graph to a local meter-based CRS so nearest-node search doesn't need scikit-learn
            G = ox.project_graph(G)
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            nodes, edges = ox.graph_to_gdfs(G)
        except Exception as e:
            st.error(f"OSM road network fetch failed: {e}")
            st.stop()

    with st.spinner("Computing isochrones..."):
        try:
            # Reproject uploaded points to the graph's CRS before nearest-node search
            gdf_points_proj = gdf_points.to_crs(nodes.crs)
            xs = gdf_points_proj.geometry.x.to_list()
            ys = gdf_points_proj.geometry.y.to_list()
            origins = ox.distance.nearest_nodes(G, X=xs, Y=ys)

            # multi-source Dijkstra
            dist_dicts = []
            for o in origins:
                dist = nx.single_source_dijkstra_path_length(G, o, weight=weight_key)
                dist_dicts.append(dist)
            all_nodes = set().union(*[d.keys() for d in dist_dicts])
            min_dist = {n: min(d.get(n, np.inf) for d in dist_dicts) for n in all_nodes}
            nodes_iso = nodes.copy()
            nodes_iso["dist"] = nodes_iso.index.map(min_dist)
            nodes_iso = nodes_iso.dropna(subset=["dist"])
            iso_gdf = make_iso_polys(G, nodes_iso, thresholds)
            # Convert isochrones to WGS84 for display/download
            iso_gdf = iso_gdf.to_crs(4326)
            iso_gdf["label"] = [label_from_thr(t) for t in iso_gdf["threshold"]]

            iso_gdf = iso_gdf.sort_values("threshold").reset_index(drop=True)
        except Exception as e:
            st.error(f"Isochrone computation failed: {e}")
            st.stop()

    with st.spinner("Fetching buildings and summarizing..."):
        buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        if bsrc.startswith("Microsoft") and HAS_PC:
            ms = fetch_ms_buildings_pc(study_poly)
            if ms is not None and not ms.empty:
                buildings = ms
            else:
                st.warning("Microsoft buildings not available here; falling back to OSM buildings.")
                buildings = fetch_osm_buildings(study_poly)
        else:
            buildings = fetch_osm_buildings(study_poly)

        stats_df = summarize_buildings_by_band(buildings, iso_gdf, centroid)

    # -------------------- Map --------------------
    st.subheader("Interactive Map")
    center = (centroid.y, centroid.x)
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")

    palette = ["#d4f0f0", "#86c5da", "#2f8fcb", "#1b5e9a", "#0a3a66", "#06243d"]
    for i, row in iso_gdf.iterrows():
        color = palette[min(i, len(palette)-1)]
        gj = folium.GeoJson(
            row.geometry.__geo_interface__,
            name=f"Isochrone {row['label']}",
            style_function=lambda x, c=color: {"fillColor": c, "color": c, "weight": 1, "fillOpacity": 0.35}
        )
        gj.add_to(m)

    # origin markers
    for _, s in gdf_points.iterrows():
        folium.CircleMarker(location=(s.geometry.y, s.geometry.x), radius=7, color="black", fill=True, fill_opacity=1).add_to(m)

    # Draw a light road overlay for context
    try:
        edges_simple = edges[["geometry"]].to_crs(4326).copy()
        folium.GeoJson(edges_simple.to_json(), name="Road network", style_function=lambda x: {"color":"#555","weight":1}).add_to(m)
    except Exception as e:
        st.warning(f"Road overlay skipped: {e}")

    # Building sample layer (to keep map light)
    try:
        if buildings is not None and not buildings.empty:
            bpts = buildings.copy()
            bpts["geometry"] = bpts.geometry.representative_point()
            sample = bpts.sample(min(5000, len(bpts)))
            folium.GeoJson(sample.to_json(), name="Building centroids (sample)").add_to(m)
    except Exception as e:
        st.warning(f"Building sample layer skipped: {e}")

    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=640)

    # -------------------- Outputs --------------------
    st.subheader("Downloads")

    # Isochrones GeoJSON
    try:
        iso_bytes = gdf_to_download_bytes(iso_gdf, driver="GeoJSON")
        st.download_button("Download Isochrones (GeoJSON)", data=iso_bytes, file_name="isochrones.geojson", mime="application/geo+json")
    except Exception as e:
        st.warning(f"Could not prepare isochrones download: {e}")

    # Building stats CSV
    if stats_df is not None and not stats_df.empty:
        st.dataframe(stats_df, use_container_width=True)
        st.download_button("Download Building Stats (CSV)", data=stats_df.to_csv(index=False).encode("utf-8"),
                           file_name="building_stats_by_band.csv", mime="text/csv")
    else:
        st.info("No building stats available for the selected area.")

    with st.expander("Debug info"):
        st.write(f"Origins: {len(gdf_points)}")
        st.write(f"Isochrone bands: {list(iso_gdf['label'])}")
        st.write(f"Buildings source: {'Microsoft PC' if bsrc.startswith('Microsoft') else 'OSM'}")
        st.write(f"Planetary Computer available: {HAS_PC}")
