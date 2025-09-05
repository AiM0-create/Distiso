
# Isochrone Streamlit App (OSM Roads + Buildings)

**Generated:** 2025-09-05 13:00:44

Upload a **GeoJSON** or **zipped Shapefile**, choose **distance (km)** or **time (min)** bands, and compute **network-based isochrones** using OSM roads. Overlay/summarize **buildings** from **OSM (default)** or **Microsoft Global ML Building Footprints** via **Planetary Computer**.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app_streamlit.py
```

### Notes
- Graph is **projected** before nearest-node search → avoids BallTree/scikit-learn requirement.
- Cached functions accept Shapely geometries thanks to `hash_funcs={BaseGeometry: lambda g: g.wkb}`.
- If Microsoft buildings aren't available, app falls back to OSM buildings.

### Conda (Windows-friendly)
```bash
conda create -n iso python=3.11
conda activate iso
conda install -c conda-forge geopandas shapely fiona pyproj rtree osmnx networkx folium pystac-client planetary-computer pyarrow
pip install streamlit streamlit-folium scikit-learn
```
