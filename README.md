
# Isochrone Streamlit App (OSM Roads + Buildings)

**Generated:** 2025-09-05 11:49:26

This app lets you upload a **GeoJSON** or **zipped Shapefile**, select **distance (km)** or **time (min)** isochrone bands, and computes **network-based isochrones** using OSM roads. It overlays/summarizes **buildings** either from **OSM** (default) or **Microsoft Global ML Building Footprints** via **Planetary Computer** (experimental).

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app_streamlit.py
```

Then upload your area file and click **Compute**.

### Notes
- **Distance mode** uses edge lengths (meters). **Time mode** uses OSMnx-estimated speeds → `travel_time` (seconds).
- **Microsoft Buildings**: requires internet access and the `planetary-computer` + `pystac-client` libs. If fetch fails or isn't available in your area, the app falls back to OSM buildings (if you select OSM).
- Building layer on the map is sampled to keep it light. Download full results from the stats or compute externally.
- For very large buffers or many thresholds, computation may take longer.

### Outputs
- **Interactive map** in the app.
- **Isochrones (GeoJSON)** download.
- **Building stats (CSV)** table and download.

### Troubleshooting
- If you see errors about GDAL/Fiona on Windows, install via Conda:
  ```bash
  conda create -n iso python=3.11
  conda activate iso
  conda install -c conda-forge geopandas shapely fiona pyproj rtree osmnx networkx folium pystac-client planetary-computer pyarrow
  pip install streamlit streamlit-folium
  ```

### Limitations
- Google Open Buildings typically requires BigQuery access; integrating that would need credentials. This app supports OSM (default) and Microsoft PC (experimental) out of the box.
- Time estimates depend on OSM tags and defaults set by OSMnx; they are approximate.
