import os
os.environ["PROJ_LIB"] = r"C:\venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_DATA"] = r"C:\venv\Lib\site-packages\rasterio\proj_data"

import streamlit as st    
import matplotlib.pyplot as plt    
import numpy as np    
import folium
import io
from datetime import datetime
from streamlit_folium import st_folium
from idw import load_data, build_idw_grid, zonal_mean_nitrate, run_regression
from idw import make_nitrate_colorbar_html
from jinja2 import Template
from branca.element import MacroElement 
        
st.set_page_config(layout="wide")    
st.title("Nitrate & Cancer Rate Correlation in Wisconsin: A Spatial Relationship Analysis")

with st.expander("Methodology: How this analysis works", expanded=False):
    st.markdown("""
**This analysis assesses the correlation between nitrate levels in drinking water and cancer rates.
In this analysis, nitrate levels were measured at scattered well locations, and cancer
rate data came from census tracts (each census tract has a cancer value called 'canrate'). 
These two datasets' spatial reference systems used degrees instead of feet or meters, and IDW cannot be done using degrees. 
Therefore, their spatial reference systems were reprojected to one that uses meters.

**Analysis Steps**

**1 — Inverse Distance Weighting (IDW):** nitrate levels are interpolated from the well
points into a continuous surface covering the entire study area. Each unsampled location's
estimated nitrate value is a weighted average of nearby wells, where closer wells count
more:
""")
    st.latex(r"\hat{z}(p) = \frac{\sum_i w_i z_i}{\sum_i w_i}, \quad w_i = \frac{1}{d_i^{\,k}}")
    st.markdown("""
The **k** slider controls how fast that influence drops off with distance. Low k lets
distant wells still matter (giving a smoother, more gradual surface); high k means only the
closest wells matter (a sharper, more localized surface). There's no single "correct" k, so 
this app lets you test a range and see whether the nitrate-cancer relationship still appears 
under different k increments.

**2 — Zonal aggregation:** the interpolated nitrate surface within each census tract area
is averaged to produce a nitrate number for each tract.  This way each tract has a cancer rate 
number and a nitrate number.

**3 — Linear regression:** each tract's average nitrate is regressed against the
tract's cancer rate ('canrate') to look for a statistically significant
association.

**Note:** roughly 37% of tracts statewide show an identical cancer rate of exactly 0.01 — 
a data spike inconsistent with the otherwise smooth distribution of values.  
This is unusual, but it is possible. The checkbox below lets you choose whether the 
regression result includes or excludes these 0.01 tracts.
""")
    
@st.cache_data    
def get_data():    
    return load_data()    
        
wells, tracts, county = get_data() 

st.sidebar.markdown("**Move the sliders to adjust the analysis results. Select the checkbox below to exclude an unusual concentration of identical values.**")

k = st.sidebar.slider("IDW distance-decay exponent (k)", 1.0, 6.0, 2.0, 0.5)    
cell_size = st.sidebar.select_slider(    
    "Grid resolution (meters)", options=[500, 1000, 2000, 5000], value=2000    
)    
n_neighbors = st.sidebar.slider("Neighbors used per grid cell", 4, 30, 12)    

exclude_flagged = st.sidebar.checkbox(
    "Exclude tracts with canrate = 0.01 (Unusual concentration of identical values)",
    value=False,
    help="524 of 1,401 tracts statewide show exactly canrate=0.01 — a spike "
         "inconsistent with the smooth decline in neighboring values, likely "
         "a data quality artifact rather than a real measurement."
)

def make_regression_fig(result, footer_lines=None):
    fig, ax = plt.subplots(figsize=(6, 4.6 if footer_lines else 4))
    df = result["df"]
    ax.scatter(df["avg_nitrate"], df["canrate"], alpha=0.4, s=10)
    xline = np.linspace(df["avg_nitrate"].min(), df["avg_nitrate"].max(), 100)
    ax.plot(xline, result["slope"] * xline + result["intercept"], color="red")
    ax.set_xlabel("Avg. Nitrate (mg/L)")
    ax.set_ylabel("Cancer Rate")
    if footer_lines:  # only the saved copy gets the title and settings
        ax.set_title(
            f"Cancer Rate vs. Avg. Nitrate per Tract\n"
            f"R² = {result['r_squared']:.4f}  |  p = {result['p_value']:.3g}  |  "
            f"slope = {result['slope']:.5f}  |  n = {result['n']}", fontsize=9)
        fig.text(0.02, 0.01, "\n".join(footer_lines), fontsize=7,
                 ha="left", va="bottom", family="monospace")
        fig.subplots_adjust(bottom=0.30, top=0.88)
    return fig


def make_map_fig(raster, bounds, tracts_nitrate, wells, footer_lines):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.2))
    im = ax1.imshow(np.flipud(raster),
                    extent=[bounds[0], bounds[2], bounds[1], bounds[3]], cmap="YlOrRd")
    wells.plot(ax=ax1, markersize=1.5, color="black", alpha=0.35)
    fig.colorbar(im, ax=ax1, shrink=0.6, label="Nitrate (mg/L)")
    ax1.set_title("IDW Nitrate Surface")
    tracts_nitrate.plot(column="canrate", ax=ax2, cmap="Blues", legend=True,
                        edgecolor="grey", linewidth=0.1,
                        legend_kwds={"shrink": 0.6, "label": "Cancer rate"})
    ax2.set_title("Census Tract Cancer Rate")
    for ax in (ax1, ax2):
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (meters, EPSG:3070)")
        ax.set_ylabel("Northing (meters, EPSG:3070)")
    fig.text(0.02, 0.01, "\n".join(footer_lines), fontsize=8,
             ha="left", va="bottom", family="monospace")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig

@st.cache_data
def compute_pipeline(_wells, _tracts, k, cell_size, n_neighbors):
    raster, gx, gy, bounds = build_idw_grid(_wells, k, cell_size, n_neighbors)
    tracts_nitrate = zonal_mean_nitrate(raster, gx, gy, _tracts)
    return raster, gx, gy, bounds, tracts_nitrate

with st.spinner("Interpolating nitrate surface..."):
    raster, gx, gy, bounds, tracts_nitrate = compute_pipeline(wells, tracts, k, cell_size, n_neighbors)

    if exclude_flagged:
        regression_input = tracts_nitrate[tracts_nitrate["canrate"] != 0.01]
    else:
        regression_input = tracts_nitrate

    result = run_regression(regression_input)

st.subheader("Interactive Map: Nitrate Surface & Cancer Rate Layers")
st.caption("Use the layer control (top-right of the map) to toggle each layer on or off.")

from idw import raster_to_png_overlay

tracts_wgs84 = tracts_nitrate.to_crs("EPSG:4326")
tracts_wgs84["geometry"] = tracts_wgs84["geometry"].simplify(tolerance=0.001, preserve_topology=True)

minx, miny, maxx, maxy = tracts_wgs84.total_bounds
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles=None)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri",
    name="Light Gray basemap",
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri",
    name="Street basemap",
    show=False,
).add_to(m)

@st.cache_data
def get_overlay(raster, bounds, k):
    return raster_to_png_overlay(raster, bounds)

img_url, img_bounds = get_overlay(raster, bounds, k)
folium.raster_layers.ImageOverlay(
    image=img_url,
    bounds=img_bounds,
    opacity=0.7,
    name=f"IDW Nitrate Surface (k={k})",
).add_to(m)

folium.Choropleth(
    geo_data=tracts_wgs84.__geo_interface__,
    data=tracts_wgs84,
    columns=["GEOID10", "canrate"],
    key_on="feature.properties.GEOID10",
    fill_color="Blues",
    fill_opacity=0.6,
    line_opacity=0.3,
    legend_name="Cancer Rate",
    name="Census Tract Cancer Rate",
).add_to(m)

folium.LayerControl(collapsed=False, position="topright").add_to(m)

# --- Legend for the nitrate surface ---
vmin, vmax = np.nanmin(raster), np.nanmax(raster)
colorbar_b64 = make_nitrate_colorbar_html(vmin, vmax)
legend_html = f"""
<div style="position: fixed; bottom: 30px; left: 50px; z-index: 9999;
            background-color: white; padding: 6px 8px; border: 2px solid grey;
            border-radius: 6px;">
    <img src="data:image/png;base64,{colorbar_b64}" style="width: 180px; display: block;">
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# --- Home button to reset view ---
class HomeButton(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
        var homeControl = L.control({position: 'topleft'});
        homeControl.onAdd = function (map) {
            var div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
            div.innerHTML = '<a href="#" title="Reset view" style="font-size:18px; text-align:center; width:30px; height:30px; line-height:30px; display:block; text-decoration:none;">🏠</a>';
            div.onclick = function(e){
                e.preventDefault();
                {{this._parent.get_name()}}.setView([{{this.lat}}, {{this.lon}}], {{this.zoom}});
            };
            return div;
        };
        homeControl.addTo({{this._parent.get_name()}});
        {% endmacro %}
    """)
    def __init__(self, lat, lon, zoom):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.zoom = zoom

m.add_child(HomeButton(center_lat, center_lon, 7))

st_folium(m, use_container_width=True, height=600, returned_objects=[])

st.subheader("Regression Model: Cancer Rate vs. Avg. Nitrate per Tract")
if exclude_flagged:
    st.caption(f"⚠️ Excluding {(tracts_nitrate['canrate'] == 0.01).sum()} tracts flagged with canrate = 0.01")
else:
    st.caption(f"Including all tracts")  
fig3 = make_regression_fig(result)
st.pyplot(fig3)
plt.close(fig3)    
        
c1, c2, c3 = st.columns(3)    
c1.metric("R²", f"{result['r_squared']:.4f}")    
c2.metric("p-value", f"{result['p_value']:.4g}")    
c3.metric("Slope", f"{result['slope']:.5f}")    
st.caption(f"n = {result['n']} tracts | intercept = {result['intercept']:.4f}")   
# ---------------- Save a copy of results ----------------
st.subheader("Save a copy of your results")
st.caption("Saved files include the settings you chose above.")

signature = (k, cell_size, n_neighbors, exclude_flagged)
excl_text = "excluded" if exclude_flagged else "included"
footer_lines = [
    f"IDW distance-decay exponent (k): {k}",
    f"Grid resolution: {cell_size} m   |   Neighbors per cell: {n_neighbors}",
    f"Tracts with canrate = 0.01: {excl_text}",
    f"Saved: {datetime.now():%Y-%m-%d %H:%M}",
]
file_tag = f"k{k}_grid{cell_size}_nn{n_neighbors}_{'excl' if exclude_flagged else 'incl'}001"

if st.button("Choose output type for current settings"):
    with st.spinner("Preparing files..."):
        reg_fig = make_regression_fig(result, footer_lines)
        reg_buf = io.BytesIO()
        reg_fig.savefig(reg_buf, format="png", dpi=200)
        plt.close(reg_fig)

        map_fig = make_map_fig(raster, bounds, tracts_nitrate, wells, footer_lines)
        map_buf = io.BytesIO()
        map_fig.savefig(map_buf, format="png", dpi=200)
        plt.close(map_fig)

        settings_html = f"""
        <div style="position: fixed; bottom: 30px; right: 10px; z-index: 9999;
                    background: white; padding: 6px 10px; border: 2px solid grey;
                    border-radius: 6px; font: 12px/1.4 Arial, sans-serif;">
          <b>Analysis settings</b><br>
          k = {k}<br>Grid = {cell_size} m<br>Neighbors = {n_neighbors}<br>
          0.01 tracts: {excl_text}<br>
          R² = {result['r_squared']:.4f}, p = {result['p_value']:.3g}
        </div>"""
        map_html = m.get_root().render().replace("</body>", settings_html + "</body>")

        st.session_state["export"] = {
            "signature": signature,
            "tag": file_tag,
            "reg_png": reg_buf.getvalue(),
            "map_png": map_buf.getvalue(),
            "map_html": map_html.encode("utf-8"),
        }

export = st.session_state.get("export")
if export and export["signature"] == signature:
    c1, c2, c3 = st.columns(3)
    c1.download_button("Regression chart (PNG)", export["reg_png"],
                       file_name=f"regression_{export['tag']}.png", mime="image/png")
    c2.download_button("Map (PNG)", export["map_png"],
                       file_name=f"map_{export['tag']}.png", mime="image/png")
    c3.download_button("Interactive map (HTML)", export["map_html"],
                       file_name=f"map_{export['tag']}.html", mime="text/html")
elif export:
    st.info("Your settings changed since the files were prepared. "
            "Click the button again to update them.") 
st.caption("Created by Elijah Gardner Woods, 2026")