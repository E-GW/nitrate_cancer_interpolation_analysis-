import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from pyproj import Transformer
from scipy.spatial import cKDTree
from rasterstats import zonal_stats
from affine import Affine
from scipy import stats as scistats



WI_CRS = "EPSG:3070"


def load_data(data_dir="data"):
    wells = gpd.read_file(f"{data_dir}/well_nitrate.shp").to_crs(WI_CRS)
    tracts = gpd.read_file(f"{data_dir}/cancer_tracts.shp").to_crs(WI_CRS)
    county = gpd.read_file(f"{data_dir}/cancer_county.shp").to_crs(WI_CRS)
    return wells, tracts, county


def build_idw_grid(wells, k_exp, cell_size=1000, n_neighbors=12, power_eps=1e-6, bounds=None):
    """
    wells: GeoDataFrame of points (projected CRS, meters) with 'nitr_ran'
    k_exp: IDW distance-decay exponent (k in 1/d^k)
    cell_size: grid resolution in meters (1000 = 1km cells; lower = finer/slower)
    n_neighbors: how many nearest wells to use per cell (speeds things up,
                 avoids letting a well 200 miles away matter)
    """
    xs = wells.geometry.x.values
    ys = wells.geometry.y.values
    zs = wells['nitr_ran'].values

    if bounds is None:
        minx, miny, maxx, maxy = wells.total_bounds
    else:
        minx, miny, maxx, maxy = bounds

    grid_x = np.arange(minx, maxx, cell_size)
    grid_y = np.arange(miny, maxy, cell_size)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    tree = cKDTree(np.column_stack([xs, ys]))
    dists, idxs = tree.query(grid_pts, k=n_neighbors)

    dists = np.where(dists == 0, power_eps, dists)  # avoid div-by-zero at well locations
    weights = 1.0 / (dists ** k_exp)
    z_vals = zs[idxs]
    interp = np.sum(weights * z_vals, axis=1) / np.sum(weights, axis=1)

    raster = interp.reshape(gx.shape)
    transform_bounds = (minx, miny, maxx, maxy)
    return raster, grid_x, grid_y, transform_bounds


def zonal_mean_nitrate(raster, grid_x, grid_y, tracts):
    cell_size = grid_x[1] - grid_x[0]
    # top-left corner + cell size; note y-axis flips (raster row 0 = max y)
    transform = Affine(cell_size, 0, grid_x.min(),
                        0, -cell_size, grid_y.max())
    raster_flipped = np.flipud(raster)

    stats = zonal_stats(tracts, raster_flipped, affine=transform,
                         stats=["mean"], nodata=np.nan, all_touched=True)
    tracts = tracts.copy()
    tracts["avg_nitrate"] = [s["mean"] for s in stats]
    return tracts


# Export the nitrate average continuous surface as a raster
def export_raster_geotiff(raster, grid_x, grid_y, out_path="nitrate_idw.tif", crs="EPSG:3070"):
    cell_size = grid_x[1] - grid_x[0]
    transform = Affine(cell_size, 0, grid_x.min(),
                        0, -cell_size, grid_y.max())
    raster_flipped = np.flipud(raster).astype("float32")

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=raster_flipped.shape[0],
        width=raster_flipped.shape[1],
        count=1,
        dtype=raster_flipped.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(raster_flipped, 1)


def run_regression(tracts_with_nitrate):
    df = tracts_with_nitrate.dropna(subset=["avg_nitrate", "canrate"])
    slope, intercept, r_value, p_value, std_err = scistats.linregress(
        df["avg_nitrate"], df["canrate"]
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
        "std_err": std_err,
        "n": len(df),
        "df": df,
    }

def raster_to_png_overlay(raster, bounds_3070, cmap_name="YlOrRd"):
    norm = mcolors.Normalize(vmin=np.nanmin(raster), vmax=np.nanmax(raster))
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(raster))
    rgba[np.isnan(raster)] = (0, 0, 0, 0)  # transparent where no data
    rgba_flipped = np.flipud(rgba)

    buf = io.BytesIO()
    plt.imsave(buf, rgba_flipped, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    img_url = f"data:image/png;base64,{encoded}"

    minx, miny, maxx, maxy = bounds_3070
    transformer = Transformer.from_crs("EPSG:3070", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(minx, miny)
    lon_max, lat_max = transformer.transform(maxx, maxy)

    return img_url, [[lat_min, lon_min], [lat_max, lon_max]]

def make_nitrate_colorbar_html(vmin, vmax, cmap_name="YlOrRd", label="Nitrate (mg/L)"):
    fig, ax = plt.subplots(figsize=(4, 0.4))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name)), cax=ax, orientation="horizontal")
    cb.set_label(label)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()