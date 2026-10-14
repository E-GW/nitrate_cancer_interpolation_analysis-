import os
os.environ["PROJ_LIB"] = r"C:\venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_DATA"] = r"C:\venv\Lib\site-packages\rasterio\proj_data"

from idw import load_data, build_idw_grid, zonal_mean_nitrate, run_regression, export_raster_geotiff


wells, tracts, county = load_data()

minx, miny, maxx, maxy = tracts.total_bounds
buffer = 20000  # meters — pad 20km outward on every side
padded_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

raster, gx, gy, bounds = build_idw_grid(wells, k_exp=2, cell_size=2000, bounds=padded_bounds)
tracts_with_nitrate = zonal_mean_nitrate(raster, gx, gy, tracts)

#export_raster_geotiff(raster, gx, gy, "nitrate_k2.tif") # Export the nitrate average as a raster

import matplotlib.pyplot as plt
tracts_with_nitrate.plot(column=tracts_with_nitrate["avg_nitrate"].isna(), legend=True, cmap="coolwarm")
plt.title("Tracts with missing nitrate estimate (red = missing)")
tracts.plot(column=(tracts["canrate"] == 0.01), legend=True, cmap="coolwarm")
plt.title("Tracts where canrate = 0.01 (red = TRUE)")
plt.show()
'''
print(tracts_with_nitrate[tracts_with_nitrate["avg_nitrate"].isna()])
print(tracts_with_nitrate["avg_nitrate"].isna().sum())  # should now be 0 or much lower

print((tracts["canrate"] == 0.01).sum())
print(tracts["canrate"].value_counts().head(10))


df_all = tracts_with_nitrate.dropna(subset=["avg_nitrate", "canrate"])
df_filtered = df_all[df_all["canrate"] != 0.01]

result_all = run_regression(tracts_with_nitrate)  # includes the 0.01 tracts
# then re-run linregress manually on df_filtered, or refactor run_regression 
# to accept a pre-filtered dataframe
result_filtered = run_regression(df_filtered)

print("\n--- Including canrate=0.01 tracts ---")
print(f"R²={result_all['r_squared']:.4f}  p={result_all['p_value']:.4g}  "
      f"slope={result_all['slope']:.5f}  n={result_all['n']}")

print("\n--- Excluding canrate=0.01 tracts ---")
print(f"R²={result_filtered['r_squared']:.4f}  p={result_filtered['p_value']:.4g}  "
      f"slope={result_filtered['slope']:.5f}  n={result_filtered['n']}")
'''
for k_test in [1.5, 2, 3, 4, 5]:    
        raster, gx, gy, bounds = build_idw_grid(wells, k_test, cell_size=2000)    
        tn = zonal_mean_nitrate(raster, gx, gy, tracts)    
        r = run_regression(tn)    
        print(f"k={k_test}: R²={r['r_squared']:.4f}  p={r['p_value']:.4g}  slope={r['slope']:.5f}")    
