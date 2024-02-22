import xarray as xr
import matplotlib.pyplot as plt
import glob
import pandas as pd
from sim2real.config import paths

# ERA 5
## single
d = xr.open_dataset('data/germany/gridded/ERA5/raw/T2M/T2M_latlon_2012.nc')
d.isel(time=0)['t2m'].plot()
plt.show()

## multiple
paths_list = []
paths = glob.glob('data/germany/gridded/ERA5/raw/T2M/*.nc')
paths_list += paths
era5 = xr.open_mfdataset(paths_list)

cutoff_time = "2000-01-01"
cutoff_time = pd.to_datetime(cutoff_time)
era5 = xr.load_dataset(paths.raw_era5)
era5 = era5.rename(
    {
        "t2m": "T2M",
        "latitude": "LAT",
        "longitude": "LON",
        "time": "TIME",
    }
)
era5["T2M"] -= 273.15
era5 = era5.where(era5["TIME"] > cutoff_time, drop=True)
era5.to_netcdf(paths.era5)

## Elevation
elevation = xr.open_dataset('data/raw/srtm_dem/srtm_germany_dtm.tif')

## station
df = pd.read_feather(paths.dwd)
df_meta = pd.read_feather(paths.dwd_meta)
df_splits = pd.read_feather(paths.station_split)
