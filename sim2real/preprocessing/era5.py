# %%
from sim2real.datasets import load_era5
from sim2real.config import paths, names
import xarray as xr
import pandas as pd
import glob

def process_era5(cutoff_time):
    cutoff_time = pd.to_datetime(cutoff_time)
    print("Loading raw ERA5 dataset...")
    paths_list = []
    paths_ind = glob.glob(f'{paths.raw_era5}/*.nc')
    paths_list += paths_ind
    era5 = xr.open_mfdataset(paths_list)

    era5 = era5.rename(
        {
            "t2m": names.temp,
            "latitude": names.lat,
            "longitude": names.lon,
            "time": names.time,
        }
    )
    era5[names.temp] -= 273.15
    era5 = era5.where(era5[names.time] > cutoff_time, drop=True)
    era5.to_netcdf(paths.era5)


if __name__ == "__main__":
    cutoff_time = "2000-01-01"
    process_era5(cutoff_time)
