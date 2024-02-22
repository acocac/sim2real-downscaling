import os
import argparse
import cdsapi

import xarray as xr
import numpy as np

# Argument parser ###############################

parser = argparse.ArgumentParser()
parser.add_argument('--region', default='europe')
parser.add_argument('--var', default='tcc')

args = parser.parse_args()

region = args.region
var = args.var

#################################################

overwrite = True # Re-download already present yearly files?

#################################################

# Regions' lat-lon boundaries
if region == 'global':
    area = [90, -180, -90, 180] # N, W, S, E
elif region == 'europe':
    area = [75, -15, 30, 45]
elif region == 'usa':
    area = [50, -125, 25, -60]
elif region == 'china':
    area = [45, 100, 15, 145]
elif region == 'amazon':
    area = [15, -85, -30, -30]
elif region == 'africa':
    area = [15, -15, -15, 30]
elif region == 'germany':
    area = [47.2, 5.8, 54.95, 15.05]
else:
    pass

# Years
years = [str(year) for year in range(2021, 2021+1)]

months = [
    '01', '02', '03', '04', '05', '06',
    '07', '08', '09', '10', '11', '12'
]

days = [
    '01', '02', '03',
    '04', '05', '06',
    '07', '08', '09',
    '10', '11', '12',
    '13', '14', '15',
    '16', '17', '18',
    '19', '20', '21',
    '22', '23', '24',
    '25', '26', '27',
    '28', '29', '30',
    '31',
]

times = [
    '00:00', '01:00', '02:00',
    '03:00', '04:00', '05:00',
    '06:00', '07:00', '08:00',
    '09:00', '10:00', '11:00',
    '12:00', '13:00', '14:00',
    '15:00', '16:00', '17:00',
    '18:00', '19:00', '20:00',
    '21:00', '22:00', '23:00',
]

variables = {
    'ssr': 'surface_net_solar_radiation',
    'tcc': 'total_cloud_cover',
    'T2M': '2m_temperature'
}

varnames = [
            '2m_temperature', 'high_vegetation_cover', 'low_vegetation_cover',
            'surface_net_solar_radiation', 'surface_net_thermal_radiation', 'surface_pressure',
            'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards', 'toa_incident_solar_radiation',
            'top_net_solar_radiation', 'top_net_thermal_radiation', 'total_cloud_cover',
            'total_precipitation', 'type_of_high_vegetation', 'type_of_low_vegetation',
        ]

# Download --------------------------------------
cds = cdsapi.Client()

for year in years:
    skipyear = False
    print(f'\n{year}')

    # set up target folders for raw and processed data
    raw_var_folder = f'data/{region}/gridded/ERA5/raw/{var}'
    proc_var_folder = f'data/{region}/gridded/ERA5/processed/{var}'
    for folder in [raw_var_folder, proc_var_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # define download file path and processed/averaged file path
    download_path = os.path.join(raw_var_folder, f'{var}_latlon_hourly_{year}.nc')
    daily_latlon_fpath = os.path.join(raw_var_folder, f'{var}_latlon_{year}.nc')

    # check if file already exists
    if os.path.exists(daily_latlon_fpath) and not overwrite:
        print(f'Skipping this year due to already existing at: {daily_latlon_fpath}')
        skipyear = True

    if skipyear is True:
        continue

    # API request parameter
    retrieve_dict = {
        'product_type': 'reanalysis',
        'variable': variables[var],
        'year': year,
        'month': months,
        'day': days,
        'time': times,
        'area': area,
        'format': 'netcdf'
    }
    
    # define dataset based on required var or year etc
    dataset_str = 'reanalysis-era5-single-levels'

    # Download 3-hourly data ------------------------------
    print(f'Downloading data for {var}...')
    # CDS API request
    cds.retrieve(dataset_str, retrieve_dict, download_path)
    print('Download completed.')

    # Compute daily average --------------------------------
    print('Computing daily averages...')
    da = xr.open_dataarray(download_path)
    da_daily = da.resample(time='1D').reduce(np.mean)
    da_daily.to_netcdf(daily_latlon_fpath)

    # Delete 3-hourly data
    da.close()
    os.remove(download_path)
    print('Done.')