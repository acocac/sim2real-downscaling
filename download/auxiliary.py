import os
import argparse
import rioxarray as rioxr

'''
For Antarctica: downloads coarse-scale (500m) REMA surface elevation from
MEaSUREs BedMachine Antarctica, Version 2, and regrids it to an EASE2 grid.

Requires Earthdata login credentials: https://urs.earthdata.nasa.gov/home

TODO: make this a generic 'download_auxiliary_data' script with --aux_ID input?
'''

### Specify the region
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--region', default='germany')
args = parser.parse_args()

region = args.region

##############################################################################

in_dir = "data/raw/srtm_dem"
os.makedirs(in_dir, exist_ok=True)

if region == 'germany':

    target_gmted2010_download_fpath = 'data/raw/srtm_dem/srtm_germany_dtm.tif'

    if not os.path.exists(target_gmted2010_download_fpath):

        ### download the data
        ##############################################################################

        target_gmted2010_download_fpath_zip = \
            os.path.join('data/raw/srtm_dem', 'GMTED2010.zip')

        if not os.path.exists(target_gmted2010_download_fpath_zip):
            cmd = \
                f'wget -O {target_gmted2010_download_fpath_zip} ' \
                'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/Grid_ZipFiles/mn30_grd.zip'

            os.system(cmd)

        target_gmted2010_download_fpath_unzip = \
            os.path.join('data/raw/srtm_dem', 'GMTED2010')

        if not os.path.exists(target_gmted2010_download_fpath_unzip):
            cmd = f'unzip {target_gmted2010_download_fpath_zip} -d {target_gmted2010_download_fpath_unzip}'

            os.system(cmd)

        dem = rioxr.open_rasterio(os.path.join('data/raw/srtm_dem', 'GMTED2010', 'mn30_grd/w001001.adf'))

        dem_ds = dem.to_dataset(dim="band")
        dem_ds = dem_ds.rename({1:"elevation"})

        dem_ds['elevation'].attrs = {'units': 'm', 'long_name': 'elevation', 'standard_name': 'altitude'}

        # Crop to just UK
        xlim = (5.8, 15.05)
        ylim = (54.95, 47.2)

        dem_grid = dem_ds.sel(x=slice(*xlim), y=slice(*ylim))
        dem_grid.to_netcdf(target_gmted2010_download_fpath)