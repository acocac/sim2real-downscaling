import numpy as np

from sim2real.test import Evaluator

from sim2real.config import (
    opt,
    out,
    model,
    data,
    paths,
    tune,
)

from dataclasses import replace
from copy import deepcopy
import shapely
import geopandas as gpd
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev, MeanStddev, OracleRMSE

from sim2real.utils import ensure_dir_exists

def corr_df(acqfn_names):
    acqfn0_flat_df = pd.DataFrame([acqfn0_flat_dict[name] for name in acqfn_names], index=acqfn_names).transpose()
    acqfn_corr = acqfn0_flat_df.corr()
    return acqfn_corr

def country_mask(ds):
    lats = ds["LAT"].values
    lons = ds["LON"].values

    country = gpd.read_file(paths.shapefile)
    geom = country.geometry[0]

    y, x = np.meshgrid(lats, lons)
    in_shape = shapely.vectorized.contains(geom, x, y)
    mask = xr.DataArray(in_shape.T, dims=ds.dims, coords=ds.coords).astype(int)
    return mask

def save_acq_ds(acquisition_fn_ds, acquisition_fn, num_stations, tuned):
    finetuned_str = "_tuned" if tuned else ""
    fpath = f"_outputs/al/acq_{acquisition_fn}_{num_stations}{finetuned_str}.nc"
    ensure_dir_exists(fpath)
    acquisition_fn_ds.to_netcdf(fpath)

def save_X_new_df(X_new_df, acquisition_fn, num_stations, tuned):
    finetuned_str = "_tuned" if tuned else ""
    fpath = f"_outputs/al/x_{acquisition_fn}_{num_stations}{finetuned_str}.csv"
    ensure_dir_exists(fpath)
    X_new_df.to_csv(fpath)

e = Evaluator(paths, opt, out, data, model, tune, 1024, False)

tuned=True
num_stations = 500
num_proposed= 1

t = replace(tune, num_tasks=10000, num_stations=num_stations, era5_frac=0.0)

if tuned:
    e._init_weights(t)
else:
    e._init_weights_era5_baseline()

test_taskset, _ = e._init_testloader(t)

test_stations_df = e.dwd_raw.meta_df[e.dwd_raw.meta_df.STATION_ID.isin(e.test_stations)]
stn_locs_df = test_stations_df.set_index(['LAT', 'LON'])
stn_locs_df = stn_locs_df[stn_locs_df.TO_DATE.dt.year >= 2012]

X_t = stn_locs_df
X_s = e.raw_aux.coarsen({"LAT": 15, "LON": 15}, boundary="trim").mean()

greedy_gt_args = {
    'task_loader': e.test_set.task_loader
}

## Greedy Algorithm
greedy_alg = GreedyAlgorithm(
    model=e.model,
    X_t=X_t,
    X_s=X_s,
    #X_t_mask=ger_mask(X_s),
    X_s_mask=country_mask(X_s),
    context_set_idx=0,
    target_set_idx=0,
    N_new_context=1,
    progress_bar=True,
    ** greedy_gt_args
)

## create tasks
tyear = [2013, 2014]
test_dates = [e.test_set[i]['time'] for i in range(len(e.test_set)) if e.test_set[i]['time'].year not in tyear]

test_tasks = test_taskset.task_loader(test_dates, ('all', 'all'), 'all')

# acquisition function
acquisition_fn = MeanStddev(e.model)

## evaluate acquisition function
X_new_df_meanstd, acquisition_fn_ds_meanstd = greedy_alg(acquisition_fn, test_tasks, diff=True)
save_acq_ds(acquisition_fn_ds_meanstd, 'MeanStddev', num_stations, tuned)
save_X_new_df(X_new_df_meanstd, 'MeanStddev', num_stations, tuned)

## acquisition function
acquisition_fn = OracleRMSE(e.model)

## evaluate acquisition function
X_new_df_oraclermse, acquisition_fn_ds_oraclermse = greedy_alg(acquisition_fn, test_tasks, diff=True)
save_acq_ds(acquisition_fn_ds_oraclermse, 'OracleRMSE', num_stations, tuned)
save_X_new_df(X_new_df_oraclermse, 'OracleRMSE', num_stations, tuned)

## load files
finetuned_str = "_tuned" if tuned else ""

acquisition_fn_ds_meanstd = xr.open_dataset(f"_outputs/al/acq_MeanStddev_{num_stations}{finetuned_str}.nc")
acquisition_fn_ds_oraclermse = xr.open_dataset(f"_outputs/al/acq_OracleRMSE_{num_stations}{finetuned_str}.nc")

flat_meanstd = acquisition_fn_ds_meanstd['acquisition_fn'].sel(iteration=0).mean(dim='time').values.flatten()
flat_oraclermse = acquisition_fn_ds_oraclermse['acquisition_fn'].sel(iteration=0).mean(dim='time').values.flatten()

acqfn0_flat_dict = {'meanstd': flat_meanstd, 'oraclermse': flat_oraclermse}

acqfn_corr_tune = corr_df(acqfn_names.keys)
acqfn_corr_tune