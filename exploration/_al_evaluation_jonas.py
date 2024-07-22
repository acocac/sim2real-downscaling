import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sim2real.utils import get_default_data_processor
from sim2real.test import Evaluator, evaluate_many
from sim2real.config import paths, tune, opt, out, model, data
from dataclasses import replace

from sim2real.config import (
    DataSpec,
    ModelSpec,
    OptimSpec,
    OutputSpec,
    Paths,
    TuneSpec,
    TunerType,
    opt,
    out,
    model,
    data,
    names,
    paths,
    tune,
)

from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev, MeanStddev


## Load evaluator
e = Evaluator(paths, opt, out, data, model, tune, 128, False)

tuned=True
num_stations = 31
num_proposed= 1

t = replace(tune, num_tasks=10000, num_stations=num_stations, era5_frac=0.0)

if tuned:
    e._init_weights(t)
else:
    e._init_weights_era5_baseline()

test_taskset, _ = e._init_testloader(t)

remaining_stations_df = e.dwd_raw.meta_df[~e.dwd_raw.meta_df.STATION_ID.isin(e.train_stations + e.val_stations + e.test_stations)]
test_stations_df = e.dwd_raw.meta_df[e.dwd_raw.meta_df.STATION_ID.isin(e.test_stations)]

X_s = e.raw_aux

## rasterize points
def point_mask(ds, location_df):
    lats = ds["LAT"].values
    lons = ds["LON"].values

    location_df['new_column'] = 0
    location_df = location_df.dissolve(by='new_column')

    geom = location_df.geometry.buffer(0.5)

    y, x = np.meshgrid(lats, lons)

    in_shape = shapely.vectorized.contains(geom.iloc[0], x, y)
    mask = xr.DataArray(in_shape.T, dims=ds.dims, coords=ds.coords).astype(int)
    return mask

station_mask_test = point_mask(X_s, test_stations_df)
station_mask_remaining = point_mask(X_s, remaining_stations_df)

fig, ax = plt.subplots(figsize=(15, 15))
station_mask_test.plot(ax=ax)
test_stations_df.plot(ax=ax, facecolor='none', edgecolor='red')
plt.show()

fig, ax = plt.subplots(figsize=(15, 15))
station_mask_remaining.plot(ax=ax)
remaining_stations_df.plot(ax=ax, facecolor='none', edgecolor='red')
plt.show()

## Greedy Algorithm
greedy_alg = GreedyAlgorithm(
    model=e.model,
    X_t=X_s,
    X_s=X_s,
    #X_t_mask=station_mask,
    #X_s_mask=station_mask_remaining,
    context_set_idx=1,
    target_set_idx=0,
    N_new_context=num_proposed,
    task_loader=test_taskset.task_loader,
)

## set dates
test_tasks = [e.test_set[i] for i in range(50)]
[test_tasks[i].update({'time': test_tasks[i]["time"].to_datetime64()}) for i in
 range(len(test_tasks))]  # quick fix to error with dates, wo modifying deepsensor codebase

## acquisition function
acquisition_fn = Stddev(e.model)

## evaluate acquisition function
X_new_df, acquisition_fn_ds = greedy_alg(acquisition_fn, test_tasks)

