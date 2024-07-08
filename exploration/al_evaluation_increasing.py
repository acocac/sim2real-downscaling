import exploration.al
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

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

from sim2real.train.trainer import Trainer
from sim2real.test import Evaluator
from deepsensor.plot import offgrid_context

import deepsensor


tuned = True
num_stations = 31
num_proposed = 45 - num_stations
e = Evaluator(paths, opt, out, data, model, tune, 1024, False)

newplace_path = f'_outputs/active_learning/x_{num_stations}_tuned.csv'
newplace_df = pd.read_csv(newplace_path, index_col='iteration')
newplace_df.columns = ['LAT', 'LON']
raw_newplace = newplace_df.to_numpy().transpose()
X_new = e.model.data_processor.map_coord_array(raw_newplace)

meta_df = e.dwd_raw.meta_df
meta_df_training = meta_df[~meta_df.STATION_ID.isin(e.test_stations)]
stn_locs_df = meta_df_training[['LAT', 'LON']]
raw_stn_locs = stn_locs_df.to_numpy().transpose()
X_stn = e.model.data_processor.map_coord_array(raw_stn_locs)

dates = [e.test_set[i]['time'] for i in range(50)]

tasks = e.test_set.task_loader(dates, ('all', 'all'))
fig = deepsensor.plot.task(tasks[0], e.test_set.task_loader)
plt.show()

mae_list = []
for i in range(num_proposed + 1):
    X_added = np.concatenate((X_stn, X_new[:, :i]), axis=1)
    tasks = e.test_set.task_loader(dates, (X_added, 'all'))

    df = e.deterministic_results(tasks)

    sqrt_N = np.sqrt(df.shape[0])
    mae = (df["mean"] - df["VWC"]).abs().mean()
    mae_std = (df["mean"] - df["VWC"]).abs().std() / sqrt_N

    mae_list.append(mae)
