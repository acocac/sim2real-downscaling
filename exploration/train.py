from sim2real.train import tune
from sim2real.config import (
    TunerType,
)
from sim2real.train.taskset import NonDetMultiTaskset, Taskset


nums_stations = [4]  # 4, 20, 100, 500?
nums_tasks = [16]  # 400, 80, 16
tuners = [TunerType.naive]
era5_fracs = [0.0]  # , 0.05, 0.1, 0.2, 0.4, 0.8]
tune.run_experiments(nums_stations, nums_tasks, tuners, era5_fracs)


#Taskset