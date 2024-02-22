from sim2real import datasets
from sim2real.config import Paths, paths, names, data, out



a = datasets.DWDStationData(paths)
a.full()
o = a._load_data(paths)
o.head()

o[0]['T2M'].index.min()