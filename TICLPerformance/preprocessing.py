import pandas as pd
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from glob import glob
from Processor import *
plt.style.use(hep.style.CMS)

path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/PSOPerformance/CMSSW_14_X/D99/DoubleKaonPU0_v5/histo/"
file_paths = glob(path + "*root")
print(file_paths)
# Create an instance of MultipleRootFileProcessor with optional entry range
dProcess = {
            'simtrackstersSC': [],
            'clusters': []
            }
multi_processor = MultipleRootFileProcessor(file_paths, list_entries = dProcess, number_of_threads = 10)

# Get a specific DataFrame
tracksters_df = multi_processor.get_dataframe('tracksters')
print(f"DataFrame for 'tracksters':\n{tracksters_df.head()}\n")

# Get all DataFrames
all_dfs = multi_processor.get_all_dataframes()

multi_processor.save_all_dataframes('DoubleKaonPU0_v5', format='arrow')


