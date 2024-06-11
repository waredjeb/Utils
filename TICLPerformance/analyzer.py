import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
import os
import pyarrow.feather as feather 
import awkward as ak
from plotUtils import * 
from utils import *
# Define the directory path

# List of files in the directory
directories = ['DoubleKaonPU0_v5/', 'DoubleKaonPU0_v5/']

# Dictionary to store dataframes
all_dataframes = []


# Loop through each file and load it into a dataframe
for i, directory in enumerate(directories):
  files = os.listdir(directory)
  dataframes = {}
  for file in files:
      if file.endswith('.arrow'):
          filename = os.path.join(directory, file)
          # Assuming you're using Arrow format for data storage
  #        table = feather.read_feather(filename)
          dataframes[file.split('.')[0]] = pd.read_feather(filename) 
  all_dataframes.append(dataframes)


collection_to_process = 'trackstersMerged'


####################### Plots Tracksters variables  ######################################
labels = ["TICLv5", "TICLv4"]
if(len(all_dataframes) == 1):
  all_dataframes = all_dataframes[0]
  labels = labels[0]

eosPlotDir = "/eos/user/w/wredjeb/www/HGCAL/TICLv5Performance/"
directoryPlot = 'Comparison/'

collection = 'simtrackstersCP'
plot_trackster(collection, eosPlotDir, all_dataframes, directoryPlot, labels = labels )
collection = 'trackstersMerged'
plot_trackster(collection, eosPlotDir, all_dataframes, directoryPlot, labels = labels)
collection = 'tracksters'
plot_trackster(collection, eosPlotDir, all_dataframes, directoryPlot, labels = labels)




efficiencyDir = eosPlotDir + directoryPlot + "/trackstersMerged"

simTrackstersCP = []
trackstersMerged = []
associations = []
for dataframes in  all_dataframes:
  simTrackstersCP.extend([dataframes["simtrackstersCP"]])
  trackstersMerged.extend([dataframes["trackstersMerged"]])
  associations.extend([dataframes["associations"]])
R = RatioPlot(efficiencyDir, None)


R.compute_efficiency(simTrackstersCP, trackstersMerged, associations, variable = "raw_energy", label = labels, xLabel = "Raw Energy [GeV]", bins = 20)
R.compute_efficiency(simTrackstersCP, trackstersMerged, associations, variable = "barycenter_eta", label = labels, xLabel = "eta", bins = 10, rangeV = (1.5,3.0))
R.compute_efficiency(simTrackstersCP, trackstersMerged, associations, variable = "barycenter_phi", label = labels, xLabel = "phi", bins = 10)

R.compute_merge_rate(simTrackstersCP, trackstersMerged, associations, variable = "raw_energy", label = labels, xLabel = "Raw Energy [GeV]", bins = 20)
R.compute_merge_rate(simTrackstersCP, trackstersMerged, associations, variable = "barycenter_eta", label = labels, xLabel = "eta", bins = 10, rangeV = (1.5, 3.0))
R.compute_merge_rate(simTrackstersCP, trackstersMerged, associations, variable = "barycenter_phi", label = labels, xLabel = "phi", bins = 10)

R.compute_fake_rate(simTrackstersCP, trackstersMerged, associations, variable = "raw_energy", label = labels, xLabel = "Raw Energy [GeV]", bins = 20)
R.compute_fake_rate(simTrackstersCP, trackstersMerged, associations, variable = "barycenter_eta", label = labels, xLabel = "eta", bins = 10, rangeV = (1.5,3.0))
R.compute_fake_rate(simTrackstersCP, trackstersMerged, associations, variable = "barycenter_phi", label = labels, xLabel = "phi", bins = 10)


print(associations[0].columns)

