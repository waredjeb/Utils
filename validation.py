import uproot 
import ROOT
import os
import numpy as np
import pandas as pd
import awkward as ak
import math as m
import mplhep as hep
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use(hep.style.CMS)

class Point3D:
  def __init__(self, x,y,z, eta, phi):
    self.x = x
    self.y = y
    self.z = z
    self.eta = eta
    self.phi = phi


class Trackster:
  def __init__(self, energy: float, barycenter: Point3D, index : int):
    self.energy = energy
    self.barycenter = barycenter
    self.index = index
    self.eta = barycenter.eta
    self.phi = barycenter.phi


def create_efficiency_plots(numerator_list, denominator_list, minX, maxX, bins, variable_name, yTitle, title):
    # Create histograms for passing and total
    h_pass = ROOT.TH1F(f"hist_pass_{variable_name}", f"Passing {variable_name};{variable_name};Counts", bins, minX, maxX)
    h_total = ROOT.TH1F(f"hist_total_{variable_name}", f"Total {variable_name};{variable_name};Counts", bins, minX, maxX)

    # Fill histograms with numerator and denominator values
    for numerator in numerator_list:
        variable_value = getattr(numerator, variable_name)
        h_pass.Fill(variable_value)
        
    for denominator in denominator_list:
        variable_value = getattr(denominator, variable_name)
        h_total.Fill(variable_value)

    # Create a TEfficiency object using histograms
    efficiency = ROOT.TEfficiency(h_pass, h_total)

    # Extract efficiency points and errors
    n_points = efficiency.GetTotalHistogram().GetNbinsX()
    x_values = [efficiency.GetTotalHistogram().GetBinCenter(i) for i in range(1, n_points + 1)]
    y_values = [efficiency.GetEfficiency(i) for i in range(1, n_points + 1)]
    y_errors_low = [efficiency.GetEfficiencyErrorLow(i) for i in range(1, n_points + 1)]
    y_errors_high = [efficiency.GetEfficiencyErrorUp(i) for i in range(1, n_points + 1)]

    # Create Matplotlib plot
    fig = plt.figure(figsize = (15,10))
    plt.errorbar(x_values, y_values, yerr=[y_errors_low, y_errors_high], fmt='o', label='Efficiency')
    plt.xlims(minX, maxX)
    plt.ylims(0, 1)
    plt.xlabel(variable_name)
    plt.ylabel(yTitle)
    plt.title(title)
    plt.legend()

    # Create the "plots" directory if it doesn't exist
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    # Save the plot
    plot_filename = os.path.join("plots", f"efficiency_plot_{variable_name}.png")
    plt.savefig(plot_filename)



fu = "./Data/histo.root"

files = [fu]

trackstersMerged_data = list() 
association_data = list() 
simTrackstersCP_data = list()
simTrackstersSC_data = list()

for f in tqdm(files):
    file = uproot.open(f)
    simtrackstersSC = file["ticlDumper/simtrackstersSC;2"]
    simtrackstersCP = file["ticlDumper/simtrackstersCP;2"]
    tracksters  = file["ticlDumper/tracksters;2"]
    trackstersMerged = file["ticlDumper/trackstersMerged;2"]
    associations = file["ticlDumper/associations"]
    tracks = file["ticlDumper/tracks"]
    simTICLCandidate = file["ticlDumper/simTICLCandidate;1"]
    TICLCandidate = file["ticlDumper/candidates"]
    clusters = file["ticlDumper/clusters;13"]

    trackstersMerged_data.append(trackstersMerged.arrays(trackstersMerged.keys()))
    association_data.append(associations.arrays(associations.keys()))
    simTrackstersCP_data.append(simtrackstersCP.arrays(simtrackstersCP.keys()))


#compute fake rate and duplicate
fake_tracksters = []
merged_tracksters = []
all_tracksters = []
for f in range(len(association_data)):
    association_f = association_data[f]
    simTrackstersCP_f = simTrackstersCP_data[f]
    trackstersMerged_f = trackstersMerged_data[f]
#    for ev in tqdm(range(len(association_f))):
    for ev in tqdm(range(200)):
      association_ev = association_f[ev]
      simTrackstersCP_ev = simTrackstersCP_f[ev]
      trackstersMerged_ev = trackstersMerged_f[ev]
      recoToSim_mergeTracksterCP = association_ev.Mergetracksters_recoToSim_CP
      recoToSim_mergeTracksterCP_score = association_ev.Mergetracksters_recoToSim_CP_score
      recoToSim_mergeTracksterCP_sharedE = association_ev.Mergetracksters_recoToSim_CP_sharedE
      recoToSim_mergeTracksterPU = association_ev.Mergetracksters_recoToSim_PU
      recoToSim_mergeTracksterPU_score = association_ev.Mergetracksters_recoToSim_PU_score
      recoToSim_mergeTracksterPU_sharedE = association_ev.Mergetracksters_recoToSim_PU_sharedE
#      print(len(trackstersMerged_ev.raw_energy), len(recoToSim_mergeTracksterPU_score))
      sts_inTrackster = np.zeros(len(trackstersMerged_ev.raw_energy)) 
      for ri in range(len(recoToSim_mergeTracksterPU_score)):
          stsInTrackster_i = np.array([])
#          for si in range(len(recoToSim_mergeTracksterPU_score[ri])):
#            score = recoToSim_mergeTracksterPU_score[ri][si]
#            simIdx = recoToSim_mergeTracksterPU[ri][si]
#            if(score <= 0.6):
#              sts_inTrackster[ri] += 1
          hasSomeSignal = False
          for si in range(len(recoToSim_mergeTracksterCP_score[ri])):
             score = recoToSim_mergeTracksterCP_score[ri][si]
             simIdx = recoToSim_mergeTracksterCP[ri][si]
             simVertices = simTrackstersCP_ev.vertices_indexes[si]
             recoVertices = trackstersMerged_ev.vertices_indexes[ri]
             common_vertices = set(simVertices) & set(recoVertices)
             if(recoToSim_mergeTracksterPU_sharedE[ri][0] / trackstersMerged_ev.raw_energy[ri] < 0.95):# and len(common_vertices) >= 2):
               hasSomeSignal = True
               if(score <= 0.6):
                 sts_inTrackster[ri] += 1
          if(hasSomeSignal == False):
            sts_inTrackster[ri] = -1

      for ri, sts_in_ri in enumerate(sts_inTrackster):
        if(sts_in_ri > -1):
          energyReco = trackstersMerged_ev.raw_energy[ri]
          xReco = trackstersMerged_ev.barycenter_x[ri]
          yReco = trackstersMerged_ev.barycenter_y[ri]
          zReco = trackstersMerged_ev.barycenter_z[ri]
          etaReco = trackstersMerged_ev.barycenter_eta[ri]
          phiReco = trackstersMerged_ev.barycenter_phi[ri]
          pointReco = Point3D(xReco, yReco,zReco, etaReco,phiReco)
          tracksterReco = Trackster(energyReco, pointReco, ri)
          all_tracksters.append(tracksterReco)
          if(sts_in_ri == 0):
            fake_tracksters.append(tracksterReco)
          if(sts_in_ri > 1):
            merged_tracksters.append(tracksterReco)


print(len(all_tracksters), len(fake_tracksters))
create_efficiency_plots(fake_tracksters, all_tracksters, 1.5 , 3.0, 10, "eta", "Fake rate", "Fake Rate")
create_efficiency_plots(fake_tracksters, all_tracksters, -m.pi , m.pi, 10, "phi", "Fake rate", "Fake Rate")
create_efficiency_plots(fake_tracksters, all_tracksters, 0 , 600, 20, "energy", "Fake rate", "Fake Rate")




