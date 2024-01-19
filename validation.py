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
import numpy as np
from numba import jit, njit, types, prange, typed, typeof, int64, float64
from numba.experimental import jitclass
from numba.typed import List
from tqdm import tqdm
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

spec_trackster = [
    ('energy', types.float64),
    ('index', types.int64),
    ('x', types.float64),  # optional for handling empty lists
    ('y', types.float64),  # optional for handling empty lists
    ('z', types.float64),  # optional for handling empty lists
    ('eta', types.float64),  # optional for handling empty lists
    ('phi', types.float64),  # optional for handling empty lists
    ('ev', types.int64)
]
@jitclass(spec_trackster)
class Trackster:
   def __init__(self, energy = 0, x = 0, y = 0, z = 0, eta= 0, phi = 0,  index = 0, ev = -1):
       self.energy = energy
       self.index = index
       self.x = x
       self.y = y
       self.z = z
       self.eta = eta
       self.phi = phi
       self.ev = ev

@njit
def argmaxNumba(arr):
    if len(arr) == 0:
        raise ValueError("argmax: array is empty")
    
    max_index = 0
    max_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
            max_index = i

    return max_index

@njit
def argminNumba(arr):
    if len(arr) == 0:
        raise ValueError("argmin: array is empty")
    
    min_index = 0
    min_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] < min_value:
            min_value = arr[i]
            min_index = i

    return min_index

@njit
def create_trackster(trackstersMerged_ev, ri, ev):
    energyReco = trackstersMerged_ev.raw_energy[ri]
    xReco = trackstersMerged_ev.barycenter_x[ri]
    yReco = trackstersMerged_ev.barycenter_y[ri]
    zReco = trackstersMerged_ev.barycenter_z[ri]
    etaReco = trackstersMerged_ev.barycenter_eta[ri]
    phiReco = trackstersMerged_ev.barycenter_phi[ri]
    
    tracksterReco = Trackster(energyReco, xReco, yReco, zReco, etaReco, phiReco, ri, ev) 
    
    return tracksterReco

def create_efficiency_plots(numerator_list, denominator_list, minX, maxX, bins, variable_name, yTitle, title, plotName):
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
    plt.xlim(minX, maxX)
    plt.ylim(0, 1)
    plt.xlabel(variable_name)
    plt.ylabel(yTitle)
    plt.title(title)
    plt.legend()

    # Create the "plots" directory if it doesn't exist
    plotDir = 'plotsNumba' 
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    # Save the plot
    plot_filename = os.path.join(plotDir,  f"{plotName}_{variable_name}.png")
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

@njit
def process_event(association_data, simTrackstersCP_data, trackstersMerged_data): 
    num_events = 800
    all_tracksters = [] 
    all_simtracksters = []
    fake_tracksters = []
    efficient_simtracksters = []
    pure_simtracksters = []
    efficient_simtracksters_corrected = []
    merged_tracksters = []
    for f in prange(len(association_data)):
        association_f = association_data[f]
        simTrackstersCP_f = simTrackstersCP_data[f]
        trackstersMerged_f = trackstersMerged_data[f]

        for ev in prange(num_events):
            association_ev = association_f[ev]
            simTrackstersCP_ev = simTrackstersCP_f[ev]
            trackstersMerged_ev = trackstersMerged_f[ev]
            recoToSim_mergeTracksterCP = association_ev.Mergetracksters_recoToSim_CP
            recoToSim_mergeTracksterCP_score = association_ev.Mergetracksters_recoToSim_CP_score
            recoToSim_mergeTracksterCP_sharedE = association_ev.Mergetracksters_recoToSim_CP_sharedE
            recoToSim_mergeTracksterPU = association_ev.Mergetracksters_recoToSim_PU
            recoToSim_mergeTracksterPU_score = association_ev.Mergetracksters_recoToSim_PU_score
            recoToSim_mergeTracksterPU_sharedE = association_ev.Mergetracksters_recoToSim_PU_sharedE

            simToReco_mergeTracksterCP = association_ev.Mergetracksters_simToReco_CP
            simToReco_mergeTracksterCP_score = association_ev.Mergetracksters_simToReco_CP_score
            simToReco_mergeTracksterCP_sharedE = association_ev.Mergetracksters_simToReco_CP_sharedE
            sts_inTrackster = np.zeros(len(trackstersMerged_ev.raw_energy)) 
            for ri in range(len(association_ev.Mergetracksters_recoToSim_CP)):
                hasSomeSignal = False
                for si in range(len(association_ev.Mergetracksters_recoToSim_CP_score[ri])):
                    score = association_ev.Mergetracksters_recoToSim_CP_score[ri][si]
                    simIdx = association_ev.Mergetracksters_recoToSim_CP[ri][si]
                    simVertices = simTrackstersCP_ev.vertices_indexes[si]
                    recoVertices = trackstersMerged_ev.vertices_indexes[ri]
                    common_vertices = set(simVertices) & set(recoVertices)

                    # Check additional condition
                    if (association_ev.Mergetracksters_recoToSim_PU_sharedE[ri][0] / trackstersMerged_ev.raw_energy[ri] < 0.95):
                        hasSomeSignal = True
                        if score <= 0.6:
                            sts_inTrackster[ri] += 1

                if not hasSomeSignal:
                    sts_inTrackster[ri] = -1

            for ri, sts_in_ri in enumerate(sts_inTrackster):
                if sts_in_ri > -1:
                    energyReco = trackstersMerged_ev.raw_energy[ri]
                    xReco = trackstersMerged_ev.barycenter_x[ri]
                    yReco = trackstersMerged_ev.barycenter_y[ri]
                    zReco = trackstersMerged_ev.barycenter_z[ri]
                    etaReco = trackstersMerged_ev.barycenter_eta[ri]
                    phiReco = trackstersMerged_ev.barycenter_phi[ri]
                    pointReco = np.array([xReco, yReco, zReco, etaReco, phiReco], dtype=np.float64)
                    tracksterReco = create_trackster(trackstersMerged_ev, ri, ev)
                    all_tracksters.append(tracksterReco)

                    if sts_in_ri == 0:
                        fake_tracksters.append(tracksterReco)
                    elif sts_in_ri > 1:
                        merged_tracksters.append(tracksterReco)
        # Efficiency and Purity
            for si in range(len(simToReco_mergeTracksterCP)):
                simEnergy = simTrackstersCP_ev.raw_energy[si]
                sharedSI= simToReco_mergeTracksterCP_sharedE[si]
                sumSE = 0
                for sE in sharedSI:
                  sumSE += sE
                argmaxShared = argmaxNumba(simToReco_mergeTracksterCP_sharedE[si])
                argminScore= argminNumba(simToReco_mergeTracksterCP_score[si])
                maxSE = simToReco_mergeTracksterCP_sharedE[si][argmaxShared]
                minScore = simToReco_mergeTracksterCP_score[si][argminScore]
                simTrackster = create_trackster(simTrackstersCP_ev, si, ev)
                all_simtracksters.append(simTrackster)
                if(maxSE / simTrackster.energy >= 0.5):
                  efficient_simtracksters.append(simTrackster)
                if(sumSE > 0.):
                  if(maxSE / sumSE >= 0.5):
                    efficient_simtracksters_corrected.append(simTrackster)
                if(minScore <= 0.2):
                  pure_simtracksters.append(simTrackster)
    return fake_tracksters, merged_tracksters, all_tracksters, efficient_simtracksters, efficient_simtracksters_corrected, all_simtracksters, pure_simtracksters



fake_tracksters, merged_tracksters, all_tracksters, efficient_simtracksters, efficient_simtracksters_corrected, all_simtracksters, pure_simtracksters = process_event(association_data, simTrackstersCP_data, trackstersMerged_data)

print(len(fake_tracksters), len(merged_tracksters), len(all_tracksters), len(efficient_simtracksters_corrected), len(efficient_simtracksters), len(all_simtracksters))

create_efficiency_plots(fake_tracksters, all_tracksters, 1.5 , 3.0, 10, "eta", "Fake rate", "Fake Rate", 'fake')
create_efficiency_plots(fake_tracksters, all_tracksters, -m.pi , m.pi, 10, "phi", "Fake rate", "Fake Rate",'fake')
create_efficiency_plots(fake_tracksters, all_tracksters, 0 , 600, 20, "energy", "Fake rate", "Fake Rate", 'fake')

create_efficiency_plots(efficient_simtracksters, all_simtracksters, 1.5 , 3.0, 10, "eta", "Efficiency", "Efficiency",'eff')
create_efficiency_plots(efficient_simtracksters, all_simtracksters, -m.pi , m.pi, 10, "phi", "Efficiency", "Efficiency", 'eff')
create_efficiency_plots(efficient_simtracksters, all_simtracksters, 0 , 600, 20, "energy", "Efficiency", "Efficiency", 'eff')

create_efficiency_plots(efficient_simtracksters_corrected, all_simtracksters, 1.5 , 3.0, 10, "eta", "Efficiency", "Efficiency",'effCorr')
create_efficiency_plots(efficient_simtracksters_corrected, all_simtracksters, -m.pi , m.pi, 10, "phi", "Efficiency", "Efficiency", 'effCorr')
create_efficiency_plots(efficient_simtracksters_corrected, all_simtracksters, 0 , 600, 20, "energy", "Efficiency", "Efficiency", 'effCorr')

create_efficiency_plots(pure_simtracksters, all_simtracksters, 1.5 , 3.0, 10, "eta", "Purity", "Purity", 'pur')
create_efficiency_plots(pure_simtracksters, all_simtracksters, -m.pi , m.pi, 10, "phi", "Purity", "Purity", 'pur')
create_efficiency_plots(pure_simtracksters, all_simtracksters, 0 , 600, 20, "energy", "Purity", "Purity", 'pur')




