import uproot 
import ROOT
import os
import numpy as np
import pandas as pd
import awkward as ak
import math as m
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import numpy as np
from numba import jit, njit, types, prange, typed, typeof, int64, float64
from numba.experimental import jitclass
from numba.typed import List
from tqdm import tqdm
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

plt.style.use(hep.style.CMS)

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
    ('puFr', types.float64),
    ('bestRecoEnergy', types.float64),
    ('ev', types.int64)
]

@jitclass(spec_trackster)
class Trackster:
   def __init__(self, energy = 0, x = 0, y = 0, z = 0, eta= 0, phi = 0,  index = 0, puFr = 0, bestRecoEnergy = 0,  ev = -1):
       self.energy = energy
       self.index = index
       self.x = x
       self.y = y
       self.z = z
       self.eta = eta
       self.phi = phi
       self.ev = ev
       self.puFr = puFr
       self.bestRecoEnergy = bestRecoEnergy

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
def create_trackster(trackstersMerged_ev, ri, ev, puFr = 0, bestRecoEnergy = 0):
    energyReco = trackstersMerged_ev.raw_energy[ri]
    xReco = trackstersMerged_ev.barycenter_x[ri]
    yReco = trackstersMerged_ev.barycenter_y[ri]
    zReco = trackstersMerged_ev.barycenter_z[ri]
    etaReco = trackstersMerged_ev.barycenter_eta[ri]
    phiReco = trackstersMerged_ev.barycenter_phi[ri]
    
    tracksterReco = Trackster(energyReco, xReco, yReco, zReco, etaReco, phiReco, ri,  puFr, bestRecoEnergy, ev) 
    
    return tracksterReco

def make_histo(data, minX, maxX, bins, variable_name, yTitle, title, plotName, plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = False):
    labels = [label1, label2]
    fig = plt.figure(figsize = (15,10))

    for i, d in enumerate(data):
      variables = []
      for di in d:
        variables.append(getattr(di, variable_name))
      plt.hist(variables, bins=bins, histtype = 'step', label = labels[i], lw = 2) 
      plt.xlabel(variable_name)
      plt.ylabel(yTitle)
      if(logY):
        plt.yscale('log')
     

    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    plt.xlim(minX,maxX)
    plt.title(title)
    plt.legend()
    plt.savefig(plotDir + "/" + plotName)

def make_histo2d(data, minX, maxX, title, plotName, plotDir, label): 
    fig = plt.figure(figsize = (15,10))
    
    variablesX = []
    variablesY = []
    for di in data:
      variablesX.append(getattr(di, 'bestRecoEnergy'))
      variablesY.append(getattr(di, 'energy'))
    plt.scatter(variablesX, variablesY, label = label)
    plt.xlabel("RecoEnergy")
    plt.ylabel("SimEnergy")
    plt.plot([minX, maxX], [minX, maxX], color='red', linestyle='--')

    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    plt.xlim(minX,maxX)
    plt.ylim(minX,maxX)
    plt.title(title)
    plt.legend()
    plt.savefig(plotDir + "/" + plotName)

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
    return x_values, y_values, y_errors_low, y_errors_high

#    # Create Matplotlib plot
#    fig = plt.figure(figsize = (15,10))
#    plt.errorbar(x_values, y_values, yerr=[y_errors_low, y_errors_high], fmt='o', label='Efficiency')
#    plt.xlim(minX, maxX)
#    plt.ylim(0, 1)
#    plt.xlabel(variable_name)
#    plt.ylabel(yTitle)
#    plt.title(title)
#    plt.legend()
#
#    # Create the "plots" directory if it doesn't exist
#    plotDir = 'plotsNumba' 
#    if not os.path.exists(plotDir):
#        os.makedirs(plotDir)
#
#    # Save the plot
#    plot_filename = os.path.join(plotDir,  f"{plotName}_{variable_name}.png")
#    plt.savefig(plot_filename)

def error_division(num, den, err_num, err_den):
    return num/den * np.sqrt((err_num/num)**2 + (err_den / den)**2)

def compute_ratio(numerator_list, denominator_list , minX, maxX, bins, variable_name, yTitle, title, plotName, plotDir, label1 = 'TICLv5', label2 = 'TICLv4'):
    # Create efficiency plots for the first set of data
    assert(len(numerator_list) ==  len(denominator_list))
    data1 = create_efficiency_plots(numerator_list[0], denominator_list[0], minX, maxX, bins, variable_name, yTitle, title, plotName)
    data2 = create_efficiency_plots(numerator_list[1], denominator_list[1], minX, maxX, bins, variable_name, yTitle, title, plotName)

    # Create efficiency plots for the second set of data

    x1, y1, yL1, yH1 = data1
    x2, y2, yL2, yH2 = data2

    # Calculate the ratio of y values
    ratio = np.nan_to_num(np.divide(y1, y2, out=np.zeros_like(y1), where=y2 != 0))

    # Calculate the error on the ratio using the provided formula
    error_ratio = np.nan_to_num(error_division(np.array(y1),np.array(y2), yL1, yL2))

    # Plot the individual data points
    # Create the main plot
    plt.figure(figsize=(10, 10))

    gs = GridSpec(2, 1, height_ratios=[2, 1])
    plt.subplot(gs[0])  
    plt.errorbar(x1, y1, yerr=[yL1, yH1], fmt='o', label=label1)
    plt.errorbar(x2, y2, yerr=[yL2, yH2], fmt='o', label=label2)
    plt.ylim(0.,1.)
    plt.xlabel(variable_name)
    plt.ylabel(yTitle)
    plt.title(title)
    plt.subplot(gs[1])
    print(x1, ratio, error_ratio)
    plt.errorbar(x1, ratio, yerr=error_ratio, fmt='o', label='Ratio')
    plt.axhline(1, color='gray', linestyle='--', linewidth=1)  # Include a line at y=1 for reference
    plt.xlabel(variable_name)
    plt.ylabel('Ratio')
    plt.savefig(plotDir + "/" + plotName)



fu = "./Data/histo.root"



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
            for ri in range(len(recoToSim_mergeTracksterCP)):
                hasSomeSignal = False
                for si in range(len(recoToSim_mergeTracksterCP_score[ri])):
                    score = recoToSim_mergeTracksterCP_score[ri][si]
                    simIdx = recoToSim_mergeTracksterCP[ri][si]
                    simVertices = simTrackstersCP_ev.vertices_indexes[si]
                    recoVertices = trackstersMerged_ev.vertices_indexes[ri]
                    common_vertices = set(simVertices) & set(recoVertices)

                    # Check additional condition
                    puFraction = recoToSim_mergeTracksterPU_sharedE[ri][0] / trackstersMerged_ev.raw_energy[ri]
                    if (puFraction < 0.95):
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
                    puFraction = recoToSim_mergeTracksterPU_sharedE[ri][0] / energyReco 
                    tracksterReco = create_trackster(trackstersMerged_ev, ri, ev, puFraction, energyReco)
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
                indexMaxShared = simToReco_mergeTracksterCP[si][argmaxShared]
                puFractionMaxShared = recoToSim_mergeTracksterPU_sharedE[indexMaxShared][0] / trackstersMerged_ev.raw_energy[indexMaxShared]
                indexMinScore = simToReco_mergeTracksterCP[si][argmaxShared]
                puFractionMinScore = recoToSim_mergeTracksterPU_sharedE[indexMinScore][0] / trackstersMerged_ev.raw_energy[indexMinScore]
                maxSE = simToReco_mergeTracksterCP_sharedE[si][argmaxShared]
                minScore = simToReco_mergeTracksterCP_score[si][argminScore]
                simTracksterMaxShared = create_trackster(simTrackstersCP_ev, si, ev, puFractionMaxShared, trackstersMerged_ev.raw_energy[indexMaxShared])
                simTracksterMinScore = create_trackster(simTrackstersCP_ev, si, ev, puFractionMinScore, trackstersMerged_ev.raw_energy[indexMinScore])
                all_simtracksters.append(simTracksterMaxShared)
                if(maxSE / simTracksterMaxShared.energy >= 0.5):
                  efficient_simtracksters.append(simTracksterMaxShared)
                if(sumSE > 0.):
                  if(maxSE / sumSE >= 0.5):
                    efficient_simtracksters_corrected.append(simTracksterMaxShared)
                if(minScore <= 0.2):
                  pure_simtracksters.append(simTracksterMinScore)
    return fake_tracksters, merged_tracksters, all_tracksters, efficient_simtracksters, efficient_simtracksters_corrected, all_simtracksters, pure_simtracksters

def load_branch_with_highest_cycle(file_path, branch_name):
    # Open the ROOT file
    file = uproot.open(file_path)

    # Get all keys in the file
    all_keys = file.keys()

    # Filter keys that match the specified branch name
    matching_keys = [key for key in all_keys if key.startswith(branch_name)]
    print(matching_keys)

    if not matching_keys:
        raise ValueError(f"No branch with name '{branch_name}' found in the file.")

    # Find the key with the highest cycle
    highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))

    # Load the branch with the highest cycle
    print(highest_cycle_key)
    branch = file[highest_cycle_key]

    return branch

files = ['./Data/histoBestConfigTICLv5SingleIterationRecord946.root', './Data/histoCLUE3DTICLv4.root']

trackstersMerged_data = list() 
association_data = list() 
simTrackstersCP_data = list()
simTrackstersSC_data = list()

all_trackstersList = [] 
all_simtrackstersList = []
fake_trackstersList = []
efficient_simtrackstersList = []
pure_simtrackstersList = []
efficient_simtracksters_correctedList = []
merged_trackstersList = []

for f in tqdm(files):
    file_path = f
    simtrackstersSC = load_branch_with_highest_cycle(file_path, "ticlDumper/simtrackstersSC")
    simtrackstersCP = load_branch_with_highest_cycle(file_path, "ticlDumper/simtrackstersCP")
    tracksters = load_branch_with_highest_cycle(file_path, "ticlDumper/tracksters")
    trackstersMerged = load_branch_with_highest_cycle(file_path, "ticlDumper/trackstersMerged")
    associations = load_branch_with_highest_cycle(file_path, "ticlDumper/associations")
    tracks = load_branch_with_highest_cycle(file_path, "ticlDumper/tracks")
    simTICLCandidate = load_branch_with_highest_cycle(file_path, "ticlDumper/simTICLCandidate")
    TICLCandidate = load_branch_with_highest_cycle(file_path, "ticlDumper/candidates")
    clusters = load_branch_with_highest_cycle(file_path, "ticlDumper/clusters")

    trackstersMerged_data.append(trackstersMerged.arrays(trackstersMerged.keys()))
    association_data.append(associations.arrays(associations.keys()))
    simTrackstersCP_data.append(simtrackstersCP.arrays(simtrackstersCP.keys()))
    fake_tracksters, merged_tracksters, all_tracksters, efficient_simtracksters, efficient_simtracksters_corrected, all_simtracksters, pure_simtracksters = process_event(association_data, simTrackstersCP_data, trackstersMerged_data)
    all_trackstersList.append(all_tracksters) 
    all_simtrackstersList.append(all_simtracksters)
    fake_trackstersList.append(fake_tracksters)
    efficient_simtrackstersList.append(efficient_simtracksters)
    pure_simtrackstersList.append(pure_simtracksters)
    efficient_simtracksters_correctedList.append(efficient_simtracksters_corrected)
    merged_trackstersList.append(merged_tracksters)


binsEta = 10
binsPhi = 10
binsEnergy = 10
maxEta = 3.0
minEta = 1.5
maxPhi = 3.14
minPhi = -3.14
minEnergy = 0
maxEnergy = 600
minpuFr = 0
maxPuFr = 1
binsPuFr = 10
plotDir = 'plotCLUE3DSingleIterationCLUE3D_EffLowHigh_EffLowHighEta_FakeLowHigh_NumTrackster_record946'

make_histo(fake_trackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Fake Trackster #eta', 'histo_fake_eta', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(fake_trackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Fake Trackster #phi', 'histo_fake_phi', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(fake_trackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Fake Trackster Energy', 'histo_fake_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

make_histo(fake_trackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Fake Trackster #eta', 'log_histo_fake_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(fake_trackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Fake Trackster #phi', 'log_histo_fake_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(fake_trackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Fake Trackster Energy', 'log_histo_fake_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4', logY = True)

make_histo(efficient_simtrackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Efficient SimTracksters #eta', 'histo_eff_eta', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(efficient_simtrackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Efficient SimTracksters #phi', 'histo_eff_phi', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(efficient_simtrackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Efficient SimTracksters Energy', 'histo_eff_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

make_histo(efficient_simtrackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Efficient SimTracksters #eta', 'log_histo_eff_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(efficient_simtrackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Efficient SimTracksters #phi', 'log_histo_eff_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(efficient_simtrackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Efficient SimTracksters Energy', 'log_histo_eff_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4', logY = True)

make_histo(efficient_simtracksters_correctedList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Efficient Corrected SimTracksters #eta', 'histo_effCorr_eta', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(efficient_simtracksters_correctedList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Efficient Corrected SimTracksters #phi', 'histo_effCorr_phi', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(efficient_simtracksters_correctedList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Efficient Corrected SimTracksters Energy', 'histo_effCorr_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

make_histo(efficient_simtracksters_correctedList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Efficient Corrected SimTracksters #eta', 'log_histo_effCorr_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(efficient_simtracksters_correctedList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Efficient Corrected SimTracksters #phi', 'log_histo_effCorr_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(efficient_simtracksters_correctedList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Efficient Corrected SimTracksters Energy', 'log_histo_effCorr_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4', logY = True)

make_histo(all_trackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'all Trackster #eta', 'histo_all_eta', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(all_trackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'all Trackster #phi', 'histo_all_phi', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(all_trackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'all Trackster Energy', 'histo_all_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

make_histo(all_trackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'all Trackster #eta', 'log_histo_all_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(all_trackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'all Trackster #phi', 'log_histo_all_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(all_trackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'all Trackster Energy', 'log_histo_all_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4', logY = True)

make_histo(all_simtrackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'all SimTrackster #eta', 'histo_all_sim_eta', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(all_simtrackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'all SimTrackster #phi', 'histo_all_sim_phi', plotDir, label1= 'TIClv5', label2 = 'TICLv4')
make_histo(all_simtrackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'all SimTrackster Energy', 'histo_all_sim_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

make_histo(all_simtrackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'all SimTrackster #eta', 'log_histo_all_sim_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(all_simtrackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'all SimTrackster #phi', 'log_histo_all_sim_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4', logY = True)
make_histo(all_simtrackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'all SimTrackster Energy', 'log_histo_all_sim_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4', logY = True)

compute_ratio(efficient_simtrackstersList, all_simtrackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Efficiency #eta', 'effSim_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(efficient_simtrackstersList, all_simtrackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Efficiency #phi', 'effSim_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(efficient_simtrackstersList, all_simtrackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Efficiency SimTracksters Energy', 'effSim_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

compute_ratio(efficient_simtracksters_correctedList, all_simtrackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Efficiency Corrected #eta', 'effCorrSim_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(efficient_simtracksters_correctedList, all_simtrackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Efficiency Corrected #phi', 'effCorrSim_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(efficient_simtracksters_correctedList, all_simtrackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Efficiency Corrected SimTracksters Energy', 'effCorrSim_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

compute_ratio(pure_simtrackstersList, all_simtrackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Purity #eta', 'purSim_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(pure_simtrackstersList, all_simtrackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Purity #phi', 'purSim_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(pure_simtrackstersList, all_simtrackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Purity SimTracksters Energy', 'purSim_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

compute_ratio(fake_trackstersList, all_trackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'Fake Rate #eta', 'fake_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(fake_trackstersList, all_trackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'Fake Rate #phi', 'fake_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(fake_trackstersList, all_trackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'Fake Rate Energy', 'fakeSim_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')

compute_ratio(merged_trackstersList, all_trackstersList, minEta, maxEta, binsEta, 'eta', 'Entries', 'merge Rate #eta', 'merge_eta', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(merged_trackstersList, all_trackstersList, minPhi, maxPhi, binsPhi, 'phi', 'Entries', 'merge Rate #phi', 'merge_phi', plotDir, label1 =  'TIClv5', label2 = 'TICLv4')
compute_ratio(merged_trackstersList, all_trackstersList, minEnergy, maxEnergy, binsEnergy, 'energy', 'Entries', 'merge Rate Energy', 'mergeSim_Energy', plotDir, label1 = 'TIClv5', label2 = 'TICLv4')
make_histo2d(efficient_simtrackstersList[0], minEnergy, maxEnergy, 'Sim vs Reco energy Efficient SimTrackster', 'plot2d_efficient_simV5', plotDir, label = 'TICLv5') 
make_histo2d(efficient_simtrackstersList[1], minEnergy, maxEnergy, 'Sim vs Reco energy Efficient SimTrackster', 'plot2d_efficient_simV4', plotDir, label = 'TICLv4') 
make_histo2d(efficient_simtracksters_correctedList[0], minEnergy, maxEnergy, 'Sim vs Reco energy Efficient SimTrackster Corrected', 'plot2d_efficientCorr_simV5', plotDir, label = 'TICLv5') 
make_histo2d(efficient_simtracksters_correctedList[1], minEnergy, maxEnergy, 'Sim vs Reco energy Efficient SimTrackster Corrected', 'plot2d_efficientCorr_simV4', plotDir, label = 'TICLv4') 
make_histo2d(pure_simtrackstersList[0], minEnergy, maxEnergy, 'Sim vs Reco energy Pure SimTrackster', 'plot2d_pure_simV5', plotDir, label = 'TICLv5') 
make_histo2d(pure_simtrackstersList[1], minEnergy, maxEnergy, 'Sim vs Reco energy Pure SimTrackster', 'plot2d_pure_simV4', plotDir, label = 'TICLv4') 

#create_efficiency_plots(fake_tracksters, all_tracksters, 1.5 , 3.0, 10, "eta", "Fake rate", "Fake Rate", 'fake')
#create_efficiency_plots(fake_tracksters, all_tracksters, -m.pi , m.pi, 10, "phi", "Fake rate", "Fake Rate",'fake')
#create_efficiency_plots(fake_tracksters, all_tracksters, 0 , 600, 20, "energy", "Fake rate", "Fake Rate", 'fake')
#
#create_efficiency_plots(efficient_simtracksters, all_simtracksters, 1.5 , 3.0, 10, "eta", "Efficiency", "Efficiency",'eff')
#create_efficiency_plots(efficient_simtracksters, all_simtracksters, -m.pi , m.pi, 10, "phi", "Efficiency", "Efficiency", 'eff')
#create_efficiency_plots(efficient_simtracksters, all_simtracksters, 0 , 600, 20, "energy", "Efficiency", "Efficiency", 'eff')
#
#create_efficiency_plots(efficient_simtracksters_corrected, all_simtracksters, 1.5 , 3.0, 10, "eta", "Efficiency", "Efficiency",'effCorr')
#create_efficiency_plots(efficient_simtracksters_corrected, all_simtracksters, -m.pi , m.pi, 10, "phi", "Efficiency", "Efficiency", 'effCorr')
#create_efficiency_plots(efficient_simtracksters_corrected, all_simtracksters, 0 , 600, 20, "energy", "Efficiency", "Efficiency", 'effCorr')
#
#create_efficiency_plots(pure_simtracksters, all_simtracksters, 1.5 , 3.0, 10, "eta", "Purity", "Purity", 'pur')
#create_efficiency_plots(pure_simtracksters, all_simtracksters, -m.pi , m.pi, 10, "phi", "Purity", "Purity", 'pur')
#create_efficiency_plots(pure_simtracksters, all_simtracksters, 0 , 600, 20, "energy", "Purity", "Purity", 'pur')




