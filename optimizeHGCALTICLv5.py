import optimizer
import subprocess
from utils import get_metrics, write_csv
import numpy as np
import uproot
import argparse
import os

# parsing argument
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--continuing', type=int, action='store')
parser.add_argument('-d', '--default', action='store_true')
parser.add_argument('-p2', '--phase2', action='store_true')
parser.add_argument('-p', '--num_particles', default=200,
                    type=int, action='store')
parser.add_argument('-i', '--num_iterations',
                    default=20, type=int, action='store')
parser.add_argument('-e', '--num_events', default=100,
                    type=int, action='store')
args = parser.parse_args()

# define the lower and upper bounds
# params CLUE3D:
# criticalDensity, criticalSelfDensity, densitySiblingLayers, desntiyXYDistanceSQR, kernelDensityFactor, criticalXYDistance, cridicalZDistanceLyr


##EM PARAMS
#Critical Density
defaults = []
EMParams_lb_criticalDensity = [0.1,0.1,0.1] #0,1,2
EMParams_ub_criticalDensity = [5.0,5.0,5.0]
defaults.extend([0.6,0.6,0.6])
EMParams_lb_criticalSelfDensity = [0.05,0.05,0.05] # 3,4,5
EMParams_ub_criticalSelfDensity = [2. ,2. ,2. ]
defaults.extend([0.15,0.15,0.15])
EMParams_lb_criticalXYDistance = [0.05,0.05,0.05] #6,7,8
EMParams_ub_criticalXYDistance = [2. ,2. ,2. ]
defaults.extend([1.8,1.8,1.8])
EMParams_lb_criticalZDistanceLyr= [1,1,1] #9,10,11
EMParams_ub_criticalZDistanceLyr= [10 ,10 ,10 ] 
defaults.extend([5,5,5])
cutHadProb_lb = [0.1] #12
cutHadProb_ub = [1]
defaults.extend([0.5])
EMParams_lb_densityOnSameLayer = [0]#13
EMParams_ub_densityOnSameLayer = [1]
defaults.extend([0])
EMParams_lb_densitySiblingLayers = [1,1,1]#14,15,16
EMParams_ub_densitySiblingLayers = [10,10,10]
defaults.extend([3,3,3])
EMParams_lb_densityXYDistanceSqr = [1,1,1]#17,18,19
EMParams_ub_densityXYDistanceSqr = [10,10,10]
defaults.extend([3.24,3.24,3.24])
EMParams_lb_kernelDensityFactor = [0.05,0.05,0.05]#20,21,22
EMParams_ub_kernelDensityFactor = [2,2,2]
defaults.extend([0.2,0.2,0.2])
EMParams_lb_minNumLayerCluster = [0,0,0]#23,24,25
EMParams_ub_minNumLayerCluster = [5,5,5]
defaults.extend([2,2,2])
EMParams_lb_outlierMultiplier = [0,0,0]#26,27,28
EMParams_ub_outlierMultiplier = [3,3,3]
defaults.extend([0.2,0.2,0.2])

EMParamsUB = []
EMParamsUB.extend(EMParams_ub_criticalDensity) 
EMParamsUB.extend(EMParams_ub_criticalSelfDensity) 
EMParamsUB.extend(EMParams_ub_criticalXYDistance) 
EMParamsUB.extend(EMParams_ub_criticalZDistanceLyr) 
EMParamsUB.extend(cutHadProb_ub) 
EMParamsUB.extend(EMParams_ub_densityOnSameLayer) 
EMParamsUB.extend(EMParams_ub_densitySiblingLayers) 
EMParamsUB.extend(EMParams_ub_densityXYDistanceSqr) 
EMParamsUB.extend(EMParams_ub_kernelDensityFactor) 
EMParamsUB.extend(EMParams_ub_minNumLayerCluster) 
EMParamsUB.extend(EMParams_ub_outlierMultiplier) 

EMParamsLB = []
EMParamsLB.extend(EMParams_lb_criticalDensity) 
EMParamsLB.extend(EMParams_lb_criticalSelfDensity) 
EMParamsLB.extend(EMParams_lb_criticalXYDistance) 
EMParamsLB.extend(EMParams_lb_criticalZDistanceLyr) 
EMParamsLB.extend(cutHadProb_lb) 
EMParamsLB.extend(EMParams_lb_densityOnSameLayer) 
EMParamsLB.extend(EMParams_lb_densitySiblingLayers) 
EMParamsLB.extend(EMParams_lb_densityXYDistanceSqr) 
EMParamsLB.extend(EMParams_lb_kernelDensityFactor) 
EMParamsLB.extend(EMParams_lb_minNumLayerCluster) 
EMParamsLB.extend(EMParams_lb_outlierMultiplier) 
print(len(EMParamsLB), len(EMParamsUB))

print(f"Default {len(defaults)}")

HADParams_lb_criticalDensity = [0.1,0.1,0.1] #29,30,31
HADParams_ub_criticalDensity = [5.0,5.0,5.0] 
defaults.extend([0.6,0.6,0.6])
HADParams_lb_criticalEtaPhiDistance = [0.005,0.005,0.005] #33,34,35
HADParams_ub_criticalEtaPhiDistance = [0.3,0.3,0.3]
defaults.extend([0.025,0.025,0.025])
HADParams_lb_criticalSelfDensity = [0.05,0.05,0.05] #36,37,38
HADParams_ub_criticalSelfDensity = [2. ,2. ,2. ]
defaults.extend([0.15,0.15,0.15])
HADParams_lb_criticalXYDistance = [0.05,0.05,0.05] #39,40,41
HADParams_ub_criticalXYDistance = [2. ,2. ,2. ]
defaults.extend([1.8,1.8,1.8])
HADParams_lb_criticalZDistanceLyr= [1,1,1] #42,43,44
HADParams_ub_criticalZDistanceLyr= [10 ,10 ,10 ]
defaults.extend([5,5,5])
HADParams_lb_densityEtaPhiDistanceSqr = [0.0001, 0.0001, 0.0001] #45,46,47
HADParams_ub_densityEtaPhiDistanceSqr = [0.001 ,0.001 ,0.001 ]
defaults.extend([0.0008,0.0008,0.0008])
HADParams_lb_densityOnSameLayer = [0] #48
HADParams_ub_densityOnSameLayer = [1]
defaults.extend([0])
HADParams_lb_densitySiblingLayers = [1,1,1]#49,50,51
HADParams_ub_densitySiblingLayers = [10,10,10]
defaults.extend([3,3,3])
HADParams_lb_densityXYDistanceSqr = [1,1,1]#52,53,54
HADParams_ub_densityXYDistanceSqr = [10,10,10]
defaults.extend([3.24,3.24,3.24])
HADParams_lb_kernelDensityFactor = [0.05,0.05,0.05]#55,56,57
HADParams_ub_kernelDensityFactor = [2,2,2]
defaults.extend([0.2,0.2,0.2])
HADParams_lb_minNumLayerCluster = [0,0,0]#58,59,60
HADParams_ub_minNumLayerCluster = [5,5,5]
defaults.extend([2,2,2])
HADParams_lb_outlierMultiplier = [0,0,0]#61,62,63
HADParams_ub_outlierMultiplier = [3,3,3]
defaults.extend([2,2,2])

HADParamsUB = []
HADParamsUB.extend(HADParams_ub_criticalDensity) 
HADParamsUB.extend(HADParams_ub_criticalEtaPhiDistance) 
HADParamsUB.extend(HADParams_ub_criticalSelfDensity) 
HADParamsUB.extend(HADParams_ub_criticalXYDistance) 
HADParamsUB.extend(HADParams_ub_criticalZDistanceLyr) 
HADParamsUB.extend(HADParams_ub_densityEtaPhiDistanceSqr ) 
HADParamsUB.extend(HADParams_ub_densityOnSameLayer) 
HADParamsUB.extend(HADParams_ub_densitySiblingLayers) 
HADParamsUB.extend(HADParams_ub_densityXYDistanceSqr) 
HADParamsUB.extend(HADParams_ub_kernelDensityFactor) 
HADParamsUB.extend(HADParams_ub_minNumLayerCluster) 
HADParamsUB.extend(HADParams_ub_outlierMultiplier ) 

HADParamsLB = []
HADParamsLB.extend(HADParams_lb_criticalDensity) 
HADParamsLB.extend(HADParams_lb_criticalEtaPhiDistance) 
HADParamsLB.extend(HADParams_lb_criticalSelfDensity) 
HADParamsLB.extend(HADParams_lb_criticalXYDistance) 
HADParamsLB.extend(HADParams_lb_criticalZDistanceLyr) 
HADParamsLB.extend(HADParams_lb_densityEtaPhiDistanceSqr ) 
HADParamsLB.extend(HADParams_lb_densityOnSameLayer) 
HADParamsLB.extend(HADParams_lb_densitySiblingLayers) 
HADParamsLB.extend(HADParams_lb_densityXYDistanceSqr) 
HADParamsLB.extend(HADParams_lb_kernelDensityFactor) 
HADParamsLB.extend(HADParams_lb_minNumLayerCluster) 
HADParamsLB.extend(HADParams_lb_outlierMultiplier ) 

ub = []
ub.extend(EMParamsUB)
ub.extend(HADParamsUB)

lb = []
lb.extend(EMParamsLB)
lb.extend(HADParamsLB)

print(len(ub), len(lb))
for p in defaults:
    print(f"{p:.18f}", end=',')
config = 'reconstructionHGCALTICLv5.py'
input_file = 'step3.root'

# run pixel reconstruction and simple validation

working_dir = 'tempNewFake'
def reco_and_validate(params):
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    write_csv(f'{working_dir}/parameters.csv', params)
    validation_result = f'{working_dir}/simple_validation.root'
    subprocess.run(['cmsRun', config, 'nEvents=' + str(args.num_events),
                     f'parametersFile={working_dir}/parameters.csv', 'outputFile=' + validation_result])
    num_particles = len(params)
    with uproot.open(validation_result) as uproot_file:
        #print(f"Get Metric {get_metrics(uproot_file,0)}")
        population_fitness = np.array(
            [get_metrics(uproot_file, i) for i in range(num_particles)], dtype = float)
#    print(f" Pop fitness {population_fitness}, {params}")
    return population_fitness


# get default metrics
if args.default:
    default_params = [[0.6, 0.15, 3., 3.24, 0.2, 1.8, 5.]]
    default_metrics = reco_and_validate(default_params)
    write_csv(f'{working_dir}/default.csv',
              [np.concatenate([default_params[0], default_metrics[0]])])

optimizer.FileManager.working_dir=working_dir
optimizer.FileManager.loading_enabled = False 
optimizer.FileManager.saving_enabled = True

objective = optimizer.Objective([reco_and_validate], num_objectives = 4 )


# create the PSO object
#if not args.continuing:
#    os.system('rm history/*')
pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub, 
            num_particles=args.num_particles, num_iterations=args.num_iterations, 
            inertia_weight=0.5, cognitive_coefficient=1., social_coefficient=2., #num_batch = args.num_batches, 
            max_iter_no_improv=10, optimization_mode='global')
#else:
#    pso = optimizer.MOPSO(objective=objective,lower_bounds=lb, upper_bounds=ub) 

# run the optimization algorithm
pso.optimize()
