# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: step3 -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM --conditions auto:phase2_realistic_T21 --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO -n 10 --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM --geometry Extended2026D95 --era Phase2C17I13M9 --no_exec --filein file:step2.root --fileout file:step3.root
import numpy as np
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
from Configuration.AlCa.GlobalTag import GlobalTag
import FWCore.ParameterSet.Config as cms
# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
# Reconstruction
from RecoHGCal.TICL.iterativeTICL_cff import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClustersEE, hgcalLayerClustersHSi, hgcalLayerClustersHSci
from RecoLocalCalo.HGCalRecProducers.hgcalMergeLayerClusters_cfi import hgcalMergeLayerClusters
from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper
# Validation
from Validation.HGCalValidation.HGCalValidator_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalRecHitMapProducer_cfi import hgcalRecHitMapProducer

# Load DNN ESSource
from RecoTracker.IterativeTracking.iterativeTk_cff import trackdnn_source

# Automatic addition of the customisation function from RecoHGCal.Configuration.RecoHGCal_EventContent_cff
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseHGCalOnlyEventContent
from SimCalorimetry.HGCalAssociatorProducers.simTracksterAssociatorByEnergyScore_cfi import simTracksterAssociatorByEnergyScore as simTsAssocByEnergyScoreProducer
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationLinking, tracksterSimTracksterAssociationPR, tracksterSimTracksterAssociationLinkingbyCLUE3D, tracksterSimTracksterAssociationPRbyCLUE3D
#, tracksterSimTracksterAssociationLinkingPU, tracksterSimTracksterAssociationPRPU
from RecoHGCal.TICL.simpleValidation_cfi import simpleValidation 

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from FWCore.ParameterSet.VarParsing import VarParsing
from utils import read_csv

# VarParsing instance
options = VarParsing('analysis')

# Custom options
options.register('parametersFile',
              'default/default_params.csv',
              VarParsing.multiplicity.singleton,
              VarParsing.varType.string,
              'Name of parameters file')

options.register('nEvents',
              100,
              VarParsing.multiplicity.singleton,
              VarParsing.varType.int,
              'Number of events')

#options.register('outputFile',
#              'temp/simple_validation.root',
#              VarParsing.multiplicity.singleton,
#              VarParsing.varType.string,
#              'output file validation')

# options.register('inputFile',
#               'file:input/step2.root',
#               VarParsing.multiplicity.singleton,
#               VarParsing.varType.string,
#               'Name of input file')

options.parseArguments()

process = cms.Process('RECO3', Phase2C17I13M9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D95Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
print(f"Running over {options.nEvents}")
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(options.nEvents),
    output=cms.optional.untracked.allowed(cms.int32, cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames=cms.untracked.vstring(
                            ['file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_0.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_1.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_10.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_11.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_13.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_14.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_15.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_16.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_17.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_18.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_19.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_2.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_20.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_21.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_22.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_23.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_24.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_25.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_26.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_27.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_28.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_29.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_3.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_30.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_31.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_32.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_33.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_34.root',
                            'file:/data/user/wredjeb/CA-parameter-tuning/step3/step3_16511637_35.root']
                            ),
                            secondaryFileNames=cms.untracked.vstring()
                            )

process.options = cms.untracked.PSet(
    IgnoreCompletely=cms.untracked.vstring(),
    Rethrow=cms.untracked.vstring(),
    accelerators=cms.untracked.vstring('*'),
    allowUnscheduled=cms.obsolete.untracked.bool,
    canDeleteEarly=cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules=cms.untracked.bool(True),
    dumpOptions=cms.untracked.bool(False),
    emptyRunLumiMode=cms.obsolete.untracked.string,
    eventSetup=cms.untracked.PSet(
        forceNumberOfConcurrentIOVs=cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs=cms.untracked.uint32(0)
    ),
    fileMode=cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun=cms.untracked.bool(False),
    holdsReferencesToDeleteEarly=cms.untracked.VPSet(),
    makeTriggerResults=cms.obsolete.untracked.bool,
    modulesToIgnoreForDeleteEarly=cms.untracked.vstring(),
    numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(0),
    numberOfConcurrentRuns=cms.untracked.uint32(1),
    numberOfStreams=cms.untracked.uint32(0),
    numberOfThreads=cms.untracked.uint32(1),
    printDependencies=cms.untracked.bool(False),
    sizeOfStackForThreadsInKB=cms.optional.untracked.uint32,
    throwIfIllegalParameter=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation=cms.untracked.string('step3 nevts:10'),
    name=cms.untracked.string('Applications'),
    version=cms.untracked.string('$Revision: 1.19 $')
)


# Additional output definition

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases:
    delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string(
    "randomEngineStateProducer")
process.GlobalTag = GlobalTag(
    process.GlobalTag, 'auto:phase2_realistic_T21', '')

# Path and EndPath definitions
process.filteredLayerClustersCLUE3DEM = cms.EDProducer("FilteredLayerClustersProducer",
  LayerClusters = cms.InputTag("hgcalMergeLayerClusters"),
  LayerClustersInputMask = cms.InputTag("hgcalMergeLayerClusters","InitialLayerClustersMask"),
  algo_number = cms.vint32(6, 7), 
  clusterFilter = cms.string('ClusterFilterByAlgoAndSizeAndLayerRange'),
  iteration_label = cms.string('CLUE3DEM'),
  max_cluster_size = cms.int32(9999),
  max_layerId = cms.int32(28),    
  mightGet = cms.optional.untracked.vstring, 
  min_cluster_size = cms.int32(2),
  min_layerId = cms.int32(0)
)

process.ticlSeedingRegionProducer = cms.EDProducer('TICLSeedingRegionProducer',
  seedingPSet = cms.PSet(
    algo_verbosity = cms.int32(0),
    type = cms.string('SeedingRegionGlobal')
  
  ),
  mightGet = cms.optional.untracked.vstring
)


params = read_csv(options.parametersFile)
#params = [[i for i in range(100)]]
totalTask = len(params)
for i, v in enumerate(params):
    setattr(process, 'trackstersProducerCLUE3DEM' + str(i), cms.EDProducer('TrackstersProducer',
        detector = cms.string('HGCAL'),
        filtered_mask = cms.InputTag("filteredLayerClustersCLUE3DEM","CLUE3DEM"),
        itername = cms.string('CLUE3DEM'),
        layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
        layer_clusters_hfnose_tiles = cms.InputTag("ticlLayerTileHFNose"),
        layer_clusters_tiles = cms.InputTag("ticlLayerTileProducer"),
        mightGet = cms.optional.untracked.vstring,
        original_mask = cms.InputTag("hgcalMergeLayerClusters","InitialLayerClustersMask"),
        patternRecognitionBy = cms.string('CLUE3D'),
        pluginPatternRecognitionByCA = cms.PSet(
            algo_verbosity = cms.int32(0),
            eid_input_name = cms.string('input'),
            eid_min_cluster_energy = cms.double(1),
            eid_n_clusters = cms.int32(10),
            eid_n_layers = cms.int32(50),
            eid_output_name_energy = cms.string('output/regressed_energy'),
            eid_output_name_id = cms.string('output/id_probabilities'),
            energy_em_over_total_threshold = cms.double(-1),
            etaLimitIncreaseWindow = cms.double(2.1),
            filter_on_categories = cms.vint32(0),
            max_delta_time = cms.double(3),
            max_longitudinal_sigmaPCA = cms.double(9999),
            max_missing_layers_in_trackster = cms.int32(9999),
            max_out_in_hops = cms.int32(10),
            min_cos_pointing = cms.double(-1),
            min_cos_theta = cms.double(0.915),
            min_layers_per_trackster = cms.int32(10),
            oneTracksterPerTrackSeed = cms.bool(False),
            out_in_dfs = cms.bool(True),
            pid_threshold = cms.double(0),
            promoteEmptyRegionToTrackster = cms.bool(False),
            root_doublet_max_distance_from_seed_squared = cms.double(9999),
            shower_start_max_layer = cms.int32(9999),
            siblings_maxRSquared = cms.vdouble(0.0006, 0.0006, 0.0006),
            skip_layers = cms.int32(0),
            type = cms.string('CA')
        ),
        pluginPatternRecognitionByCLUE3D = cms.PSet(
            algo_verbosity = cms.int32(0),
            criticalDensity = cms.vdouble(v[0],v[1],v[2]),
            criticalEtaPhiDistance = cms.vdouble(0.025, 0.025, 0.025),
            criticalSelfDensity = cms.vdouble(v[3],v[4],v[5]),
            criticalXYDistance = cms.vdouble(v[6],v[7],v[8]),
            criticalZDistanceLyr = cms.vint32(int(v[9]) ,int(v[10]),int(v[11])),
            cutHadProb = cms.double(v[12]),
            densityEtaPhiDistanceSqr = cms.vdouble(0.0008, 0.0008, 0.0008),
            densityOnSameLayer = cms.bool(bool(int(v[13]))),
            densitySiblingLayers = cms.vint32(int(v[14]), int(v[15]), int(v[16])),
            densityXYDistanceSqr = cms.vdouble(v[17],v[18],v[19]),
            doPidCut = cms.bool(True),
            eid_input_name = cms.string('input'),
            eid_min_cluster_energy = cms.double(1),
            eid_n_clusters = cms.int32(10),
            eid_n_layers = cms.int32(50),
            eid_output_name_energy = cms.string('output/regressed_energy'),
            eid_output_name_id = cms.string('output/id_probabilities'),
            kernelDensityFactor = cms.vdouble(v[20], v[21], v[22]),
            minNumLayerCluster = cms.vint32(int(v[23]),int(v[24]),int(v[25])),
            nearestHigherOnSameLayer = cms.bool(False),
            outlierMultiplier = cms.vdouble(v[26],v[27],v[28]),
            rescaleDensityByZ = cms.bool(False),
            type = cms.string('CLUE3D'),
            useAbsoluteProjectiveScale = cms.bool(True),
            useClusterDimensionXY = cms.bool(False)
        ),
        pluginPatternRecognitionByFastJet = cms.PSet(
            algo_verbosity = cms.int32(0),
            antikt_radius = cms.double(0.09),
            eid_input_name = cms.string('input'),
            eid_min_cluster_energy = cms.double(1),
            eid_n_clusters = cms.int32(10),
            eid_n_layers = cms.int32(50),
            eid_output_name_energy = cms.string('output/regressed_energy'),
            eid_output_name_id = cms.string('output/id_probabilities'),
            minNumLayerCluster = cms.int32(5),
            type = cms.string('FastJet')
        ),
        seeding_regions = cms.InputTag("ticlSeedingRegionProducer"),
        tfDnnLabel = cms.string('tracksterSelectionTf'),
        time_layerclusters = cms.InputTag("hgcalMergeLayerClusters","timeLayerCluster")
        )
    )
    setattr(process, 'filteredLayerClustersCLUE3DHAD' + str(i), cms.EDProducer('FilteredLayerClustersProducer',
            LayerClusters = cms.InputTag('hgcalMergeLayerClusters'),
            LayerClustersInputMask = cms.InputTag('trackstersProducerCLUE3DEM' + str(i)),
            iteration_label = cms.string('CLUE3DHAD'),
            clusterFilter = cms.string('ClusterFilterBySize'),
            min_cluster_size = cms.int32(2),
            max_cluster_size = cms.int32(9999),
        )
    )
    setattr(process, 'trackstersProducerCLUE3DHAD' + str(i), cms.EDProducer('TrackstersProducer',
        detector = cms.string('HGCAL'),
        filtered_mask = cms.InputTag("filteredLayerClustersCLUE3DHAD"+str(i),"CLUE3DHAD"),
        itername = cms.string('CLUE3DHAD'),
        layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
        layer_clusters_hfnose_tiles = cms.InputTag("ticlLayerTileHFNose"),
        layer_clusters_tiles = cms.InputTag("ticlLayerTileProducer"),
        mightGet = cms.optional.untracked.vstring,
        original_mask = cms.InputTag("trackstersProducerCLUE3DEM"+str(i)),
        patternRecognitionBy = cms.string('CLUE3D'),
        pluginPatternRecognitionByCA = cms.PSet(
            algo_verbosity = cms.int32(0),
            eid_input_name = cms.string('input'),
            eid_min_cluster_energy = cms.double(1),
            eid_n_clusters = cms.int32(10),
            eid_n_layers = cms.int32(50),
            eid_output_name_energy = cms.string('output/regressed_energy'),
            eid_output_name_id = cms.string('output/id_probabilities'),
            energy_em_over_total_threshold = cms.double(-1),
            etaLimitIncreaseWindow = cms.double(2.1),
            filter_on_categories = cms.vint32(0),
            max_delta_time = cms.double(3),
            max_longitudinal_sigmaPCA = cms.double(9999),
            max_missing_layers_in_trackster = cms.int32(9999),
            max_out_in_hops = cms.int32(10),
            min_cos_pointing = cms.double(-1),
            min_cos_theta = cms.double(0.915),
            min_layers_per_trackster = cms.int32(10),
            oneTracksterPerTrackSeed = cms.bool(False),
            out_in_dfs = cms.bool(True),
            pid_threshold = cms.double(0),
            promoteEmptyRegionToTrackster = cms.bool(False),
            root_doublet_max_distance_from_seed_squared = cms.double(9999),
            shower_start_max_layer = cms.int32(9999),
            siblings_maxRSquared = cms.vdouble(0.0006, 0.0006, 0.0006),
            skip_layers = cms.int32(0),
            type = cms.string('CA')
        ),
        pluginPatternRecognitionByCLUE3D = cms.PSet(
            algo_verbosity = cms.int32(0),
            criticalDensity = cms.vdouble(v[29],v[30],v[31]),
            criticalEtaPhiDistance = cms.vdouble(v[32], v[33], v[34]),
            criticalSelfDensity = cms.vdouble(v[35],v[36],v[37]),
            criticalXYDistance = cms.vdouble(v[38], v[39], v[40]),
            criticalZDistanceLyr = cms.vint32(int(v[41]), int(v[42]), int(v[43])),
            cutHadProb = cms.double(0.5),
            densityEtaPhiDistanceSqr = cms.vdouble(v[44], v[45], v[46]),
            densityOnSameLayer = cms.bool(bool(int(v[47]))),
            densitySiblingLayers = cms.vint32(int(v[48]), int(v[49]), int(v[50])),
            densityXYDistanceSqr = cms.vdouble(v[51], v[52], v[53]),
            doPidCut = cms.bool(False),
            eid_input_name = cms.string('input'),
            eid_min_cluster_energy = cms.double(1),
            eid_n_clusters = cms.int32(10),
            eid_n_layers = cms.int32(50),
            eid_output_name_energy = cms.string('output/regressed_energy'),
            eid_output_name_id = cms.string('output/id_probabilities'),
            kernelDensityFactor = cms.vdouble(v[54], v[55], v[56]),
            minNumLayerCluster = cms.vint32(int(v[57]), int(v[58]), int(v[59])),
            nearestHigherOnSameLayer = cms.bool(False),
            outlierMultiplier = cms.vdouble(v[60], v[61], v[62]),
            rescaleDensityByZ = cms.bool(False),
            type = cms.string('CLUE3D'),
            useAbsoluteProjectiveScale = cms.bool(True),
            useClusterDimensionXY = cms.bool(False)
        ),
        pluginPatternRecognitionByFastJet = cms.PSet(
            algo_verbosity = cms.int32(0),
            antikt_radius = cms.double(0.09),
            eid_input_name = cms.string('input'),
            eid_min_cluster_energy = cms.double(1),
            eid_n_clusters = cms.int32(10),
            eid_n_layers = cms.int32(50),
            eid_output_name_energy = cms.string('output/regressed_energy'),
            eid_output_name_id = cms.string('output/id_probabilities'),
            minNumLayerCluster = cms.int32(5),
            type = cms.string('FastJet')
        ),
        seeding_regions = cms.InputTag("ticlSeedingRegionProducer"),
        tfDnnLabel = cms.string('tracksterSelectionTf'),
        time_layerclusters = cms.InputTag("hgcalMergeLayerClusters","timeLayerCluster")
        )
    )
    setattr(process, 'mergedTrackstersProducer' + str(i), cms.EDProducer("TracksterLinksProducer",
            linkingPSet = cms.PSet(
              track_time_quality_threshold = cms.double(0.5),
              wind = cms.double(1.5),
              angle0 = cms.double(1.523599),
              angle1 = cms.double(1.349006),
              angle2 = cms.double(1.174532),
              maxConeHeight = cms.double(500),
              algo_verbosity = cms.int32(0),
              type = cms.string('Skeletons')
            
            ),
            tracksters_collections = cms.VInputTag(
              'trackstersProducerCLUE3DEM'+str(i),
              'trackstersProducerCLUE3DHAD'+str(i)
            ),
            original_masks = cms.VInputTag('hgcalMergeLayerClusters:InitialLayerClustersMask'),
            layer_clusters = cms.InputTag('hgcalMergeLayerClusters'),
            layer_clustersTime = cms.InputTag('hgcalMergeLayerClusters', 'timeLayerCluster'),
            detector = cms.string('HGCAL'),
            propagator = cms.string('PropagatorWithMaterial'),
            mightGet = cms.optional.untracked.vstring
        )
    )
    setattr(process, 'tracksterSimTracksterAssociationLinkingbyCLUE3D' + str(i), cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
            associator = cms.InputTag('simTracksterHitLCAssociatorByEnergyScoreProducer'),
            label_tst = cms.InputTag("mergedTrackstersProducer" + str(i)),
            label_simTst = cms.InputTag("ticlSimTracksters", "fromCPs"),
            label_lcl = cms.InputTag("hgcalMergeLayerClusters"),
            label_scl = cms.InputTag("mix", "MergedCaloTruth"),
            label_cp = cms.InputTag("mix","MergedCaloTruth"),
            )
    )
   # setattr(process, 'tracksterSimTracksterAssociationLinkingbyCLUE3DPU' + str(i), cms.EDProducer("TSToSimTSHitLCAssociatorEDProducer",
   #         associator = cms.InputTag('simTracksterHitLCAssociatorByEnergyScoreProducer'),
   #         label_tst = cms.InputTag("mergedTrackstersProducer" + str(i)),
   #         label_simTst = cms.InputTag("ticlSimTracksters", "PU"),
   #         label_lcl = cms.InputTag("hgcalMergeLayerClusters"),
   #         label_scl = cms.InputTag("mix", "MergedCaloTruth"),
   #         label_cp = cms.InputTag("mix","MergedCaloTruth"),
   #         )
   # )
    setattr(process, "simpleValidation" + str(i), cms.EDAnalyzer('SimpleValidation',
            trackstersclue3d = cms.InputTag('mergedTrackstersProducer' + str(i)),
            simtrackstersCP = cms.InputTag('ticlSimTracksters', 'fromCPs'),
            layerClusters = cms.InputTag('hgcalMergeLayerClusters'),
            recoToSimAssociatorCP = cms.InputTag('tracksterSimTracksterAssociationLinkingbyCLUE3D' + str(i), 'recoToSim'),
            simToRecoAssociatorCP = cms.InputTag('tracksterSimTracksterAssociationLinkingbyCLUE3D' + str(i), 'simToReco'),
#            recoToSimAssociatorPU = cms.InputTag('tracksterSimTracksterAssociationLinkingbyCLUE3DPU' + str(i), 'simToReco'),
            mightGet = cms.optional.untracked.vstring
            )
    )
    
taskListTrackstersEM = [getattr(process, 'trackstersProducerCLUE3DEM' + str(i)) for i in range(len(params))]
taskListTrackstersHAD = [getattr(process, 'trackstersProducerCLUE3DHAD' + str(i)) for i in range(len(params))]
mergedTrackstersProducerTasks = []

for i in range(len(params)):
    mergedTrackstersProducerTasks.append(getattr(process, 'trackstersProducerCLUE3DEM' + str(i)))
    mergedTrackstersProducerTasks.append(getattr(process, 'filteredLayerClustersCLUE3DHAD' + str(i)))
    mergedTrackstersProducerTasks.append(getattr(process, 'trackstersProducerCLUE3DHAD' + str(i)))
    mergedTrackstersProducerTasks.append(getattr(process, 'mergedTrackstersProducer' + str(i)))
TaskAssociations = [getattr(process, 'tracksterSimTracksterAssociationLinkingbyCLUE3D' + str(i)) for i in range(len(params))]
#TaskAssociations.extend([getattr(process, 'tracksterSimTracksterAssociationLinkingbyCLUE3DPU' + str(i)) for i in range(len(params))])
taskSimpleValidation = [getattr(process, 'simpleValidation' + str(i)) for i in range(len(params))]

process.TFESSource = cms.Task(process.trackdnn_source)
process.hgcalLayerClustersTask = cms.Task(process.hgcalLayerClustersEE,
                                          process.hgcalLayerClustersHSi,
                                          process.hgcalLayerClustersHSci,
                                          process.hgcalMergeLayerClusters)

process.trackstersProducersTask = cms.Task(process.ticlSeedingRegionProducer, process.filteredLayerClustersCLUE3DEM,  *mergedTrackstersProducerTasks)
#process.Tracer = cms.Service('Tracer')
process.TFileService = cms.Service('TFileService', fileName=cms.string(options.outputFile) 
                                   if cms.string(options.outputFile) else 'default.root')
# Path and EndPath definitions
process.TICL = cms.Path(process.TFESSource,
                        process.ticlLayerTileTask,
                        process.trackstersProducersTask)

process.TICL_ValidationProducers = cms.Task(process.hgcalRecHitMapProducer,
                                            process.lcAssocByEnergyScoreProducer,
                                            process.layerClusterCaloParticleAssociationProducer,
                                            process.scAssocByEnergyScoreProducer,
                                            process.layerClusterSimClusterAssociationProducer,
                                            process.simTsAssocByEnergyScoreProducer,
                                            process.simTracksterHitLCAssociatorByEnergyScoreProducer)
process.TICLAssociators = cms.Task(*TaskAssociations)
process.TICLValidation = cms.Path(process.TICL_ValidationProducers, process.TICLAssociators)
#process.consume_step = cms.EndPath(TaskAssociations[0] + TaskAssociations[1])
process.consume_step = cms.EndPath()
for t in taskSimpleValidation:
    process.consume_step += t

process.schedule = cms.Schedule(process.TICL, process.TICLValidation,  process.consume_step)

process.options.wantSummary = True
process.options.numberOfThreads =  16 
process.options.numberOfStreams = 0 

#def customiseTICLFromReco(process):
## TensorFlow ESSource
#                            process.ticlIterationsTask,
#                            process.ticlTracksterMergeTask)
## Validation
#    process.TICL_ValidationProducers = cms.Task(process.hgcalRecHitMapProducer,
#                                                process.lcAssocByEnergyScoreProducer,
#                                                process.layerClusterCaloParticleAssociationProducer,
#                                                process.scAssocByEnergyScoreProducer,
#                                                process.layerClusterSimClusterAssociationProducer,
#                                                process.simTsAssocByEnergyScoreProducer,  process.simTracksterHitLCAssociatorByEnergyScoreProducer, process.tracksterSimTracksterAssociationLinking, process.tracksterSimTracksterAssociationPR, process.tracksterSimTracksterAssociationLinkingbyCLUE3D, process.tracksterSimTracksterAssociationPRbyCLUE3D
#                                               )
#    process.TICL_Validator = cms.Task(process.hgcalValidator)
#    process.TICL_Validation = cms.Path(process.TICL_ValidationProducers,
#                                       process.TICL_Validator
#                                      )
## Path and EndPath definitions
#    process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
#    process.DQMoutput_step = cms.EndPath(process.DQMoutput)
#
## Schedule definition
#    process.schedule = cms.Schedule(process.TICL,
#                                    process.TICL_Validation,
#                                    process.FEVTDEBUGHLToutput_step,
#                                    process.DQMoutput_step)
#
# call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process=setCrossingFrameOn(process)


# Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
process=customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
process=customiseEarlyDelete(process)
# End adding early deletion
