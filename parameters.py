"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""


'''
List of shared parameters 
'''


class Parameters(object):

    # SLAM threads
    # True: move local mapping on a separate thread, False: tracking and then local mapping in a single thread
    kLocalMappingOnSeparateThread = True
    kTrackingWaitForLocalMappingToGetIdle = False
    kTrackingWaitForLocalMappingSleepTime = 0.5  # 0.5  # -1 for no sleep # [s]
    kLocalMappingParallelKpsMatching = True
    kLocalMappingParallelKpsMatchingNumWorkers = 4

    # Number of desired keypoints per frame
    kNumFeatures = 2000

    # Point triangulation
    # 0.99998   # max cos angle for triangulation (min parallax angle) in the Initializer
    kCosMaxParallaxInitializer = 0.99998
    # 0.9998                 # max cos angle for triangulation (min parallax angle)
    kCosMaxParallax = 0.9999

    # Point visibility
    # must be viewing cos < kViewingCosLimitForPoint (viewing angle must be less than 60 deg)
    kViewingCosLimitForPoint = 0.5
    kScaleConsistencyFactor = 1.5
    kMaxDistanceToleranceFactor = 1.2
    kMinDistanceToleranceFactor = 0.8

    # Feature management
    # default value; can be changed by selected feature
    kSigmaLevel0 = 1.0
    kFeatureMatchRatioTest = 0.7
    # kFeatureMatchRatioTestInitializer        # ratio test used by Initializer
    #
    kKdtNmsRadius = 3  # pixels  #3        # radius for kd-tree based Non-Maxima Suppression
    #
    kCheckFeaturesOrientation = True

    # Initializer
    # when initializing, the initial median depth is computed and forced to this value (for better visualization is > 1)
    kInitializerDesiredMedianDepth = 20
    kMinRatioBaselineDepth = 0.01
    # kMinTraslation = 0.01*kInitializerDesiredMedianDepth  # not used at the present time
    kInitializerNumMinFeatures = 100
    kInitializerNumMinTriangulatedPoints = 100
    kFeatureMatchRatioTestInitializer = 0.8   # ratio test used by Initializer

    # Tracking
    # use or not the motion model for computing a first guess pose (that will be optimized by pose optimization)
    kUseMotionModel = True
    # match frames by using frame map points projection and epipolar lines; here, the current available interframe pose estimate is used for computing the fundamental mat F
    kUseSearchFrameByProjection = True
    # if the number of tracked features is below this, then the search fails
    kMinNumMatchedFeaturesSearchFrameByProjection = 20
    # fit an essential matrix; orientation and keypoint match inliers are estimated by fitting an essential mat (5 points algorithm),
    kUseEssentialMatrixFitting = False
    # WARNING: essential matrix fitting comes with some limitations (please, read the comments of the method slam.estimate_pose_ess_mat())
    kMaxNumOfKeyframesInLocalMap = 80
    kNumBestCovisibilityKeyFrames = 10

    # Keyframe generation
    # minimum number of matched map points for spawning a new KeyFrame
    kNumMinPointsForNewKf = 15
    kThNewKfRefRatio = 0.9      # for determining if a new KF must be spawned

    # Keyframe culling
    kKeyframeCullingRedundantObsRatio = 0.9

    # Search matches by projection
    kMaxReprojectionDistanceFrame = 7  # 7   # [pixels]    o:7
    # 2.5 # [pixels]    o:1,(rgbd)3,(reloc)5 => mainly 2.5*th where th acts as a multiplicative factor
    kMaxReprojectionDistanceMap = 3
    kMaxReprojectionDistanceFuse = 3  # 3   # [pixels]    o:3
    #
    kMatchRatioTestMap = 0.8
    # used just for test function find_matches_along_line()
    kMatchRatioTestEpipolarLine = 0.8
    #
    # Reference max descriptor distance (used for initial checks and then updated and adapted)
    kMaxDescriptorDistance = 0  # it is updated by feature_manager.py at runtime

    # Search matches for triangulation by using epipolar lines
    # [pixels] Used with search by epipolar lines
    kMinDistanceFromEpipole = 10
    #
    # it is updated by feature_manager.py at runtime
    kMaxDescriptorDistanceSearchEpipolar = 0

    # Local Mapping
    # [# frames]   for generating new points and fusing them
    kLocalMappingNumNeighborKeyFrames = 20

    # Covisibility graph
    kMinNumOfCovisiblePointsForCreatingConnection = 15

    # Bundle Adjustment (BA)
    kLocalBAWindow = 20  # [# frames]
    # True: perform BA over a large window; False: do not perform large window BA
    kUseLargeWindowBA = False
    kEveryNumFramesLargeWindowBA = 10   # num of frames between two large window BA
    kLargeBAWindow = 20  # [# frames]

    # Pointcloud
    kColorPatchDelta = 1  # center +- delta

    # other parameters
    # chi-square 2 DOFs, used for reprojection error  (Hartley Zisserman pg 119)
    kChi2Mono = 5.991
