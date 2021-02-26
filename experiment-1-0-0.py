#!/usr/bin/python
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

kfpose/0001.txt:
    timestamp, x, y, z, qw, qx, qy, qz
trials.txt:
    trial_count, 1(success)or0(failure), num_frames, frame_id_start, frame_id_end, duration_mean, duration_median

"""

import numpy as np
import cv2
import statistics
import math
import time
import os
import sys
import argparse

import pandas as pd

kDebug = True


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    experiment 1
    ''')
    parser.add_argument(
        'sequence_path', help='path to sequence'
    )
    # parser.add_argument(
    #     '--first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    # parser.add_argument(
    #     '--second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    # parser.add_argument(
    #     '--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    args = parser.parse_args()
    os.chdir(args.sequence_path)
    work_dir_path = os.getcwd()

    files = os.listdir(work_dir_path)
    # print(files)
    feature_names = [f for f in files if os.path.isdir(os.path.join(work_dir_path, f))]
    # print(feature_names)

    for feature_name in feature_names:

        # trials.txt:
        #    trial_count, 1(success)or0(failure), num_frames, frame_id_start, frame_id_end, duration_mean, duration_median
        # with open(file=os.path.join(work_dir_path, feature_name, 'trials.txt'), mode='r') as f:
        df = pd.read_csv(os.path.join(work_dir_path, feature_name, 'trials.txt'), index_col=0, sep=' ', names=(
            'trial_count', 'bSuccess', 'num_frames', 'frame_id_start', 'frame_id_end', 'duration_mean', 'duration_median'))
        if kDebug:
            print('feature_name:', feature_name)
            print(df, '\n')

    # lines = f.readlines()
    # for line in lines:
    #     trial_count, bSuccess, num_frames, frame_id_start, frame_id_end, duration_mean, duration_median = line.strip('\n').split(' ')


# from config import Config

# from slam import Slam, SlamState
# from camera import PinholeCamera
# from ground_truth import groundtruth_factory
# from dataset import dataset_factory

# #from mplot3d import Mplot3d
# #from mplot2d import Mplot2d
# from mplot_thread import Mplot2d, Mplot3d

# from display2D import Display2D
# from viewer3D import Viewer3D
# from utils import getchar, Printer, Logging

# from feature_tracker import feature_tracker_factory, FeatureTrackerTypes
# from feature_manager import feature_manager_factory
# from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
# from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

# from feature_tracker_configs import FeatureTrackerConfigs

# from parameters import Parameters

# from tqdm import tqdm

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# # +++
# kLogTrialToFile = True
# kdisplayViewer3D = False
# kdisplayDisplay2d = False
# kDebug = True
# kVerbose = False  #

# num_trials = 100
# if kDebug:
#     num_trials = 5  # number of trials

# result_dir_name = 'result'
# if kDebug:
#     result_dir_name = 'test_result'
#     img_id_end = 100
# # +++


# def main():
#     # *** for each sequences ***
#     config = Config()

#     current_dir_path = os.getcwd()

#     dataset = dataset_factory(config.dataset_settings)
#     num_frames = dataset.max_frame_id  # number of the sequence frames
#     if kDebug:
#         num_frames = min(num_frames, img_id_end)
#     frame_id_iniTh = int(0.10 * num_frames)
#     frame_id_lostTh = int(0.99 * num_frames)

#     # +++
#     dataset_name = dataset.type.name  # note: dataset.dataset_factory().type.name, e.g., KITTI
#     sequence_name = dataset.name  # see config.ini
#     save_dir_path = os.path.join(current_dir_path, result_dir_name, dataset_name, sequence_name)
#     # +++
#     groundtruth = groundtruth_factory(config.dataset_settings)
#     # not actually used by Slam() class; could be used for evaluating performances
#     # groundtruth = None

#     cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
#                         config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
#                         config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
#                         config.DistCoef, config.cam_settings['Camera.fps'])

#     num_features = 2000

#     # descriptor-based, brute force matching with knn
#     tracker_type = FeatureTrackerTypes.DES_BF
#     # tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching

#     # select your tracker configuration (see the file feature_tracker_configs.py)
#     # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
#     # tracker_config = FeatureTrackerConfigs.SHI_TOMASI_ORB

#     # +++ features (dict)
#     if kDebug:
#         tracker_configs = {'AKAZE': FeatureTrackerConfigs.AKAZE,
#                            'D2NET': FeatureTrackerConfigs.D2NET,
#                            'ORB2': FeatureTrackerConfigs.ORB2,
#                            'SUPERPOINT': FeatureTrackerConfigs.SUPERPOINT}
#     else:
#         tracker_configs = {'AKAZE': FeatureTrackerConfigs.AKAZE,
#                            'BRISK_TFEAT': FeatureTrackerConfigs.BRISK_TFEAT,
#                            'CONTEXTDESC': FeatureTrackerConfigs.CONTEXTDESC,
#                            'ORB2': FeatureTrackerConfigs.ORB2,
#                            'ROOT_SIFT': FeatureTrackerConfigs.ROOT_SIFT,
#                            'SUPERPOINT': FeatureTrackerConfigs.SUPERPOINT}

#     # tracker_config = FeatureTrackerConfigs.AKAZE
#     # tracker_config = FeatureTrackerConfigs.BRISK_TFEAT
#     # tracker_config = FeatureTrackerConfigs.CONTEXTDESC
#     # tracker_config = FeatureTrackerConfigs.D2NET
#     # tracker_config = FeatureTrackerConfigs.FAST_TFEAT
#     # tracker_config = FeatureTrackerConfigs.KEYNET
#     # tracker_config = FeatureTrackerConfigs.LFNET
#     # tracker_config = FeatureTrackerConfigs.ORB2
#     # tracker_config = FeatureTrackerConfigs.R2D2
#     # tracker_config = FeatureTrackerConfigs.ROOT_SIFT
#     # tracker_config = FeatureTrackerConfigs.SIFT
#     # tracker_config = FeatureTrackerConfigs.SUPERPOINT

#     # *** for each features ***
#     pbar1 = tqdm(tracker_configs.items())
#     for feature_name, tracker_config in pbar1:
#         pbar1.set_description('feature: {0}'.format(feature_name))
#         # make dirs for each features
#         work_dir_path = os.path.join(save_dir_path, feature_name)
#         os.makedirs(os.path.join(work_dir_path, 'kfpose'), exist_ok=True)
#         os.chdir(work_dir_path)

#         # if kLogTrialToFile:
#         # trial_logger = Logging.setup_file_logger(name='trial_logger', log_file='trials.txt', mode='+w', formatter=Logging.simple_log_formatter)

#         tracker_config['num_features'] = num_features
#         tracker_config['tracker_type'] = tracker_type

#         if kVerbose:
#             print('tracker_config: ', tracker_config)
#         feature_tracker = feature_tracker_factory(**tracker_config)

#         # *** for each trials ***
#         trial_count = len(os.listdir(os.path.join(work_dir_path, 'kfpose'))) + 1
#         pbar2 = tqdm(range(trial_count, trial_count + num_trials))
#         for current_trial_count in pbar2:
#             pbar2.set_description('current_trial_count: {0}'.format(current_trial_count))
#             # create SLAM object
#             slam = Slam(cam, feature_tracker, groundtruth)
#             time.sleep(1)  # to show initial messages

#             if kdisplayViewer3D is True:
#                 viewer3D = Viewer3D()
#             else:
#                 viewer3D = None

#             if kdisplayDisplay2d is True:
#                 display2d = Display2D(cam.width, cam.height)  # pygame interface
#             else:
#                 display2d = None  # enable this if you want to use opencv window

#             # matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches', title='# matches')

#             do_step = False
#             is_paused = False

#             # +++
#             bSuccess = False
#             bIniDone = False
#             frame_id_start = -1
#             frame_id_end = -1

#             duration_list = []  # list of each duration time
#             # +++

#             img_id = 0  # 180, 340, 400   # you can start from a desired frame id if needed

#             pbar3 = tqdm(range(num_frames), total=num_frames)
#             while dataset.isOk():
#                 pbar3.set_description('image_id: {0}'.format(img_id))
#                 if not is_paused:
#                     # +++ for debug
#                     if kDebug:
#                         if img_id == num_frames:
#                             break  # while dataset.isOk():
#                     # # +++ for debug
#                     if kVerbose:
#                         print('..................................')
#                         print('image: ', img_id)
#                     img = dataset.getImageColor(img_id)
#                     if img is None:
#                         if kVerbose:
#                             print('image is empty')
#                             # getchar()

#                     timestamp = dataset.getTimestamp()          # get current timestamp
#                     next_timestamp = dataset.getNextTimestamp()  # get next timestamp
#                     frame_duration = next_timestamp - timestamp

#                     if img is not None:
#                         time_start = time.time()
#                         slam.track(img, img_id, timestamp)  # main SLAM function

#                         # print('img_id:', img_id, 'slam.tracking.state.name:', slam.tracking.state.name)

#                         # requirement 1: do initialization within first 10% frames
#                         if not bIniDone:
#                             if slam.tracking.state is SlamState.OK:
#                                 bIniDone = True
#                                 bSuccess = True
#                                 frame_id_start = img_id
#                             elif img_id == frame_id_iniTh:
#                                 break  # while dataset.isOk():

#                         # requirement 2: continue tracking until last 1%
#                         if slam.tracking.state is SlamState.LOST:
#                             frame_id_end = img_id - 1
#                             if img_id < frame_id_lostTh:
#                                 bSuccess = False
#                             break  # while dataset.isOk():

#                         if img_id == num_frames - 1:  # last frame in sequence
#                             frame_id_end = img_id

#                             # 3D display (map display)
#                         if viewer3D is not None:
#                             viewer3D.draw_map(slam)

#                         if kdisplayDisplay2d is True:
#                             img_draw = slam.map.draw_feature_trails(img)

#                             # 2D display (image display)
#                             if display2d is not None:
#                                 display2d.draw(img_draw)
#                             else:
#                                 cv2.imshow('Camera', img_draw)

#                         duration = time.time() - time_start
#                         # +++
#                         duration_list.append(duration)
#                         # +++
#                         if(frame_duration > duration):
#                             if kVerbose:
#                                 print('sleeping for frame')
#                             time.sleep(frame_duration - duration)

#                     img_id += 1
#                 else:
#                     time.sleep(1)

#                 # get keys
#                 # key = matched_points_plt.get_key()
#                 key_cv = cv2.waitKey(1) & 0xFF

#                 # manage interface infos

#                 # --- if slam.tracking.state == SlamState.LOST:
#                 # ---     if display2d is not None:
#                 # ---         getchar()
#                 # --- else:
#                 # --- # useful when drawing stuff for debugging
#                 # --- key_cv = cv2.waitKey(0) & 0xFF

#                 # if key == 'q' or (key_cv == ord('q')):
#                 if key_cv == ord('q'):
#                     if display2d is not None:
#                         display2d.quit()
#                     if viewer3D is not None:
#                         viewer3D.quit()
#                     # if matched_points_plt is not None:
#                     #     matched_points_plt.quit()
#                     break

#                 if viewer3D is not None:
#                     is_paused = not viewer3D.is_paused()

#                 pbar3.update()

#             # after finish tracking
#             # +++ kfpose/0001.txt
#             slam.tracking.print_tracking_history(os.path.join(work_dir_path, 'kfpose', str(current_trial_count).zfill(4) + '.txt'))
#             # +++
#             if kLogTrialToFile:
#                 duration_mean = -1.
#                 duration_median = -1.
#                 if duration_list is not None:
#                     duration_mean = statistics.mean(duration_list)
#                     duration_median = statistics.median(duration_list)
#                     if kVerbose:
#                         print('duration_mean[s]:', duration_mean, 'duration_median[s]:', duration_median)
#                 # trial_logger.info('{0} {1}'.format(duration_mean, duration_median))
#                 # trials.txt:
#                 #   trial_count, 1(success)or0(failure), num_frames, frame_id_start, frame_id_end, duration_mean, duration_median
#                 with open(file='trials.txt', mode='a') as f:
#                     f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(str(current_trial_count).zfill(4), bSuccess, num_frames, frame_id_start, frame_id_end, duration_mean, duration_median))

#             # +++
#             # if matched_points_plt is not None:
#             #     matched_points_plt.savefigimage()
#             # +++
#             slam.quit()
#             if display2d is not None:
#                 display2d.quit()
#             if viewer3D is not None:
#                 viewer3D.quit()
#         # *** for each trials ***
#     # *** for each features ***
#     # *** for each sequences ***

#     # cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
