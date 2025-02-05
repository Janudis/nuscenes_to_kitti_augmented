# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

"""
This script converts nuScenes data to KITTI format and KITTI results to nuScenes.
It is used for compatibility with software that uses KITTI-style annotations.

We do not encourage this, as:
- KITTI has only front-facing cameras, whereas nuScenes has a 360 degree horizontal fov.
- KITTI has no radar data.
- The nuScenes database format is more modular.
- KITTI fields like occluded and truncated cannot be exactly reproduced from nuScenes data.
- KITTI has different categories.

Limitations:
- We don't specify the KITTI imu_to_velo_kitti projection in this code base.
- We map nuScenes categories to nuScenes detection categories, rather than KITTI categories.
- Attributes are not part of KITTI and therefore set to '' in the nuScenes result format.
- Velocities are not part of KITTI and therefore set to 0 in the nuScenes result format.
- This script uses the `train` and `val` splits of nuScenes, whereas standard KITTI has `training` and `testing` splits.

This script includes three main functions:
- nuscenes_gt_to_kitti(): Converts nuScenes GT annotations to KITTI format.
- render_kitti(): Render the annotations of the (generated or real) KITTI dataset.
- kitti_res_to_nuscenes(): Converts a KITTI detection result to the nuScenes detection results format.

To launch these scripts run:
- python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/nusc_kitti    
PX) python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir /media/dimitris/4b3f6643-e758-40b9-9b58-9e98f88e5c791/dimitris/nusc_kitti_mini_with_nusc_pc
python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir /media/harddrive/github/testing_repo 
- python export_kitti.py render_kitti --nusc_kitti_dir ~/nusc_kitti --render_2d False
- python export_kitti.py kitti_res_to_nuscenes --nusc_kitti_dir ~/nusc_kitti
Note: The parameter --render_2d specifies whether to draw 2d or 3d boxes.

To work with the original KITTI dataset, use these parameters:
 --nusc_kitti_dir /data/sets/kitti --split training

See https://www.nuscenes.org/object-detection for more information on the nuScenes result format.
"""
import sys
sys.path.append('/media/harddrive/github') #change this to the path of the repo

import json
import os
from typing import List, Dict, Any
import torch
import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import pickle
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.devkit_dataloader.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs

import logging
# Set up logging to file
logging.basicConfig(level=logging.DEBUG, filename='kitti_conversion.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

class KittiConverter:
    def __init__(self,
                 nusc_kitti_dir: str = '~/nusc_kitti_full_with_egopose', #irrelevant dilonetai sto terminal
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                #  image_count: int = 10,
                 nusc_version: str = 'v1.0-mini',
                #  nusc_version: str = 'v1.0-trainval',
                #  split: str = 'train'):
                #  split: str = 'val'):
                 split: str = 'mini_train'):
        """
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        # self.image_count = image_count
        self.nusc_version = nusc_version
        self.split = split

        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=nusc_version)

    def quaternion_to_rotation_matrix(self, quaternion):
        w, x, y, z = quaternion

        # Calculate rotation matrix elements
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        # Construct rotation matrix
        rotation_matrix = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])

        return rotation_matrix
    
    def is_inside_box(self, points, sample):
        cx, cy, cz = sample['translation']
        w, l, h = sample['size']
        R = self.quaternion_to_rotation_matrix(sample['rotation'])

        rotated_points = np.dot(R.T, (points.T - np.array([cx, cy, cz]).reshape((3, 1))))
        inside = np.abs(rotated_points[0, :]) <= l/2
        inside &= np.abs(rotated_points[1, :]) <= w/2
        inside &= np.abs(rotated_points[2, :]) <= h/2
        return inside
    
    def is_inside_box_lidar_frame(self, points, bbox):
        for box in bbox:
            cx, cy, cz = box.center
            w, l, h = box.wlh
            R = box.rotation_matrix

        rotated_points = np.dot(R.T, (points.T - np.array([cx, cy, cz]).reshape((3, 1))))
        inside = np.abs(rotated_points[0, :]) <= l/2
        inside &= np.abs(rotated_points[1, :]) <= w/2
        inside &= np.abs(rotated_points[2, :]) <= h/2
        return inside
    
    def remove_close(self, radius):
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """
        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def lidar_augmentation(self, start_sample_token):
        """
        This function augments the lidar from next timestamps based on the camera times. Each dt is almost equal to 0.5 s.
        """
        sequence_data = []
        current_sample_token = start_sample_token

        # Skip the initial token by moving to the next one immediately
        first_sample = self.nusc.get('sample', current_sample_token)
        # print(f"first_sample {first_sample}")
        if first_sample['next']:
            current_sample_token = first_sample['next']
            # print(f"current_sample_token {current_sample_token}")
        else:
            # If there's no next sample, return whatever the appropriate response is (empty list, error, etc.)
            print(f"No next sample found for token {current_sample_token}. Exiting.")
            return sequence_data 

        sample = self.nusc.get('sample', start_sample_token)
        lidar_token = sample['data'][self.lidar_name]
        sd_record_lid = self.nusc.get('sample_data', lidar_token) #pointsensor
        ref_time_export = 1e-6 * sd_record_lid['timestamp']
        # print(f"sd_record_lid init {sd_record_lid}")
        # print(f"start_sample {start_sample_token}")
    
        # Now process the next 5 samples, starting from the new current_sample_token
        # for _ in range(7):  # Adjusted to N iterations to process the next N tokens
        for _ in range(6):  # Adjusted to N iterations to process the next N tokens
            annotation_list = []
            try:
                sample = self.nusc.get('sample', current_sample_token)
            except KeyError:
                print(f"Sample token {current_sample_token} does not exist. Stopping iteration.")
                break

            sample = self.nusc.get('sample', current_sample_token)
            sample_annotation_tokens = sample['anns']
            cam_front_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]
            # Retrieve sensor records.
            sd_record_cam = self.nusc.get('sample_data', cam_front_token)
            # print(f"sd_record_cam {sd_record_cam}")
            current_sd_record_lid = self.nusc.get('sample_data', lidar_token) #pointsensor
            # print(f"sd_record_lid test {sd_record_lid}")
            # cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            # cs_record_lid = self.nusc.get('calibrated_sensor', current_sd_record_lid['calibrated_sensor_token']) #cs_record
            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            # print("nuScenes filename: ", filename_cam_full)
            filename_lid_full = current_sd_record_lid['filename']
            # Convert image (jpg to png).
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            # Read the image
            image = Image.open(src_im_path)
            # plt.imshow(image)
            # plt.show()
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            pcl = LidarPointCloud.from_file(src_lid_path)
            # print(f"pcl_points b {pcl.points.shape}")
            pcl.remove_close(2.0)
            # print(f"pcl_points a {pcl.points.shape}")
            pcl_points = pcl.points.T
            time_lag = -ref_time_export + 1e-6 * current_sd_record_lid['timestamp']  # Positive difference.
            # print(f"time_lag {time_lag}")
            times_export = time_lag * np.ones((1, pcl.nbr_points()))
            # print(f"times_export {times_export}")
            # print(f"pcl_points {pcl_points.shape}")
            #vehicle pose
            lidar_ego_pose = self.nusc.get('ego_pose', current_sd_record_lid['ego_pose_token'])
            # print(f"lidar_ego_pose {lidar_ego_pose}")
            # Get the ego pose (vehicle pose)
            # cam_ego_pose = self.nusc.get('ego_pose', sd_record_cam['ego_pose_token'])
            # print(f"cam_ego_pose {cam_ego_pose}")
            # Convert quaternion to rotation matrix
            # Combine rotation matrix and translation into a single 3x4 matrix
            lidar_ego_pose_matrix = np.hstack((Quaternion(lidar_ego_pose['rotation']).rotation_matrix, np.array(lidar_ego_pose['translation']).reshape(3, 1)))
            # cam_ego_pose_matrix = np.hstack((Quaternion(cam_ego_pose['rotation']).rotation_matrix, np.array(cam_ego_pose['translation']).reshape(3, 1)))

            for sample_annotation_token in sample_annotation_tokens:
                sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)
                annotation_list.append(sample_annotation) # global frame
                # print(f"sample_annotation_token {sample_annotation_token}")
                # print(f"sample_annotation {sample_annotation}")
                # print(f"lidar_token {lidar_token}")
                # Get box in LIDAR frame.
                _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                selected_anntokens=[sample_annotation_token])
            # print(f"annotation_list {len(annotation_list)}")
            sequence_data.append({
                    'start_point': torch.zeros((pcl_points.shape[0], 4)),
                    'end_point': pcl_points,
                    'vehicle_pose': lidar_ego_pose_matrix,
                    'annotation_list': annotation_list
                })
            # Attempt to move to the next sample token
            if sample['next']:
                current_sample_token = sample['next']
            else:
                print(f"Reached the end of the scene after processing token {current_sample_token}.")
                break
        return sequence_data

    def lidar_augmentation_test(self, start_sample_token):
        """
        This function augments the lidar from next timestamps based on the sweep times. Each dt is almost equal to 0.05 s.
        """
        sequence_data = []
        # current_sample_token = start_sample_token

        # Skip the initial token by moving to the next one immediately
        # first_sample = self.nusc.get('sample', current_sample_token)
        # print(f"first_sample {first_sample}")
        # if first_sample['next']:
        #     current_sample_token = first_sample['next']
        #     # print(f"current_sample_token {current_sample_token}")
        # else:
        #     # If there's no next sample, return whatever the appropriate response is (empty list, error, etc.)
        #     print(f"No next sample found for token {current_sample_token}. Exiting.")
        #     return sequence_data

        sample = self.nusc.get('sample', start_sample_token)
        lidar_token = sample['data'][self.lidar_name]
        sd_record_lid = self.nusc.get('sample_data', lidar_token) #pointsensor
        ref_time_export = 1e-6 * sd_record_lid['timestamp']
        # print(f"sd_record_lid init {sd_record_lid}")
        # print(f"start_sample {start_sample_token}")
    
        # Skip the initial token by moving to the next one immediately
        if sd_record_lid['next']:
            current_sd_record_lid = self.nusc.get('sample_data', sd_record_lid['next'])
            current_sample_token = current_sd_record_lid['sample_token']
            # print(f"current_sample_token first {current_sample_token}")
            # print(f"current_sd_record_lid first {current_sd_record_lid}")
        else:
            # If there's no next sample, return whatever the appropriate response is (empty list, error, etc.)
            print(f"No next sample found for token {sd_record_lid}. Exiting.")
            return sequence_data
        # print(f"current_sd_record_lid {current_sd_record_lid}")
        # annotation_list = []
        # Now process the next 5 samples, starting from the new current_sample_token
        for _ in range(20):  # Adjusted to N iterations to process the next N tokens
            annotation_list = []
            # try:
            #     sample = self.nusc.get('sample', current_sample_token)
            # except KeyError:
            #     print(f"Sample token {current_sample_token} does not exist. Stopping iteration.")
            #     break

            sample = self.nusc.get('sample', current_sample_token)
            # print(f"sample {sample}")
            # print(f"current_sd_record_lid { current_sd_record_lid}")
            # sample = self.nusc.get('sample', start_sample_token)
            sample_annotation_tokens = sample['anns']
            cam_front_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]
            # Retrieve sensor records.
            sd_record_cam = self.nusc.get('sample_data', cam_front_token)
            # print(f"sd_record_cam {sd_record_cam}")
            # sd_record_lid = self.nusc.get('sample_data', lidar_token) #pointsensor
            # print(f"sd_record_lid test {sd_record_lid}")
            # cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            # cs_record_lid = self.nusc.get('calibrated_sensor', current_sd_record_lid['calibrated_sensor_token']) #cs_record
            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            # print("nuScenes filename: ", filename_cam_full)
            filename_lid_full = current_sd_record_lid['filename']
            # Convert image (jpg to png).
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            # Read the image
            image = Image.open(src_im_path)
            plt.imshow(image)
            plt.show()
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            pcl = LidarPointCloud.from_file(src_lid_path)
            pcl_points = pcl.points.T
            time_lag = ref_time_export - 1e-6 * current_sd_record_lid['timestamp']  # Positive difference.
            print(f"time_lag {time_lag}")
            times_export = time_lag * np.ones((1, pcl.nbr_points()))
            # print(f"times_export {times_export}")
            # print(f"pcl_points {pcl_points.shape}")
            #vehicle pose
            lidar_ego_pose = self.nusc.get('ego_pose', current_sd_record_lid['ego_pose_token'])
            # print(f"lidar_ego_pose {lidar_ego_pose}")
            # Get the ego pose (vehicle pose)
            # cam_ego_pose = self.nusc.get('ego_pose', sd_record_cam['ego_pose_token'])
            # print(f"cam_ego_pose {cam_ego_pose}")
            # Convert quaternion to rotation matrix
            # Combine rotation matrix and translation into a single 3x4 matrix
            lidar_ego_pose_matrix = np.hstack((Quaternion(lidar_ego_pose['rotation']).rotation_matrix, np.array(lidar_ego_pose['translation']).reshape(3, 1)))
            # cam_ego_pose_matrix = np.hstack((Quaternion(cam_ego_pose['rotation']).rotation_matrix, np.array(cam_ego_pose['translation']).reshape(3, 1)))

            for sample_annotation_token in sample_annotation_tokens:
                sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)
                annotation_list.append(sample_annotation) # global frame
                # print(f"sample_annotation_token {sample_annotation_token}")
                # print(f"sample_annotation {sample_annotation}")
                # print(f"lidar_token {lidar_token}")
                # Get box in LIDAR frame.
                _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                selected_anntokens=[sample_annotation_token])
            # print(f"annotation_list {len(annotation_list)}")
            sequence_data.append({
                    'start_point': torch.zeros((pcl_points.shape[0], 4)),
                    'end_point': pcl_points,
                    'vehicle_pose': lidar_ego_pose_matrix,
                    'annotation_list': annotation_list
                })
            # Attempt to move to the next sample token
            # if sample['next']:
            #     current_sample_token = sample['next']
            # else:
            #     print(f"Reached the end of the scene after processing token {current_sample_token}.")
            #     break
            if current_sd_record_lid['next']:
                current_sd_record_lid = self.nusc.get('sample_data', current_sd_record_lid['next'])
                current_sample_token = current_sd_record_lid['sample_token']
                # print(f"current_sample_token next {current_sample_token}")
                # print(f"current_sd_record_lid next {current_sd_record_lid}")
            else:
                break
        return sequence_data

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)
        # print(f"self.nusc {self.nusc}")
        token_idx = 0 #28130  # Start tokens from 0.
        # print(f"self.split {self.split}") #=mini_train
        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)
        # print(f"split_logs {split_logs}")
        calib_counter = 0 #28130 
        file_counter = 0 #28130 
        tokens = []
        # Create output folders.
        label_folder = os.path.join(self.nusc_kitti_dir, self.split, 'label_2')
        calib_folder = os.path.join(self.nusc_kitti_dir, self.split, 'calib')   
        image_folder = os.path.join(self.nusc_kitti_dir, self.split, 'image_2')
        lidar_folder = os.path.join(self.nusc_kitti_dir, self.split, 'velodyne')
        lidar_folder_start_points = os.path.join(self.nusc_kitti_dir, self.split, 'velodyne_start_points')
        for folder in [label_folder, calib_folder, image_folder, lidar_folder, lidar_folder_start_points]:
            if not os.path.isdir(folder):
                os.makedirs(folder)
        for split_log in split_logs: 
            print(f"split_log {split_log}") #n015-2018-10-02-10-50-40+0800
            # Use only the samples from the current split.
            sample_tokens = self._split_to_samples(split_log)
            # print(f"sample_tokens {sample_tokens}")
            # sample_tokens = sample_tokens[:self.image_count]
            print("Number of sample tokens in this split_log:", len(sample_tokens))
            # calib_counter = 0 #28130 #0 
            # file_counter = 0 #28130 #0
            # tokens = []
            # print(f"sample_tokens {sample_tokens}")
            for sample_token in sample_tokens:
                # print(f"sample_token {sample_token}")
                # sample_token = 'ca9a282c9e77460f8360f564131a8af5'
                # Get sample data.
                sample = self.nusc.get('sample', sample_token)
                # print(f"sample_token {sample_token}")
                sample_annotation_tokens = sample['anns']
                cam_front_token = sample['data'][self.cam_name]
                lidar_token = sample['data'][self.lidar_name]
                # print(f"cam_front_token {cam_front_token}")
                # Retrieve sensor records.
                sd_record_cam = self.nusc.get('sample_data', cam_front_token)
                sd_record_lid = self.nusc.get('sample_data', lidar_token)
                # print(f"sd_record_lid {sd_record_lid}")
                cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
                cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])
                # ref_time_export = 1e-6 * sd_record_lid['timestamp']
                # print(f"ref_time {ref_time_export}")
                # print(f"cs_record_cam {cs_record_cam}")
                # print(f"cs_record_lid {cs_record_lid}")
                # Combine transformations and convert to KITTI format.
                # Note: cam uses same conventions in KITTI and nuScenes.
                lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                            inverse=False)
                ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                            inverse=True)
                velo_to_cam = np.dot(ego_to_cam, lid_to_ego)
                # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
                velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

                lidar_extrinsics = lid_to_ego
                camera_extrinsics = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                            inverse=False)
                lidar_ego_pose = self.nusc.get('ego_pose', sd_record_lid['ego_pose_token'])
                # Get the ego pose (vehicle pose)
                cam_ego_pose = self.nusc.get('ego_pose', sd_record_cam['ego_pose_token'])
                # Convert quaternion to rotation matrix
                # Combine rotation matrix and translation into a single 3x4 matrix
                lidar_ego_pose_matrix = np.hstack((Quaternion(lidar_ego_pose['rotation']).rotation_matrix, np.array(lidar_ego_pose['translation']).reshape(3, 1)))
                cam_ego_pose_matrix = np.hstack((Quaternion(cam_ego_pose['rotation']).rotation_matrix, np.array(cam_ego_pose['translation']).reshape(3, 1)))

                # Currently not used.
                imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
                r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

                # Projection matrix.
                p_left_kitti = np.zeros((3, 4))
                p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

                # Create KITTI style transforms.
                velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
                velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

                # Check that the rotation has the same format as in KITTI.
                assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
                assert (velo_to_cam_trans[1:3] < 0).all()

                # Retrieve the token from the lidar.
                # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
                # not the camera.
                filename_cam_full = sd_record_cam['filename']
                # print("filename_cam_full: ", filename_cam_full)
                filename_lid_full = sd_record_lid['filename']
                # print("filename_lid_full: ", filename_lid_full)
                # token = '%06d' % token_idx # Alternative to use KITTI names.
                token_idx += 1

                # Convert image (jpg to png).
                src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
                # print(f"src_im_path {src_im_path}")
                dst_im_path = os.path.join(image_folder, '{:06d}.png'.format(file_counter))  # use file_counter for file name
                # print(f"dst_im_path {dst_im_path}")
                # # Read the image
                # image = Image.open(src_im_path)
                # Display the image
                # plt.imshow(image)
                # plt.show()
                # print(f"file_counter{file_counter}")
                #dst_im_path = os.path.join(image_folder, sample_token + '.png')
                if not os.path.exists(dst_im_path):
                    im = Image.open(src_im_path)
                    im.save(dst_im_path, "PNG")
                # Convert lidar.
                # Note that we are only using a single sweep, instead of the commonly used n sweeps.
                src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
                #dst_lid_path = os.path.join(lidar_folder, sample_token + '.bin')
                dst_lid_path = os.path.join(lidar_folder, '{:06d}.bin'.format(file_counter))  # use file_counter for file name
                assert not dst_lid_path.endswith('.pcd.bin')

                dst_lid_path_start_points = os.path.join(lidar_folder_start_points, '{:06d}.bin'.format(file_counter))  # use file_counter for file name
                assert not dst_lid_path_start_points.endswith('.pcd.bin')

                pcl = LidarPointCloud.from_file(src_lid_path)
                pcl.remove_close(1.0)
                lidar_pcl = pcl.points.T
                # print(f"pcl {lidar_pcl.shape}")
                # pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
                augmented = self.lidar_augmentation(sample_token)
                # end_points_tensor_x = torch.tensor(augmented[0]['end_point'][:, :3], dtype=torch.float32)
                # # print(f"augmented {end_points_tensor_x}")
                # all_times = np.zeros((1, 0))
                # nusc = self.nusc
                # sample_rec = sample
                # chan = 'LIDAR_TOP' 
                # ref_chan = 'LIDAR_TOP'
                # all_pc, all_times = pcl.from_file_multisweep(nusc = self.nusc, sample_rec = sample, chan = 'LIDAR_TOP', ref_chan = 'LIDAR_TOP', nsweeps=2)
                # # print(f"all_times 2 {all_pc.points}")
                # # Get reference pose and timestamp.
                # ref_sd_token = sample_rec['data'][ref_chan]
                # ref_sd_rec = nusc.get('sample_data', ref_sd_token)
                # ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
                # ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
                # ref_time = 1e-6 * ref_sd_rec['timestamp']
                # print(f"ref_time2 {ref_time}")
                # # Aggregate current and previous sweeps.
                # sample_data_token = sample_rec['data'][chan]
                # current_sd_rec = nusc.get('sample_data', sample_data_token)
                # # Add time vector which can be used as a temporal feature.
                # time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
                # times = time_lag * np.ones((1, pcl.nbr_points()))
                # all_times = np.hstack((all_times, times))
                # print(f"all_times2 {all_times}")

                # Initialize with the point cloud at time t0
                transformed_pc_end_points = torch.tensor(lidar_pcl.reshape(-1, 4)[:, :3])
                transformed_pc_start_points = torch.zeros_like(transformed_pc_end_points)

                # SKOPOS : NA TO PAME STO GLOBAL GIA VISUALIZATION
                # transformed_pc_end_points_numpy = transformed_pc_end_points.numpy()
                # transformed_pc_end_points_ego = np.dot(lidar_extrinsics[:3, :3], transformed_pc_end_points_numpy.T).T
                # transformed_pc_end_points_ego = torch.tensor(transformed_pc_end_points_ego, dtype=torch.float32)
                # for k in range(3):
                #     transformed_pc_end_points_ego[:, k] += lidar_extrinsics[:3, 3][k]
                # P = lidar_ego_pose_matrix.reshape(3, 4)
                # P = torch.tensor(P, dtype=torch.float32)  
                # transformed_pc_end_points_ego_numpy = transformed_pc_end_points_ego.numpy()
                # transformed_pc_end_points_global = np.dot(P[:3, :3], transformed_pc_end_points_ego_numpy.T).T
                # transformed_pc_end_points_global = torch.tensor(transformed_pc_end_points_global, dtype=torch.float32)
                # for j in range(3):
                #     transformed_pc_end_points_global[:, j] += P[:3, 3][j]
                # with open('pc_global.pkl', 'wb') as f:
                #     pickle.dump(transformed_pc_end_points_global.cpu().numpy(), f)

                # print(f"first transformed_pc_end_points {transformed_pc_end_points.shape}")
                # Combine start and end points into pairs for t0 and add to the list
                # print(transformed_pc_start_points.shape) #compare with original lidar_pcl
                t0_point_pairs = torch.cat((transformed_pc_start_points.unsqueeze(2), transformed_pc_end_points.unsqueeze(2)), dim=2)
                aggregated_point_pairs = [t0_point_pairs]
                
                # print(f"aggregated_point_pairs {torch.cat(aggregated_point_pairs, dim=0).shape}")
                # end_points_inside_list = []
                for i in range(len(augmented)): #operations opws sto nuscenes.py
                    # print(f"i {i}")
                    # Convert end points to PyTorch tensors and then to homogeneous coordinates (xyz1)
                    # start_points_tensor = torch.tensor(augmented[i]['start_point'][:, :3], dtype=torch.float64)
                    start_points_tensor = augmented[i]['start_point'][:, :3].clone().detach()
                    end_points_tensor = torch.tensor(augmented[i]['end_point'][:, :3], dtype=torch.float64)
                    # end_points_tensor = augmented[i]['end_point'][:, :3].clone().detach()
                    max, _ = torch.max(end_points_tensor, dim=0)
                    # print(f"Max values in each column lidar_frame: {max}")  
                    # Apply initial transformation--> from LIDAR to EGO frame sweep time
                    start_points_numpy = start_points_tensor.numpy()
                    start_points_ego = np.dot(lidar_extrinsics[:3, :3], start_points_numpy.T).T
                    start_points_ego = torch.tensor(start_points_ego, dtype=torch.float64)
                    for j in range(3):
                        start_points_ego[:, j] += lidar_extrinsics[:3, 3][j]
                    end_points_numpy = end_points_tensor.numpy()
                    end_points_ego = np.dot(lidar_extrinsics[:3, :3], end_points_numpy.T).T
                    end_points_ego = torch.tensor(end_points_ego, dtype=torch.float64)
                    for k in range(3):
                        end_points_ego[:, k] += lidar_extrinsics[:3, 3][k]
                    max0, _ = torch.max(end_points_ego, dim=0)
                    # print(f"Max values in each column ego_frame_sweep_time: {max0}")

                    # Get the inverse of vehicle_pose and convert to tensor = Pt
                    Pt = augmented[i]['vehicle_pose'].reshape(3, 4)
                    Pt = torch.tensor(Pt, dtype=torch.float64)  
                    Pt0 = lidar_ego_pose_matrix.reshape(3, 4)
                    Pt0 = torch.tensor(Pt0, dtype=torch.float64)  

                    start_points_ego_numpy = start_points_ego.numpy()
                    transformed_start_points_pt = np.dot(Pt[:3, :3], start_points_ego_numpy.T).T
                    transformed_start_points_pt = torch.tensor(transformed_start_points_pt, dtype=torch.float64)
                    for j in range(3):
                        transformed_start_points_pt[:, j] += Pt[:3, 3][j]
                    end_points_ego_numpy = end_points_ego.numpy()
                    transformed_end_points_pt = np.dot(Pt[:3, :3], end_points_ego_numpy.T).T
                    transformed_end_points_pt = torch.tensor(transformed_end_points_pt, dtype=torch.float64)
                    for k in range(3):
                        transformed_end_points_pt[:, k] += Pt[:3, 3][k]
                    # print(f"Maximum transformed_end_points_pt-global {torch.max(transformed_end_points_pt, dim=0)}")
                    # print(f"transformed_start_points_pt {transformed_start_points_pt}")
                    # print(f"Total number of points for t=t{i}: {transformed_end_points_pt.shape}")
                    # print(f"Pt[:3, :3] {Pt[:3, :3]}")
                    # print(f"Pt[:3, 3] {Pt[:3, 3]}")

                    annotations = augmented[i]['annotation_list']
                    # print(f"len(annotations) {len(annotations)}")
                    inside_points_mask = np.zeros_like(transformed_end_points_pt, dtype=bool)
                    point_history = {}  # Initialize the dictionary 
                    # print(f"Mask initialized as: {inside_points_mask.shape}")
                    # print(f"Mask initialized as: {inside_points_mask}")
                    # print(f"annotations {annotations}")
                    for sample_ann in annotations:
                        # cx, cy, cz = sample_ann['translation']
                        # w, l, h = sample_ann['size']
                        R = self.quaternion_to_rotation_matrix(sample_ann['rotation']) #rotation matrix of a box in global frame for t=1
                        # print(f"R {R}")
                        # print(f"sample_ann['translation'] {sample_ann['translation']}")
                        instance_token = sample_ann['instance_token']
                        # print(f"sample_ann {sample_ann}")
                        # print(f"Shape of transformed_end_points_pt.numpy(): {transformed_end_points_pt.numpy().shape}") 
                        # Convert PyTorch tensor to NumPy array for boolean indexing
                        inside_points = self.is_inside_box(transformed_end_points_pt.numpy(), sample_ann)
                        for point_index in np.where(inside_points)[0]:  # Iterate over indices of  points inside
                            if point_index in point_history:
                                point_history[point_index].append(sample_ann['token']) # Assuming samples have  an 'id' field
                                # print(f"Point index {point_index} has been found inside a box before")
                            else:
                                point_history[point_index] = [sample_ann['token']]  
                        # print(f"inside_points {inside_points.shape}")
                        # print(f"inside_points {inside_points}")
                        # print(f"transformed_end_points_pt[inside_points].shape {transformed_end_points_pt[inside_points].shape}")
                        # Update only points not already marked
                        # if(inside_points_mask[inside_points].any() == True):
                            # print(inside_points_mask[inside_points])
                        # print(f"inside_points {inside_points}")
                        # inside_points_mask[inside_points] = True
                        # print(f"inside_points_mask[inside_points] {inside_points_mask[inside_points]}")
                        # print(f"inside_points_mask[inside_points] {inside_points_mask[inside_points].shape}")
                        # print(f"transformed_end_points_pt[inside_points_mask] {transformed_end_points_pt[inside_points_mask].view(-1, 3).shape}")
                        # Iterate in reverse order
                        for point_index in reversed(np.where(inside_points)[0]):
                            # print(inside_points[point_index])
                            # print(point_index)
                            # print(inside_points_mask[inside_points[point_index]])
                            # print(inside_points_mask[point_index])
                            if inside_points_mask[point_index].all() == True:  # Check if already True
                                inside_points[point_index] = False

                        # print(f"len(transformed_end_points_pt[inside_points]) {len(transformed_end_points_pt[inside_points])}")
                        # print(f"transformed_end_points_pt[inside_points].shape {transformed_end_points_pt[inside_points].shape}")
                        # if(len(transformed_end_points_pt[inside_points]) > 0):
                        if(transformed_end_points_pt[inside_points].shape[0] > 0):
                            # print(len(sample_annotation_tokens))
                            instance_token_match_found = False  # Add a flag to track if a match was found
                            for sample_annotation_token in sample_annotation_tokens:
                                sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)
                                # print(f"sample_annotation {sample_annotation}")
                                # print(f"lidar_token {lidar_token}")
                                _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                selected_anntokens=[sample_annotation_token])
                                # print(f"box_lidar_nusc {box_lidar_nusc}")                        
                                # print(f"sample_annotation {sample_annotation}")
                                distance = np.linalg.norm(np.array(sample_ann['translation']) - np.array(sample_annotation['translation']))
                                if(sample_annotation['instance_token'] == instance_token): #and distance > offset
                                    # print(f"sample_ann {sample_ann['translation']}")
                                    # print(f"sample_annotation['translation] {sample_annotation['translation']}")
                                    # distance = np.linalg.norm(np.array(sample_ann['translation']) - np.array(sample_annotation['translation']))
                                    # print(f"distance {distance}")
                                    # print(f"sample_annotation['category_name'] {sample_annotation['category_name']}")
                                    # print(f"sample_annotation {sample_annotation}")
                                    # print(f"sample_annotation['instance_token] {sample_annotation['instance_token']}")
                                    Q_T_t0 = sample_annotation['translation']
                                    Q_R_t0 = self.quaternion_to_rotation_matrix(sample_annotation['rotation'])
                                    instance_token_match_found = True  # Set the flag if a match is found
                                    # box_lidar_nusc_t0 = box_lidar_nusc
                                    # for box in box_lidar_nusc:
                                    #     x, y, z = box.center

                                    # if (np.allclose(Q_T_t0, sample_ann['translation']) and np.allclose(Q_R_t0, R)):
                                    #     print("FOUND ONE BOX THAT HASNT MOVED")
                                    #     print(sample_annotation['category_name'])

                                    # if(sample_ann['category_name'] == 'vehicle.truck'):
                                    #     print(f"sample_annotation['instance_token] {sample_annotation['instance_token']}") #e91afa15647c4c4994f19aeb302c7179
                                    #     print(f"distance {distance}")
                                        # self.nusc.render_annotation(sample_ann['token']) 
                                        # self.nusc.render_annotation(sample_annotation_token)
                                        # print(f"FOUND ONE VEHICLE.TRUCK")
                                        # print(Q_T_t0) # [409.989, 1164.099, 1.623]
                                        # print(sample_ann['translation']) # [409.998, 1164.084, 1.623]                                       
                            
                            if instance_token_match_found: 
                                # print(f"transformed_end_points_pt[inside_points].shape {transformed_end_points_pt[inside_points].shape}")
                                # print(f"sample_ann['translation'] {sample_ann['translation']}")
                                # print(f"Pt0.numpy()[:3, 3] {Pt0.numpy()[:3, 3]}")
                                inside_points_mask[inside_points] = True
                                transformed_end_points_pt_numpy_inside = transformed_end_points_pt[inside_points].numpy()
                                # transformed_end_points_pt_numpy_inside = transformed_end_points_pt[new_inside_points].numpy()

                                # print(f"Maximum of end_points_tensor[inside_points] {torch.max(end_points_tensor[inside_points], dim=0)}")
                                # First, apply translation
                                transformed_end_points_ego_inside_translated = transformed_end_points_pt_numpy_inside
                                for k in range(3):
                                    transformed_end_points_ego_inside_translated[:, k] -= sample_ann['translation'][k]  # Subtracting since you're reversing the translation
                                # Then, apply rotation
                                transformed_end_points_ego_inside_rotated = np.dot(R.T, transformed_end_points_ego_inside_translated.T).T
                                # Convert back to tensor
                                transformed_end_points_ego_inside = torch.tensor(transformed_end_points_ego_inside_rotated, dtype=torch.float64)
                                # print(f"transformed_end_points_ego_inside {transformed_end_points_ego_inside.shape}")
                                maxim0, _ = torch.max(transformed_end_points_ego_inside, dim=0)
                                # print(f"Max values in each column ego_frame_inside: {maxim0}")
                            
                                transformed_end_points_ego_inside_numpy = transformed_end_points_ego_inside.numpy()
                                transformed_end_points_global_inside = np.dot(Q_R_t0, transformed_end_points_ego_inside_numpy.T).T
                                transformed_end_points_global_inside = torch.tensor(transformed_end_points_global_inside, dtype=torch.float64)
                                for k in range(3):
                                    transformed_end_points_global_inside[:, k] += Q_T_t0[k]
                                # print(f"transformed_end_points_global_inside {transformed_end_points_global_inside.shape}")
                                maxim1, _ = torch.max(transformed_end_points_global_inside, dim=0)
                                # print(f"Max values in each column global_frame_inside: {maxim1}")

                                transformed_start_points_pt_inside_numpy = transformed_start_points_pt[inside_points].numpy()
                                # First, apply translation
                                transformed_start_points_ego_inside_translated = transformed_start_points_pt_inside_numpy
                                for j in range(3):
                                    transformed_start_points_ego_inside_translated[:, j] -= Pt0.numpy()[:3, 3][j]  # Subtracting since you're reversing the translation
                                # Then, apply rotation
                                transformed_start_points_ego_inside_rotated = np.dot(Pt0[:3, :3].T, transformed_start_points_ego_inside_translated.T).T
                                # Convert back to tensor
                                transformed_start_points_ego_inside = torch.tensor(transformed_start_points_ego_inside_rotated, dtype=torch.float64)

                                transformed_end_points_global_inside_numpy = transformed_end_points_global_inside.numpy()
                                # First, apply translation
                                for k in range(3):
                                    transformed_end_points_global_inside_numpy[:, k] -= Pt0.numpy()[:3, 3][k]  # Subtracting since you're reversing the translation
                                # Then, apply rotation
                                transformed_end_points_ego_final = np.dot(Pt0[:3, :3].T, transformed_end_points_global_inside_numpy.T).T
                                # Convert back to tensor
                                transformed_end_points_ego_final = torch.tensor(transformed_end_points_ego_final, dtype=torch.float64)
                                maxim3, _ = torch.max(transformed_end_points_ego_final, dim=0)
                                # print(f"Max values in each column ego_frame sweep: {maxim3}")

                                # Transform back to lidar coordinates
                                transformed_start_points_ego_inside_numpy = transformed_start_points_ego_inside.numpy()
                                # First, apply translation
                                transformed_start_points_inside_translated = transformed_start_points_ego_inside_numpy
                                for j in range(3):
                                    transformed_start_points_inside_translated[:, j] -= lidar_extrinsics[:3, 3][j]  # Subtracting to reverse the translation
                                # Then, apply rotation
                                transformed_start_points_inside_rotated = np.dot(lidar_extrinsics[:3, :3].T, transformed_start_points_inside_translated.T).T
                                # Convert back to tensor
                                transformed_start_points_inside = torch.tensor(transformed_start_points_inside_rotated, dtype=torch.float64)

                                transformed_end_points_ego_final_numpy = transformed_end_points_ego_final.numpy()
                                # First, apply translation
                                for k in range(3):
                                    transformed_end_points_ego_final_numpy[:, k] -= lidar_extrinsics[:3, 3][k]  # Subtracting to reverse the translation
                                # Then, apply rotation
                                transformed_end_points_inside = np.dot(lidar_extrinsics[:3, :3].T, transformed_end_points_ego_final_numpy.T).T
                                # Convert back to tensor
                                transformed_end_points_inside = torch.tensor(transformed_end_points_inside, dtype=torch.float64)
                                # print(f"transformed_end_points_lidar_inside {transformed_end_points_inside.shape}")
                                max3, _ = torch.max(transformed_end_points_inside, dim=0)
                                # print(f"Max values in each column lidar_frame transformed: {max3}")
                                # Combine start and end points (without intensity) into pairs and add to the list
                                point_pairs_inside = torch.cat((transformed_start_points_inside.unsqueeze(2), transformed_end_points_inside.unsqueeze(2)), dim=2)
                                aggregated_point_pairs.append(point_pairs_inside)
                                # end_points_inside_list.append(transformed_end_points_inside.unsqueeze(2))
                                # print(f"aggregated_point_pairs {torch.cat(aggregated_point_pairs, dim=0).shape}")

                    # Remove points inside the box from transformed_end_points_pt_modified
                    filtered_end_points_pt = transformed_end_points_pt[~inside_points_mask].view(-1, 3)
                    filtered_start_points_pt = transformed_start_points_pt[~inside_points_mask].view(-1, 3)
                    # print(f"Total number of points outside the boxes {filtered_end_points_pt.shape}")
                    # print(f"filtered_start_points_pt.shape {filtered_start_points_pt.shape}")
                    # print(f"Total number for points inside the boxes {transformed_end_points_pt[inside_points_mask].view(-1, 3).shape}")

                    transformed_start_points_pt_numpy = filtered_start_points_pt.numpy()
                    # First, apply translation
                    transformed_start_points_ego_translated = transformed_start_points_pt_numpy
                    for j in range(3):
                        transformed_start_points_ego_translated[:, j] -= Pt0.numpy()[:3, 3][j]  # Subtracting since you're reversing the translation
                    # Then, apply rotation
                    transformed_start_points_ego_rotated = np.dot(Pt0[:3, :3].T, transformed_start_points_ego_translated.T).T
                    # Convert back to tensor
                    transformed_start_points_ego = torch.tensor(transformed_start_points_ego_rotated, dtype=torch.float64)

                    transformed_end_points_pt_numpy = filtered_end_points_pt.numpy()
                    # First, apply translation
                    transformed_end_points_ego_translated = transformed_end_points_pt_numpy
                    for k in range(3):
                        transformed_end_points_ego_translated[:, k] -= Pt0.numpy()[:3, 3][k]  # Subtracting since you're reversing the translation
                    # Then, apply rotation
                    transformed_end_points_ego_rotated = np.dot(Pt0[:3, :3].T, transformed_end_points_ego_translated.T).T
                    # Convert back to tensor
                    transformed_end_points_ego = torch.tensor(transformed_end_points_ego_rotated, dtype=torch.float64)
                    max2, _ = torch.max(transformed_end_points_ego, dim=0)
                    # print(f"Max values in each column ego_frame_sweep_time transformed: {max2}")

                    # Transform back to lidar coordinates
                    transformed_start_points_ego_numpy = transformed_start_points_ego.numpy()
                    # First, apply translation
                    transformed_start_points_translated = transformed_start_points_ego_numpy
                    for j in range(3):
                        transformed_start_points_translated[:, j] -= lidar_extrinsics[:3, 3][j]  # Subtracting to reverse the translation
                    # Then, apply rotation
                    transformed_start_points_rotated = np.dot(lidar_extrinsics[:3, :3].T, transformed_start_points_translated.T).T
                    # Convert back to tensor
                    transformed_start_points = torch.tensor(transformed_start_points_rotated, dtype=torch.float64)

                    transformed_end_points_ego_numpy = transformed_end_points_ego.numpy()
                    # First, apply translation
                    transformed_end_points_translated = transformed_end_points_ego_numpy
                    for k in range(3):
                        transformed_end_points_translated[:, k] -= lidar_extrinsics[:3, 3][k]  # Subtracting to reverse the translation
                    # Then, apply rotation
                    transformed_end_points_rotated = np.dot(lidar_extrinsics[:3, :3].T, transformed_end_points_translated.T).T
                    # Convert back to tensor
                    transformed_end_points = torch.tensor(transformed_end_points_rotated, dtype=torch.float64)
                    max3, _ = torch.max(transformed_end_points, dim=0)
                    # print(f"Max values in each column lidar_frame transformed 2: {max3}")
                    all_pc, all_times = pcl.from_file_multisweep(nusc = self.nusc, sample_rec = sample, chan = 'LIDAR_TOP', ref_chan = 'LIDAR_TOP', nsweeps=2)
                    # print(f"transformed_end_points {transformed_end_points}")
                    max_values_axis0, _ = torch.max(transformed_end_points, dim=0)
                    # print(f"max_values_axis0 {max_values_axis0}")

                    # Combine start and end points (without intensity) into pairs and add to the list
                    point_pairs_filtered = torch.cat((transformed_start_points.unsqueeze(2), transformed_end_points.unsqueeze(2)), dim=2)
                    aggregated_point_pairs.append(point_pairs_filtered)
                    # print(f"aggregated_point_pairs {torch.cat(aggregated_point_pairs, dim=0).shape}")               

                # # For symmetry constraints augmentation
                # aggregated_point_pairs_symmetry = aggregated_point_pairs
                # aggregated_point_pairs_symmetry = torch.cat(aggregated_point_pairs_symmetry, dim=0) #to kanei torch ousiastika
                # end_points_symmetry = aggregated_point_pairs_symmetry[:, :, 1]
                # # symmetry_constraints_start_points = end_points_symmetry
                # symmetry_constraints_start_points = aggregated_point_pairs_symmetry[:, :, 0]
                # for sample_annotation_token in sample_annotation_tokens:
                #     sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)
                #     # Get box in LIDAR frame.
                #     _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                #                                                     selected_anntokens=[sample_annotation_token])
                #     for bbox in box_lidar_nusc:
                #         # print(f"bbox {bbox}")
                #         center = bbox.center
                #         R = bbox.rotation_matrix
                #     # inside_symmetry = self.is_inside_box_lidar_frame(symmetry_constraints_start_points, box_lidar_nusc)
                #     inside_symmetry = self.is_inside_box_lidar_frame(end_points_symmetry, box_lidar_nusc)
                #     if (len(symmetry_constraints_start_points[inside_symmetry, :]) > 0 and sample_annotation['category_name'].startswith('vehicle')):
                #         # print(f"sample_annotation['category_name'] {sample_annotation['category_name']}")
                #         # self.nusc.render_annotation(sample_annotation_token)
                #         # print(f"symmetry_constraints_start_points[inside_symmetry, :] {symmetry_constraints_start_points[inside_symmetry, :].shape}")
                #         # Subtract the bounding box's translation
                #         # translated_points = symmetry_constraints_start_points[inside_symmetry, :] - center
                #         translated_points = end_points_symmetry[inside_symmetry, :] - center
                #         # Apply the inverse rotation to align points with the bounding box's local axes
                #         R_inv = np.linalg.inv(R)
                #         local_points = np.dot(R_inv, translated_points.T).T
                #         local_points[:, 1] = -local_points[:, 1]
                #         # local_points[:, 0] = -local_points[:, 0]
                #         # local_points[:, 2] = -local_points[:, 2]
                #         # Apply the rotation
                #         mirror_points = np.dot(R, local_points.T).T
                #         # Add the bounding box's translation
                #         mirror_points += center
                #         # print(f"mirror_points {mirror_points.shape}")
                #         point_pairs_symmetry = torch.cat((symmetry_constraints_start_points[inside_symmetry, :].unsqueeze(2), torch.tensor(mirror_points).unsqueeze(2)), dim=2)
                #         # print(point_pairs_symmetry)
                #         aggregated_point_pairs.append(point_pairs_symmetry)
                #         # print(f"aggregated_point_pairs after mirror {torch.cat(aggregated_point_pairs, dim=0).shape}") 
                #         # print(f"end_points_symmetry[inside_symmetry, :] {end_points_symmetry[inside_symmetry, :].shape}")
                #         # print(f"mirror_points {mirror_points.shape}")
                #         # with open('start_points_mirroring.pkl', 'wb') as f:
                #         #     pickle.dump(end_points_symmetry[inside_symmetry, :].cpu().numpy(), f)
                #         # with open('mirror_points.pkl', 'wb') as f:
                #         #     pickle.dump(mirror_points, f)

                # Combine all aggregated point pairs into a single tensor
                aggregated_point_pairs = torch.cat(aggregated_point_pairs, dim=0)
                # print(f"aggregated_point_pairs {aggregated_point_pairs.shape}")
                # Extract the end points (assuming they are the second element in each pair)
                end_points = aggregated_point_pairs[:, :, 1]  # This will have shape [N, 3]
                start_points = aggregated_point_pairs[:, :, 0]  # This will have shape [N, 3]
                # print(f"end_points {end_points.shape}")
                # print(f"start_points {start_points[111777]}")
                # print(f"end_points {end_points[111777]}")
                # end_points_inside_list = torch.cat(end_points_inside_list, dim=0)
                # end_points_inside_list = end_points_inside_list[:, :, 0]
                # print(f"Final shape of end_points_inside_list: {end_points_inside_list.shape}")
                # print(type(pcl.points.T))
                # with open('end_points_inside_list.pkl', 'wb') as f:
                #     pickle.dump(end_points_inside_list.cpu().numpy(), f)
                # with open('end_points_extracted.pkl', 'wb') as f:
                #     pickle.dump(end_points.cpu().numpy(), f)
                # with open('end_points_extracted_only_rotation.pkl', 'wb') as f:
                #     pickle.dump(end_points.cpu().numpy(), f)
                # print(f"end_points {end_points.shape}")
                end_points = np.dot(kitti_to_nu_lidar_inv.rotation_matrix, end_points.T)
                start_points = np.dot(kitti_to_nu_lidar_inv.rotation_matrix, start_points.T)
                # Convert to float32 before writing
                end_points = end_points.astype(np.float64)
                start_points = start_points.astype(np.float64)
                # print(f"end_points {end_points}")
                # print(f"start_points {start_points}")
                if np.any(np.abs(end_points) > 1e10):
                    print("Warning: Some end points exceed the allowed range.")
                if np.any(np.abs(end_points) < -1e10):
                    print("Warning: Some end points exceed the allowed range.")
                if np.any(np.abs(start_points) > 1e10):
                    print("Warning: Some end points exceed the allowed range.")
                if np.any(np.abs(start_points) < -1e10):
                    print("Warning: Some end points exceed the allowed range.")
                # Check for any [0, 0, 0] points
                is_zero_point = (start_points.T == np.array([0.0, 0.0, 0.0])).all(axis=-1)
                # Check if there's at least one [0, 0, 0] point
                has_zero_point = np.any(is_zero_point)
                if(has_zero_point == False):
                    print(f"Is there a [0, 0, 0] point? {has_zero_point}")
                with open(dst_lid_path, "w") as lid_file:
                    end_points.T.tofile(lid_file)
                with open(dst_lid_path_start_points, "w") as lid_file_start_points:
                    start_points.T.tofile(lid_file_start_points)

                # Add to tokens.
                tokens.append(sample_token)
                # Create calibration file.
                kitti_transforms = dict()
                kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
                kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
                kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
                kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
                kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
                # # Add ego_pose to the kitti_transforms dictionary
                # ego_pose_str = ' '.join(['%.12e' % num for num in ego_pose_matrix.flatten()])
                # kitti_transforms['ego_pose'] = ego_pose_str
                kitti_transforms['cam_ego_pose'] = cam_ego_pose_matrix
                kitti_transforms['lidar_ego_pose'] = lidar_ego_pose_matrix
                kitti_transforms['camera_extrinsics'] = camera_extrinsics
                kitti_transforms['lidar_extrinsics'] = lidar_extrinsics

                # calib_path = os.path.join(calib_folder, sample_token + '.txt')
                calib_path = os.path.join(calib_folder, '{:06d}.txt'.format(calib_counter))  # use calib_counter for file name
                with open(calib_path, "w") as calib_file:
                    for (key, val) in kitti_transforms.items():
                        val = val.flatten()
                        val_str = '%.12e' % val[0]
                        for v in val[1:]:
                            val_str += ' %.12e' % v
                        calib_file.write('%s: %s\n' % (key, val_str))                      
                calib_counter += 1  # increment the counter after each loop iteration
                
                # Write label file.
                #label_path = os.path.join(label_folder, sample_token + '.txt')
                label_path = os.path.join(label_folder, '{:06d}.txt'.format(file_counter))  # use file_counter for file name
                if os.path.exists(label_path):
                    print('Skipping existing file: %s' % label_path)
                    continue
                else:
                    print('Writing file: %s' % label_path)

                simplified_boxes = []
                sample_annotation_list = []
                count_annotations_with_lidar_pts = 0
                with open(label_path, "w") as label_file:
                    # print(f"sample_annotation_tokens {len(sample_annotation_tokens)}")
                    for sample_annotation_token in sample_annotation_tokens:
                        sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)
                        sample_annotation_list.append(sample_annotation) # global frame
                        # print(f"sample_annot {sample_annotation}")
                        # Get box in LIDAR frame.
                        _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                        selected_anntokens=[sample_annotation_token])
                        # print(f"box_lidar_nusc {box_lidar_nusc}")
                        if sample_annotation['num_lidar_pts'] > 0:
                            count_annotations_with_lidar_pts += 1
                            for box in box_lidar_nusc:
                                # print(f"box.rotation_matrix {box.rotation_matrix}")
                                # Compute yaw angle from quaternion
                                orientation_quaternion = Quaternion(box.orientation)
                                yaw_angle_radians = orientation_quaternion.yaw_pitch_roll[0]
                                # Create a simplified data structure for the box
                                simplified_box = {
                                    'xyz': box.center,
                                    'wlh': box.wlh,
                                    'theta': yaw_angle_radians,
                                    'rotation_matrix': box.rotation_matrix,
                                    'lidar_pts': sample_annotation['num_lidar_pts'],
                                    'translation': sample_annotation['translation'],
                                    'category_name': sample_annotation['category_name'],
                                    'sample_annotation_rotation': sample_annotation['rotation']
                                }
                                # Add the simplified box data to the list
                                simplified_boxes.append(simplified_box)
                        # print(f"simplified_boxes {len(simplified_boxes)}")
                        # print(f"box {simplified_boxes}")

                        box_lidar_nusc = box_lidar_nusc[0]
                        # Truncated: Set all objects to 0 which means untruncated.
                        truncated = 0.0
                        # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
                        occluded = 0
                        # Convert nuScenes category to nuScenes detection challenge category.
                        # detection_name = category_to_detection_name(sample_annotation['category_name'])
                        detection_name = category_to_detection_name(sample_annotation['category_name'])
                        if detection_name:
                            detection_name = detection_name.capitalize()
                        # Skip categories that are not part of the nuScenes detection challenge.
                        if detection_name is None:
                            continue

                        # Convert from nuScenes to KITTI box format.
                        box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                            box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                        # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                        bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
                        if bbox_2d is None:
                            continue

                        # Set dummy score so we can use this file as result.
                        box_cam_kitti.score = 0

                        # Calculate the center of the 2D bounding box
                        box_center_x = (bbox_2d[0] + bbox_2d[2]) / 2.0
                        # Calculate the actual alpha value
                        alpha = KittiDB.calculate_alpha(box_center_x, p_left_kitti, box_cam_kitti.orientation.yaw_pitch_roll[0])
                        output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                                    truncation=truncated, occlusion=occluded, alpha=alpha)
                        # # Convert box to output string format.
                        # output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                        #                                truncation=truncated, occlusion=occluded)

                        # Write to disk.
                        label_file.write(output + '\n')
                        # print(f"sample_annotation_token {sample_annotation_token}")
                        # self.nusc.render_annotation(sample_annotation_token)
                    # print(f"simplified_boxes {simplified_boxes}")
                    # print(f"Number of annotations with num_lidar_pts > 0: {count_annotations_with_lidar_pts}")
                    
                    # with open('bounding_boxes.pkl', 'wb') as f:
                    #     pickle.dump(simplified_boxes, f)
                    # with open('image.pkl', 'wb') as f:
                    #     pickle.dump(image, f)
                    # with open('sample_annotation_list.pkl', 'wb') as f:
                    #     pickle.dump(sample_annotation_list, f)
                    file_counter += 1
                    print(f"file_counter {file_counter}")
                    # plt.imshow(image)
                    # plt.show()

    def render_kitti(self, render_2d: bool) -> None:
        """
        Renders the annotations in the KITTI dataset from a lidar and a camera view.
        :param render_2d: Whether to render 2d boxes (only works for camera data).
        """
        if render_2d:
            print('Rendering 2d boxes from KITTI format')
        else:
            print('Rendering 3d boxes projected from 3d KITTI format')

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=(self.split,))

        # Create output folder.
        render_dir = os.path.join(self.nusc_kitti_dir, 'render')
        if not os.path.isdir(render_dir):
            os.mkdir(render_dir)

        # Render each image.
        for token in kitti.tokens[:self.image_count]:
            for sensor in ['lidar', 'camera']:
                out_path = os.path.join(render_dir, '%s_%s.png' % (token, sensor))
                print('Rendering file to disk: %s' % out_path)
                kitti.render_sample_data(token, sensor_modality=sensor, out_path=out_path, render_2d=render_2d)
                plt.close()  # Close the windows to avoid a warning of too many open windows.

    def kitti_res_to_nuscenes(self, meta: Dict[str, bool] = None) -> None:
        """
        Converts a KITTI detection result to the nuScenes detection results format.
        :param meta: Meta data describing the method used to generate the result. See nuscenes.org/object-detection.
        """
        # Dummy meta data, please adjust accordingly.
        if meta is None:
            meta = {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False,
            }

        # Init.
        results = {}

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=(self.split, ))

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        for sample_token in sample_tokens:
            # Get the KITTI boxes we just generated in LIDAR frame.
            kitti_token = '%s_%s' % (self.split, sample_token)
            boxes = kitti.get_boxes(token=kitti_token)

            # Convert KITTI boxes to nuScenes detection challenge result format.
            sample_results = [self._box_to_sample_result(sample_token, box) for box in boxes]

            # Store all results for this image.
            results[sample_token] = sample_results

        # Store submission file to disk.
        submission = {
            'meta': meta,
            'results': results
        }
        submission_path = os.path.join(self.nusc_kitti_dir, 'submission.json')
        print('Writing submission to: %s' % submission_path)
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)

    def _box_to_sample_result(self, sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
        # Prepare data
        translation = box.center
        size = box.wlh
        rotation = box.orientation.q
        velocity = box.velocity
        detection_name = box.name
        detection_score = box.score

        # Create result dict
        sample_result = dict()
        sample_result['sample_token'] = sample_token
        sample_result['translation'] = translation.tolist()
        sample_result['size'] = size.tolist()
        sample_result['rotation'] = rotation.tolist()
        sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
        sample_result['detection_name'] = detection_name
        sample_result['detection_score'] = detection_score
        sample_result['attribute_name'] = attribute_name

        return sample_result

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        # print(f"self.nusc.sample {len(self.nusc.sample)}")
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples


if __name__ == '__main__':
    fire.Fire(KittiConverter)
