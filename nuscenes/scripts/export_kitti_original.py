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
- python export_kitti.py render_kitti --nusc_kitti_dir ~/nusc_kitti --render_2d False
- python export_kitti.py kitti_res_to_nuscenes --nusc_kitti_dir ~/nusc_kitti
python export_kitti_original.py nuscenes_gt_to_kitti --nusc_kitti_dir /media/dimitris/4b3f6643-e758-40b9-9b58-9e98f88e5c791/dimitris/nusc_kitti_mini_without_augmentation
Note: The parameter --render_2d specifies whether to draw 2d or 3d boxes.

To work with the original KITTI dataset, use these parameters:
 --nusc_kitti_dir /data/sets/kitti --split training

See https://www.nuscenes.org/object-detection for more information on the nuScenes result format.
"""
import sys
sys.path.append('/home/dimitris/PhD/PhD')

import json
import os
from typing import List, Dict, Any
import pickle
import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.devkit_dataloader.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs


class KittiConverter:
    def __init__(self,
                 nusc_kitti_dir: str = '~/nusc_kitti',
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                #  image_count: int = 10,
                 nusc_version: str = 'v1.0-mini',
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

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        token_idx = 0  # Start tokens from 0.
        calib_counter = 0 #28130 #0 
        file_counter = 0 #28130 #0

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Create output folders.
        label_folder = os.path.join(self.nusc_kitti_dir, self.split, 'label_2')
        calib_folder = os.path.join(self.nusc_kitti_dir, self.split, 'calib')
        image_folder = os.path.join(self.nusc_kitti_dir, self.split, 'image_2')
        lidar_folder = os.path.join(self.nusc_kitti_dir, self.split, 'velodyne')
        lidar_folder_start_points = os.path.join(self.nusc_kitti_dir, self.split, 'velodyne_start_points')

        for folder in [label_folder, calib_folder, image_folder, lidar_folder, lidar_folder_start_points]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        # sample_tokens = sample_tokens[:self.image_count]

        tokens = []
        for sample_token in sample_tokens:

            # Get sample data.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']
            cam_front_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]

            # Retrieve sensor records.
            sd_record_cam = self.nusc.get('sample_data', cam_front_token)
            sd_record_lid = self.nusc.get('sample_data', lidar_token)
            cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

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
            filename_lid_full = sd_record_lid['filename']
            # token = '%06d' % token_idx # Alternative to use KITTI names.
            token_idx += 1

            # Convert image (jpg to png).
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            # dst_im_path = os.path.join(image_folder, sample_token + '.png')
            dst_im_path = os.path.join(image_folder, '{:06d}.png'.format(file_counter))  # use file_counter for file name
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")
            image = Image.open(src_im_path)
            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            # dst_lid_path = os.path.join(lidar_folder, sample_token + '.bin')
            dst_lid_path = os.path.join(lidar_folder, '{:06d}.bin'.format(file_counter))  # use file_counter for file name
            assert not dst_lid_path.endswith('.pcd.bin')

            dst_lid_path_start_points = os.path.join(lidar_folder_start_points, '{:06d}.bin'.format(file_counter))  # use file_counter for file name
            assert not dst_lid_path_start_points.endswith('.pcd.bin')

            pcl = LidarPointCloud.from_file(src_lid_path)
            # print(f"pcl {pcl.points.shape}")
            pcl.remove_close(1.0)
            # print(f"pcl.remove_close {pcl.points.shape}")
            start_points = np.zeros_like(pcl.points)
            # print(f"start_points {start_points.shape}")
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)
            start_points[:3, :] = np.dot(kitti_to_nu_lidar_inv.rotation_matrix, start_points[:3, :])
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

            kitti_transforms['cam_ego_pose'] = cam_ego_pose_matrix
            kitti_transforms['lidar_ego_pose'] = lidar_ego_pose_matrix
            kitti_transforms['camera_extrinsics'] = camera_extrinsics
            kitti_transforms['lidar_extrinsics'] = lidar_extrinsics

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
            # label_path = os.path.join(label_folder, sample_token + '.txt')
            label_path = os.path.join(label_folder, '{:06d}.txt'.format(file_counter))  # use file_counter for file name
            if os.path.exists(label_path):
                print('Skipping existing file: %s' % label_path)
                continue
            else:
                print('Writing file: %s' % label_path)
            simplified_boxes = []
            with open(label_path, "w") as label_file:
                # print(f"sample_annotation_tokens {len(sample_annotation_tokens)}")
                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)
                    # print(f"sample_annotation_token {sample_annotation_token}")
                    # print(f"sample_annot {sample_annotation}")
                    # Get box in LIDAR frame.
                    _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                    selected_anntokens=[sample_annotation_token])
                    # print(f"box_lidar_nusc {box_lidar_nusc}")
                    # if sample_annotation['num_lidar_pts'] > 0:
                    #     for box in box_lidar_nusc:
                    #         # print(dir(box))
                    #         orientation_quaternion = Quaternion(box.orientation)
                    #         yaw_angle_radians = orientation_quaternion.yaw_pitch_roll[0]
                    #         pitch_angle_radians = orientation_quaternion.yaw_pitch_roll[1]
                    #         roll_angle_radians = orientation_quaternion.yaw_pitch_roll[2]
                    #         v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
                    #         yaw = -np.arctan2(v[2], v[0])
                    #         pitch = np.arcsin(box.rotation_matrix[1, 0])
                    #         roll = -np.arctan2(box.rotation_matrix[1, 2], box.rotation_matrix[1, 1])
                    #         # Create a simplified data structure for the box
                    #         simplified_box = {
                    #             'xyz': box.center,
                    #             'wlh': box.wlh,
                    #             'rotation_matrix': box.rotation_matrix,
                    #             'theta': yaw_angle_radians, # Yaw angle in radians
                    #             'theta2': yaw,
                    #             'roll': roll_angle_radians,
                    #             'pitch': pitch_angle_radians,
                    #             'roll2': roll,
                    #             'pitch2': pitch
                    #         }
                    #         # Add the simplified box data to the list
                    #         simplified_boxes.append(simplified_box)

                    box_lidar_nusc = box_lidar_nusc[0]
                    # Truncated: Set all objects to 0 which means untruncated.
                    truncated = 0.0

                    # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
                    occluded = 0

                    # Convert nuScenes category to nuScenes detection challenge category.
                    detection_name = category_to_detection_name(sample_annotation['category_name'])
                    if detection_name: #i added this
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

                    # Convert box to output string format.
                    output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                                   truncation=truncated, occlusion=occluded)
                    # self.nusc.render_annotation(sample_annotation_token)
                    # Write to disk.
                    label_file.write(output + '\n')
                # print(f"simplified_box {simplified_boxes}")
                # with open('bounding_boxes_test.pkl', 'wb') as f:
                #     pickle.dump(simplified_boxes, f)
                # plt.imshow(image)
                # plt.show()
                file_counter += 1
                print(f"file_counter {file_counter}")

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
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples


if __name__ == '__main__':
    fire.Fire(KittiConverter)