import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from pyquaternion import Quaternion
import math
# from scipy.spatial.transform import Rotation as R

# from nuscenes.devkit_dataloader.nuscenes import NuScenes
# from nuscenes.devkit_dataloader.nuscenes import NuScenesExplorer
from devkit_dataloader.nuscenes import NuScenes
from devkit_dataloader.nuscenes import NuScenesExplorer
nusc = NuScenes(version='v1.0-mini', dataroot='/home/dimitris/PhD/PhD/nuscenes/data/sets/nuscenes/v1.0-mini', verbose=True)
#nusc = NuScenes(version='v1.0-mini', dataroot='D:\Python_Projects\\PhD_project\\nuscenes\data\sets\\nuscenes\\v1.0-mini', verbose=True)
nusc2 = NuScenesExplorer(nusc)

class NuscenesDataset(Dataset):
    def __init__(self, nusc):
        self.nusc = nusc
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.permute(1, 2, 0)), # Permute the dimensions from CHW to HWC
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #to kanei skouro
        ])

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, idx): 
        """
        __getitem__ einai kati pou me tin kaleseis tn classi tha treksei. 
        Kanonika prepei get_item
        """
        my_sample = self.nusc.sample[idx]
        cam_front_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        x = cam_front_data['filename']
        # for idx in range(0,400):
        #     my_sample = self.nusc.sample[idx]
        #     cam_front_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        #     x = cam_front_data['filename']
        #     #print(x)
        #     if x == 'samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605512404.jpg':
        #         print(idx)
        #image_path = 'D:\Python_Projects\\PhD_project\\nuscenes\data\sets\\nuscenes\\v1.0-mini/' + x
        image_path = '/home/dimitris/PhD/PhD/nuscenes/data/sets/nuscenes/v1.0-mini/' + x
        image = Image.open(image_path)
        #image = self.transform(image)
        #image = image.transpose(0, 1).transpose(1, 2)
        return image, image_path
    
    def get_item_vrn(self,idx):
        my_sample = self.nusc.sample[idx]
        cam_front_data = self.nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        x = cam_front_data['filename']
        #image_path = 'D:\Python_Projects\\PhD_project\\nuscenes\data\sets\\nuscenes\\v1.0-mini/' + x
        image_path = '/home/dimitris/PhD/PhD/nuscenes/data/sets/nuscenes/v1.0-mini/' + x
        image = Image.open(image_path)
        image = self.transform(image)
        return image

    def get_points(self,idx):
        points = []
        my_sample = self.nusc.sample[idx]
        point, coloring, im, lidar_points, masked_pc = nusc2.map_pointcloud_to_image(pointsensor_token=my_sample['data']['LIDAR_TOP'],
                                                            camera_token=my_sample['data']['CAM_FRONT'])
        # print(point)
        points.append(point)  # mia lista me pinakes
        return points, lidar_points, masked_pc
    
    def get_object_pose(self, idx):
        my_sample = self.nusc.sample[idx]
        pointsensor_token=my_sample['data']['LIDAR_TOP']
        camera_token=my_sample['data']['CAM_FRONT']
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])

        object_info = []  # List to store tuples of (length, category_name)
        for my_annotation_token in my_sample['anns']:
            my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)
            my_annotation_metadata['translation']
            category_name = my_annotation_metadata['category_name']
            T3 = -np.array(poserecord['translation'])
            R3 = Quaternion(poserecord['rotation']).rotation_matrix.T
            translation_ego3 = [0, 0, 0]  # Initializing the list with zeros
            for i in range(3):
                translation_ego3[i] = my_annotation_metadata['translation'][i] + T3[i]
            translation_ego3 = np.dot(R3, translation_ego3)
            T4 = -np.array(cs_record['translation'])
            R4 = Quaternion(cs_record['rotation']).rotation_matrix.T
            translation_ego4 = [0, 0, 0]  # Initializing the list with zeros
            for i in range(3):
                translation_ego4[i] = translation_ego3[i] + T4[i]
            translation_ego4 = np.dot(R4, translation_ego4)
            transformed_translation_ego4 = [translation_ego4[0], translation_ego4[2], translation_ego4[1]]
            length = math.sqrt(transformed_translation_ego4[0]**2 + transformed_translation_ego4[1]**2 + transformed_translation_ego4[2]**2)
            object_info.append((length, category_name))  # Appending the tuple to the list
            
        return object_info  # Return all lists
    
    def get_calib(self,idx):
        """
        Lathos giati otan allazei i eikona allazei kai to calibration matrix
        """
        my_sample = self.nusc.sample[idx]
        cam_front = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
        cam_front_data = nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
        intrinsic = cam_front_data['camera_intrinsic']
        K = np.array(intrinsic)
        P = np.zeros((3,4))
        P[:,:-1] = K
        # for i in range(0,119):  
        #     a = nusc.calibrated_sensor[i]
        #     if a['token'] == cam_front_data['calibrated_sensor_token']:
        #         K = np.array(a['camera_intrinsic'])
        #         t = np.array(a['translation'])
        #         # Convert quaternion to rotation matrix
        #         quaternion = np.array(a['rotation'])
        #         r = R.from_quat(quaternion)
        #         rotation = r.as_matrix()
        #         # Concatenate rotation matrix and translation vector to form a 3x4 extrinsic matrix
        #         RT = np.hstack((rotation, t.reshape(-1, 1)))
        #         # Compute projection matrix P by multiplying K and [R|t]
        #         P = np.dot(K, RT)
        return P
            
    