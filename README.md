nuScenes to KITTI Conversion with LiDAR Augmentation

This project converts nuScenes data to KITTI format while augmenting the LiDAR point cloud data to improve scene reconstruction. The conversion process supports only the front-facing camera in nuScenes, ensuring compatibility with KITTI-based object detection frameworks.

Features

KITTI Conversion: Transforms nuScenes dataset into KITTI format.

LiDAR Augmentation: Enhances LiDAR point clouds by aggregating multiple timestamps to increase density and mitigate sparsity.

Dynamic Object Handling: Compensates for moving objects using transformation matrices.

Symmetry Constraints: Densifies occluded regions by mirroring LiDAR points across object bounding boxes.

Project Structure

The project follows this structure:

github/
├── nuscenes/
│   ├── scripts/
│   │   ├── export_kitti.py  # Main script for nuScenes to KITTI conversion
│   │   ├── ...              # Other utility scripts (not detailed here)

LiDAR Augmentation Methodology

This project improves LiDAR-based scene understanding using the following techniques:

Temporal Aggregation:

LiDAR point clouds captured from different timestamps are transformed into a common frame.

A sequence of ego-vehicle poses is used to align multiple LiDAR sweeps, providing a denser representation of the environment.

Dynamic Object Tracking:

Moving objects are transformed using available ground truth bounding boxes.

Each LiDAR ray intersecting an object is transformed according to its pose at different time steps.

Symmetry Constraints:

Vehicles and other symmetric objects are augmented by reflecting LiDAR rays about their vertical planes.

This approach fills occluded regions and enhances object completeness in the dataset.

Installation

This project does not include a requirements.txt file. To run the script, ensure you have the following dependencies installed:

Python 3

PyTorch

NumPy

Matplotlib

PIL

Fire

pyquaternion

You can install them manually using:

pip install torch numpy matplotlib pillow fire pyquaternion

Usage

Update the sys.path.append() line in export_kitti.py to match your repository location:

sys.path.append('/media/harddrive/github')  # Change this to your repository path

Run the script to convert nuScenes data into KITTI format:

python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir /media/harddrive/github/testing_repo

Notes

The --nusc_kitti_dir argument specifies the output directory where KITTI-format data will be saved.

Only front-facing camera data is used in this conversion.

Other scripts in nuscenes/scripts/ are not required for this process and are not documented here.

References

nuScenes dataset: https://www.nuscenes.org/

KITTI dataset: http://www.cvlibs.net/datasets/kitti/

