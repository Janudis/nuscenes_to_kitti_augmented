# nuScenes to KITTI Conversion with LiDAR Augmentation

This project converts nuScenes data to KITTI format while augmenting the LiDAR point cloud data to improve scene reconstruction.

## Features
- **KITTI Conversion**: Transforms nuScenes dataset into KITTI format.
- **LiDAR Augmentation**: Enhances LiDAR point clouds by aggregating multiple timestamps to increase density and mitigate sparsity.
- **Dynamic Object Handling**: Compensates for moving objects using transformation matrices.
- **Symmetry Constraints**: Densifies occluded regions by mirroring LiDAR points across object bounding boxes.

## Project Structure
The project follows this structure:

github/ ├── nuscenes/ │ ├── scripts/ │ │ ├── export_kitti.py # Main script for nuScenes to KITTI conversion │ │ ├── ... # Other utility scripts (not detailed here)


## LiDAR Augmentation Methodology
This project improves LiDAR-based scene understanding using the following techniques:

1. **Temporal Aggregation**  
   LiDAR point clouds captured from different timestamps are transformed into a common frame. A sequence of ego-vehicle poses is used to align multiple LiDAR sweeps, providing a denser representation of the environment.

2. **Dynamic Object Tracking**  
   Moving objects are transformed using ground truth bounding boxes. Each LiDAR ray intersecting an object is transformed according to its pose across time, preserving dynamic motions accurately.

3. **Symmetry Constraints**  
   Vehicles and other objects are often symmetric. By mirroring LiDAR rays about the vertical plane of each bounding box, we fill occluded regions and enhance object completeness in the dataset.

## Installation
To run the script, ensure you have the following dependencies installed:
- Python 3
- PyTorch
- NumPy
- Matplotlib
- PIL (Pillow)
- Fire
- pyquaternion

Usage

    Update Path
    In export_kitti.py, change:

sys.path.append('/media/harddrive/github')  # Change this to your repository path

to point to your local repository.

Run the Conversion
Convert nuScenes data into KITTI format by specifying the output directory:

python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir /path/to/output_dir

Example:

    python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir /media/harddrive/github/testing_repo

Notes

    Only the front-facing camera is used for KITTI-style images.
    Other scripts in nuscenes/scripts/ are not required and thus not documented here.
    This process assumes you already have the nuScenes dataset and relevant environment set up.

References

    nuScenes Dataset
    KITTI Dataset
