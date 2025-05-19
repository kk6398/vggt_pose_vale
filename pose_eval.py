import argparse
import io
import json
from copy import deepcopy
from pathlib import Path

import evo
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np
import torch
# from decord import VideoReader
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
from evo.tools.plot import PlotMode
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms as TF
from collections import defaultdict
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.visual_track import visualize_tracks_on_images
from natsort import natsorted
import glob
import os
import open3d as o3d




def eval_trajectory(poses_est, poses_gt, frame_ids, align=False):

    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_gt[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_est[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)

    ate_result = main_ape.ape(         # 计算​​平移部分​​的均方根误差（RMSE），用于评估整体轨迹的平移精度
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align)
    ate = ate_result.stats["rmse"]

    are_result = main_ape.ape(        # 计算​​旋转角度​​的RMSE（单位为度），评估旋转方向的偏差
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align)
    are = are_result.stats["rmse"]

    # RPE rotation and translation
    rpe_rots_result = main_rpe.rpe(         # 相邻帧间旋转角度的误差
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        delta=1,                           # delta=1表示计算​​每帧之间的相对误差
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True)                   # all_pairs=True启用全配对模式
    rpe_rot = rpe_rots_result.stats["rmse"]

    rpe_transs_result = main_rpe.rpe(           # 相邻帧间平移向量的误差
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True)
    rpe_trans = rpe_transs_result.stats["rmse"]

    plot_mode = PlotMode.xz
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE: {round(ate, 3)}, ARE: {round(are, 3)}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est,
        ate_result.np_arrays["error_array"],
        plot_mode,
        min_map=ate_result.stats["min"],
        max_map=ate_result.stats["max"],
    )
    ax.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=90)
    buffer.seek(0)

    pillow_image = Image.open(buffer)
    pillow_image.load()
    buffer.close()
    plt.close(fig)

    return {
        "ate": ate,
        "are": are,
        "rpe_rot": rpe_rot,
        "rpe_trans": rpe_trans
    }, pillow_image



def load_poses(data_location, interval=10, num=20):

    posefiles = natsorted(glob.glob(os.path.join(data_location, "pose/*.txt")))
    posefiles = sorted(posefiles, key=lambda x: int(os.path.basename(x).split('.')[0]))

    selected_posefiles = posefiles[::interval][:num]    
    poses = []
    for posefile in selected_posefiles:
        _pose = torch.from_numpy(np.loadtxt(posefile).astype("float32"))
        poses.append(_pose)
    
    # 合并为张量
    poses = torch.stack(poses, dim=0)
    print(f"Loaded {len(poses)} poses with interval={interval}, num={num}")
    return poses



def get_poses(data_location, relative=True, interval=10, num=10):    # relative=False

    c2ws = load_poses(data_location, interval=interval, num=num)
    c2ws = c2ws.numpy()  # remove batch dimension and convert to numpy
    inf_ids = np.where(np.isinf(c2ws).any(axis=(1, 2)))[0]      # 检测数组中是否存在正/负无穷大
    if inf_ids.size > 0:
        c2ws = c2ws[:inf_ids.min()]             # 截断无效数据
    # 计算第一个位姿的逆矩阵
    # 将逆矩阵与所有位姿相乘，将位姿转换到以第一个相机为原点的坐标系中。          这常用于多视图几何的统一坐标系对齐
    print("c2ws_before:=====", c2ws)
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws
    print("c2ws:=====", c2ws)
    
    return c2ws


def to_homogeneous(extrinsics):
    n = extrinsics.shape[0]
    homogeneous_extrinsics = np.eye(4)[None, :, :].repeat(n, axis=0)  # Create identity matrices
    homogeneous_extrinsics[:, :3, :4] = extrinsics  # Copy [R | t]
    return homogeneous_extrinsics


def select_images_with_interval(directory, interval=10, num=20):
    directory = os.path.join(directory, "color")  # 选择color文件夹
    files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0])  # 提取文件名数字部分排序[1](@ref)
    )

    selected_files = files[::interval][:num]  

    return [os.path.join(directory, f) for f in selected_files]

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default="/data2/hkk/slam/HI-SLAM2/data/ScanNet/scannet/scene0011_00/frames")
    # parser.add_argument('--split_file', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, default="/data2/hkk/git/vggt/examples/output_any/scannet/scene0011_00_0518")
    # parser.add_argument('--stride', type=int, default=3)
    # parser.add_argument('--plot', action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    print("\nRunning with config...")
    torch.manual_seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


    interval = 1
    num = 10
    
    ### get GT pose ###
    c2ws = get_poses(args.data_dir, relative=True, interval=interval, num=num)                         # cam2world    
        
    image_names = select_images_with_interval(args.data_dir, interval=interval, num=num)
    images = load_and_preprocess_images(image_names).to(device)
        
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)



    #### caculate VGGT pose ####
    all_cam_to_world_mat = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)  # images: torch.Size([20, 3, 392, 518])
            
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.detach().squeeze(0), 
                                                                    extrinsic.detach().squeeze(0), 
                                                                    intrinsic.detach().squeeze(0))
    print("point_map_by_unprojection:", point_map_by_unprojection.shape)      # point_cloud: (5, 392, 518, 3)
    
    
    ##### save the point cloud #####    
    rgb_image = images.squeeze(0).permute(0, 2, 3, 1)  
    rgb_data = rgb_image.detach().cpu().numpy().reshape(-1, 3)
    points = point_map_by_unprojection.reshape(-1, 3)  # 转换为 [N, 3] 格式的顶点数组
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_data)
    o3d.io.write_point_cloud(os.path.join(args.output_path, "output_interval{}_num{}.ply".format(interval, num)), pcd)
    
    
    extrinsic = extrinsic.cpu().squeeze(0).detach().numpy()  # remove batch dimension and convert to numpy
    extrinsic = to_homogeneous(extrinsic)               # cam2world
    
    all_cam_to_world_mat.extend(extrinsic) 
    

    # last_c2w = all_cam_to_world_mat[-1]

    # for pr_pose in extrinsic[1:]:  # the intervals intersect at the first frame  
    #     new_c2w = last_c2w @ pr_pose  # 动态更新后的新位姿     # 0@1  0@1@2   0@1@2@3
    #     all_cam_to_world_mat.append(new_c2w)                 # append(0@1)   append(0@1@2)    append(0@1@2@3)
    #     last_c2w = new_c2w  # 更新last_c2w为当前结果          # 0@1    0@1@2    0@1@2@3
        
        
    ######### evalate pose  error #########            
    traj_est_poses = np.array(all_cam_to_world_mat)
    n = traj_est_poses.shape[0]      
   
    w2cs = np.linalg.inv(c2ws)


    timestamps = list(range(n))
    stats, traj_plot = eval_trajectory(traj_est_poses, w2cs, timestamps, align=False)
    stats_aligned, traj_plot_align = eval_trajectory(traj_est_poses, w2cs, timestamps, align=True)
    
    
    all_metrics = deepcopy(stats)
    for metric_name, metric_value in stats_aligned.items():
        all_metrics[f"aligned_{metric_name}"] = metric_value
    print("all_metrics:=========", all_metrics)
    
    
        
    with open(args.output_path / "metrics{}_num{}.json".format(interval, num), "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    traj_plot.save(args.output_path / "plot_interval{}_num{}.png".format(interval, num))
    traj_plot_align.save(args.output_path / "plot_interval{}_num{}_align.png".format(interval, num))
