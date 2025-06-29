import argparse
import open3d as o3d
import numpy as np
import json
import os

def create_box_lineset(center, size, color=[1, 0, 0]):
    cx, cy, cz = center
    dx, dy, dz = size[0] / 2, size[1] / 2, size[2] / 2

    corners = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ])

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical lines
    ]

    colors = [color for _ in lines]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def main(args):
    with open("checkpoints/results/{}.json".format(args.id), "r") as f:
        data = json.load(f)
        scene_id = data['scene_id']
        pred = data['pred']
        gt = data['gt']
        boxes = [
            create_box_lineset(center=pred[:3], size=pred[3:6], color=[1, 0, 0]),
            create_box_lineset(center=gt[:3], size=gt[3:6], color=[0, 1, 0])
        ]
        mesh_vertices = np.load(os.path.join('data/scannet/pointgroup_data', scene_id) + "_aligned_vert.npy")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh_vertices[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(mesh_vertices[:, 3:6]/255)

        o3d.visualization.draw_geometries([pcd] + boxes)
        print('.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualization")
    parser.add_argument("--id", type=str, default='7851', help="id, please make sure have it in checkpoints/results")

    args = parser.parse_args()
    main(args)