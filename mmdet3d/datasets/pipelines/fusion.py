from mmdet.datasets.builder import PIPELINES
# import open3d
# import numpy as np
# import time

@PIPELINES.register_module()
class CombineLidarsIntoSingleLidar(object):
    """Combine points from lidars into single lidar.

    Load points from side lidars into top lidar.

    Args:
        lidars ([]): List of side lidars to include.
    """
    def __init__(self, lidars):
        self.lidars = lidars
    def __call__(self, results):
        merge = False
        points_list = []
        for ldr in self.lidars:
            if ldr in results:
                merge = True
                points_list.append(results[ldr]['points'])

        if merge:
            # nbr_top = len(results['points'])
            # nbr_fr = len(results['LIDAR_FRONT_RIGHT']['points'])
            # nbr_fl = len(results['LIDAR_FRONT_LEFT']['points'])
            # append the TOP points
            points_list.append(results['points'])
            # merge
            results['points'] = results['points'].cat(points_list)
            # # save the lidars to PCD for Visualisation
            # i = np.concatenate([np.tile([1, 0, 0], (nbr_fr, 1)),
            #                     np.tile([0, 0, 1], (nbr_fl, 1)),
            #                     np.tile([0, 1, 0], (nbr_top, 1))], axis=0)
            # assert len(results['points']) == i.shape[0]
            # pcd = open3d.geometry.PointCloud()
            # pcd.points = open3d.utility.Vector3dVector(results['points'].coord.cpu().numpy())
            # pcd.colors = open3d.utility.Vector3dVector(i)
            # filename = results['pts_filename'].split('/')[-1][:-4]
            # open3d.io.write_point_cloud(f"fused_pcds/{results['sample_idx']}_{filename}.pcd", pcd)
            # time.sleep(3)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = f'{self.__class__.__name__}'
        repr_str += f'(lidars={self.lidars})'
        return repr_str