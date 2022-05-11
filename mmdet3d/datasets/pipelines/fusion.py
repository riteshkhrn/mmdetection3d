from mmdet.datasets.builder import PIPELINES

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
        for ldr in self.lidars:
            if ldr in results:
                results['points'] = results['points'].cat( \
                                            [results[ldr]['points'], \
                                             results['points']])
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = f'{self.__class__.__name__}'
        repr_str += f'(lidars={self.lidars})'
        return repr_str