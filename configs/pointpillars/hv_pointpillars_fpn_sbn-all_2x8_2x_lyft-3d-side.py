_base_ = [
    '../_base_/models/hv_pointpillars_fpn_lyft.py',
    '../_base_/datasets/lyft-3d.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-80, -80, -5, 80, 80, 3]
# For Lyft we usually do 9-class detection
class_names = [
    'car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle', 'motorcycle',
    'bicycle', 'pedestrian', 'animal'
]
dataset_type = 'LyftDataset'
data_root = 'data/lyft/'
# Input modality for Lyft dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/lyft/': 's3://lyft/lyft/',
#         'data/lyft/': 's3://lyft/lyft/'
#    }))
# change of pipeline w.r.t to loading of side lidars
side_lidars = ['LIDAR_FRONT_RIGHT', 'LIDAR_FRONT_LEFT']
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromSideLidars',
        lidars=side_lidars,
        transforms=[
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='GlobalAlignment', rotation_axis=2),
        ]),
    # dict(type='CombineLidarsIntoSingleLidar', lidars=side_lidars),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names, side_lidars=side_lidars),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', f'{side_lidars[0]}_points'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromSideLidars',
        lidars=side_lidars,
        transforms=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=file_client_args),
            dict(type='GlobalAlignment', rotation_axis=2),
        ]),
    # dict(type='CombineLidarsIntoSingleLidar', lidars=side_lidars),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                side_lidars=side_lidars,
                with_label=False),
            dict(type='Collect3D', keys=['points', f'{side_lidars[0]}_points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromSideLidars',
        lidars=side_lidars,
        transforms=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=file_client_args),
            dict(type='GlobalAlignment', rotation_axis=2),
        ]),
    # dict(type='CombineLidarsIntoSingleLidar', lidars=side_lidars),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        side_lidars=side_lidars,
        with_label=False),
    dict(type='Collect3D', keys=['points', f'{side_lidars[0]}_points'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'lyft_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'lyft_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'lyft_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True))
# For Lyft dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=24, pipeline=eval_pipeline)