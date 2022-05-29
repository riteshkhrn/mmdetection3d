# Copyright (c) OpenMMLab. All rights reserved.
import os
from os import path as osp
import time
import torch.optim as optim
import mmcv
import torch
from torch.utils.tensorboard import SummaryWriter
from mmcv.image import tensor2imgs
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet3d.core import bbox3d2result
import shutil
import tempfile
import pickle

def train(inf_model, cloc_model, train_data_loader, test_data_loader,
          tmpdir=None, gpu_collect=False, cfg=None, eval=None):
    """Train model with multiple gpus.

    This method trains model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        inf_model (nn.Module): Inference Model.
        cloc_model (nn.Module): CLOC fusion Model.
        train_data_loader (nn.Dataloader): Pytorch data loader(training).
        test_data_loader (nn.Dataloader): Pytorch data loader(training).
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        cfg (Config):
        eval:
    """
    inf_model.eval()
    rank, world_size = get_dist_info()
    batch_size = train_data_loader.batch_size
    side_lidar = 'LIDAR_FRONT_RIGHT_points'
    optimizer = optim.Adam(cloc_model.parameters(), lr=0.001, weight_decay=0.01)
    writer = SummaryWriter(cfg.cloc_runtime.log_dir + 'tf_logs')
    for epoch in range(cfg.cloc_runtime.epochs):
        # set the mode for cloc model
        cloc_model.train()
        running_loss = 0
        dataset = train_data_loader.dataset
        time.sleep(2)  # This line can prevent deadlock problem in some cases.
        step = 0
        for data in train_data_loader:
            if side_lidar in data:
                with torch.no_grad():
                    # test_time_aug will format the data to be a list so create a dummy list for train
                    result_3d = inf_model(return_loss=False, rescale=True,
                                        **dict(points=[data['points']], img_metas=[data['img_metas']]))
                    result_3d_side = inf_model(return_loss=False, rescale=True, side=True,
                                        **dict(points=[data[side_lidar]], img_metas=[data['img_metas']]))
                optimizer.zero_grad()
                # TODO: why gt_bboxes_3d and img_metas is nested list?
                pred_clses = cloc_model(result_3d, result_3d_side, data['img_metas'].data[0])
                loss = cloc_model.loss(pred_clses, result_3d, data['gt_bboxes_3d'].data[0],
                                    data['gt_labels_3d'].data[0], data['img_metas'].data[0])
                loss.backward()
                optimizer.step()
                step += 1
                # print statistics
                running_loss += loss
                if step % 50 == 0:
                    running_loss = (running_loss/50).detach().cpu().numpy()
                    print(f"\nEpoch {epoch}/{cfg.cloc_runtime.epochs}, Steps {step}/{len(dataset)/batch_size}, cls_loss {running_loss}")
                    writer.add_scalar('Loss/train', running_loss, step)
                    running_loss = 0
            else:
                print('No Side Lidar found, have you offset the dataset properly')
                print('If you have offset the dataset, check the config file  if the side lidars are collected')
                print('If this is the intended behaviour it will still work. This will go through the dataset untill side lidars are found')

        if (epoch + 1) % cfg.cloc_runtime.validation_interval == 0:
            print('\nSaving the model.. checkpoint')
            log_dir = cfg.cloc_runtime.log_dir
            if not osp.exists(log_dir):
                os.makedirs(log_dir)
            torch.save({
            'epoch': epoch,
            'model_state_dict': cloc_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, cfg.cloc_runtime.log_dir + 'cloc_model.pt')

            print('\nDoing Evaluation...')
            # set the mode for cloc model
            cloc_model.eval()
            results = []    
            dataset = test_data_loader.dataset
            if rank == 0:
                prog_bar = mmcv.ProgressBar(len(dataset))
            time.sleep(2)  # This line can prevent deadlock problem in some cases.
            for i, data in enumerate(test_data_loader):
                if side_lidar in data:
                    with torch.no_grad():
                        result_3d = inf_model(return_loss=False, rescale=True,
                                            **dict(points=data['points'], img_metas=data['img_metas']))
                        result_3d_side = inf_model(return_loss=False, rescale=True, side=True,
                                            **dict(points=data[side_lidar], img_metas=data['img_metas']))
                    # TODO: why gt_bboxes_3d and img_metas is nested list?
                    pred_clses = cloc_model(result_3d, result_3d_side, data['img_metas'][0].data[0])
                    # loss = cloc_model.loss(pred_clses, result_3d, data['gt_bboxes_3d'][0].data[0],
                    #                        data['gt_labels_3d'][0].data[0], data['img_metas'][0].data[0])           
                    mlvl_cls_scores, mlvl_bbox_preds, mlvl_dir_cls_preds = result_3d
                    final_3d = (pred_clses, mlvl_bbox_preds, mlvl_dir_cls_preds)
                else:
                    with torch.no_grad():
                        result_3d = inf_model(return_loss=False, rescale=True,
                                        **dict(points=data['points'], img_metas=data['img_metas']))
                        final_3d = result_3d
                with torch.no_grad():
                    bbox_list = inf_model.module.pts_bbox_head.get_bboxes(
                        *final_3d, data['img_metas'][0].data[0], rescale=True)
                    bbox_results = [
                        bbox3d2result(bboxes, scores, labels)
                        for bboxes, scores, labels in bbox_list
                    ]
                result = [dict() for i in range(len(data['img_metas'][0].data[0]))]
                for result_dict, pts_bbox in zip(result, bbox_results):
                    result_dict['pts_bbox'] = pts_bbox
                # # encode mask results
                # if isinstance(result[0], tuple):
                #     result = [(bbox_results, encode_mask_results(mask_results))
                #                 for bbox_results, mask_results in result]
                results.extend(result)
                if rank == 0:
                    batch_size = len(result)
                    for _ in range(batch_size * world_size):
                        prog_bar.update()

            # collect results from all ranks
            if gpu_collect:
                results = collect_results_gpu(results, len(dataset))
            else:
                results = collect_results_cpu(results, len(dataset), tmpdir)
            if eval:
                kwargs = {}
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=eval, **kwargs))
                print(dataset.evaluate(results, **eval_kwargs))


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

