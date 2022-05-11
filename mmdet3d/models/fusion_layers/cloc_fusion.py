import torch
from torch import nn as nn
import numpy as np
from abc import ABCMeta, abstractmethod
from mmdet3d.core.bbox.structures.base_box3d import BaseInstance3DBoxes
from ..builder import FUSION_LAYERS
from ..builder import build_loss
from mmdet.core import (build_bbox_coder, build_prior_generator, multi_apply)

def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=np.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  dense = torch.zeros(size).fill_(default_value)
  dense[indices] = indices_value

  return dense

class Loss(object):
  """Abstract base class for loss functions."""
  __metaclass__ = ABCMeta

  def __call__(self,
               prediction_tensor,
               target_tensor,
               ignore_nan_targets=False,
               scope=None,
               **params):
    """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
    if ignore_nan_targets:
      target_tensor = torch.where(torch.isnan(target_tensor),
                                prediction_tensor,
                                target_tensor)
    return self._compute_loss(prediction_tensor, target_tensor, **params)

  @abstractmethod
  def _compute_loss(self, prediction_tensor, target_tensor, **params):
    """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
    pass

def _sigmoid_cross_entropy_with_logits(logits, labels):
  # to be compatible with tensorflow, we don't use ignore_idx
  loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits) # this is the original
  #loss = torch.clamp(logits, min=0) - torch.exp(logits) * labels.type_as(logits)
  loss += torch.log1p(torch.exp(-torch.abs(logits)))
  loss_mask = (loss < 10000)
  loss_mask = loss_mask.type(torch.FloatTensor).cuda()
  loss = loss*loss_mask
  # transpose_param = [0] + [param[-1]] + param[1:-1]
  # logits = logits.permute(*transpose_param)
  # loss_ftor = nn.NLLLoss(reduce=False)
  # loss = loss_ftor(F.logsigmoid(logits), labels)
  return loss

class SigmoidFocalClassificationLoss(Loss):
  """Sigmoid focal cross entropy loss.

  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

  def __init__(self, gamma=2.0, alpha=0.25):
    """Constructor.

    Args:
      gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
      alpha: optional alpha weighting factor to balance positives vs negatives.
      all_zero_negative: bool. if True, will treat all zero as background.
        else, will treat first label as background. only affect alpha.
    """
    self._alpha = alpha
    self._gamma = gamma

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    # weights = weights.unsqueeze(2)
    if class_indices is not None:
      weights *= indices_to_dense_vector(class_indices,
            prediction_tensor.shape[2]).view(1, 1, -1).type_as(prediction_tensor)
    per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = torch.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if self._gamma:
      modulating_factor = torch.pow(1.0 - p_t, self._gamma)
    alpha_weight_factor = 1.0
    if self._alpha is not None:
      alpha_weight_factor = (target_tensor * self._alpha +
                              (1 - target_tensor) * (1 - self._alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor*per_entry_cross_ent)
    return focal_cross_entropy_loss * weights


@FUSION_LAYERS.register_module()
class CLOCFusion(nn.Module):
    def __init__(self,
                 num_classes=9,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
                 anchor_generator=dict(
                          type='AlignedAnchor3DRangeGenerator',
                          ranges=[[-80, -80, -1.8, 80, 80, -1.8]],
                          scales=[1, 2, 4],
                          sizes=[
                              [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
                              [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
                              [1., 1., 1.],
                              [0.4, 0.4, 1],
                          ],
                          custom_values=[],
                          rotations=[0, 1.57],
                          reshape_out=True),
                 train_cfg=None,
                 test_cfg=None):
        super(CLOCFusion, self).__init__()
        self.name = 'cloc_fusion_layer'
        self.num_classes = num_classes
        # build anchor generator
        self.anchor_generator = build_prior_generator(anchor_generator)
        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size
        self.fuse_2d_3d = nn.Sequential(
            nn.Conv2d(5,18,1),
            nn.ReLU(),
            nn.Conv2d(18,36,1),
            nn.ReLU(),
            nn.Conv2d(36,36,1),
            nn.ReLU(),
            nn.Conv2d(36,1,1),
        )
        self.maxpool = nn.Sequential(nn.MaxPool2d([300,1],1),)
        self.focal_loss = SigmoidFocalClassificationLoss()


    def forward(self, mlvl_det_3d, result_3d_side=None, img_metas=None):
        """Forward training function.

        Args:
            mlvl_det_3d
            -cls_scores (list[torch.Tensor]): Multi-level class scores.
            -bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            -dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            result_3d_side list[dict] containing the results
            -dict['pts_bbox'] - dict
            --dict[str, torch.Tensor]: Bounding box results in cpu mode.
            ---boxes_3d (torch.Tensor): 3D boxes.
            ---scores (torch.Tensor): Prediction scores.
            ---labels_3d (torch.Tensor): Box labels.
            ---attrs_3d (torch.Tensor, optional): Box attributes.           
            input_metas (list[dict]): Contain pcd and img's meta info.
        """
        mlvl_cls_scores, mlvl_bbox_preds, mlvl_dir_cls_preds = mlvl_det_3d
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds)
        assert len(mlvl_cls_scores) == len(mlvl_dir_cls_preds)
        num_levels = len(mlvl_cls_scores)
        featmap_sizes = [mlvl_cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = mlvl_cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        mlvl_anchors = [anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors]

        def fuse_single_sample(cls_score, bbox_pred, anchors, single_3d_side, img_meta):
            # Assuming all of inputs present
            # First task would be to create kXNx5 matrix
            #   - k is # of detections in side lidar
            #   - N is # of detections in top lidar
            #   - 5 is channel representing (IOUij, si, sj, di, dj)
            original_shape = cls_score.shape
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)
            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            bboxes = img_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
            bboxes_side = single_3d_side['pts_bbox']['boxes_3d'].to(device)
            scores_side = single_3d_side['pts_bbox']['scores_3d'].to(device)
            labels_side = single_3d_side['pts_bbox']['labels_3d'].to(device)
            cls_score_side = nn.functional.one_hot(labels_side, num_classes=self.num_classes) * scores_side.view(-1, 1)
            overlaps = BaseInstance3DBoxes.overlaps(bboxes_side, bboxes) #kXN
            indices = torch.nonzero(overlaps, as_tuple=True)
            features = torch.zeros(len(indices[0]), self.num_classes, 5,
                                   dtype=cls_score.dtype, device=device) #pXcX5
             # IOU, overlaps would be (p,) after reshaping (p,1), broadcasted to (p, c)
            features[:, :, 0] = overlaps[indices].reshape(-1, 1)
            features[:, :, 1] = cls_score_side[indices[0]] # score pXc
            features[:, :, 2] = cls_score[indices[1]] # score pXc
            features[:, :, 3] = torch.norm(bboxes_side.bottom_center[indices[0]], dim=1).reshape(-1, 1) # distance
            features[:, :, 4] = torch.norm(bboxes.bottom_center[indices[1]], dim=1).reshape(-1, 1) # distance
            features = features.unsqueeze(0).permute(0,3,2,1) # 1x5xcxp
            x = self.fuse_2d_3d(features) # 1x1xcxp
            out = torch.zeros(self.num_classes, len(bboxes_side), len(bboxes), 
                               dtype=features.dtype, device=features.device) # kxNxc
            out[:,:,:] = -9999999
            out[:, indices[0],indices[1]] = x[0, 0, :, :]
            x = self.maxpool(out) # cX1xN
            x = x.squeeze() # cxN
            x = x.permute(1, 0) # Nxc
            x = x.reshape(original_shape[1], original_shape[2], original_shape[0]) # fs[0]xfs[1]x(c*base_anchors)
            x = x.permute(2, 0, 1) # original_shape
            return x
            
        def fuse_single_level(slvl_cls_scores, slvl_bbox_preds, slvl_anchors):
          pred_clses = []
          for i in range(len(img_metas)):
            x = fuse_single_sample(slvl_cls_scores[i],
                                   slvl_bbox_preds[i],
                                   slvl_anchors,
                                   result_3d_side[i],
                                   img_metas[i])
            pred_clses.append(x.unsqueeze(0)) # create batch dim here
          return [torch.cat(pred_clses, dim=0)] #bxNxc, making it a list cause of zip in multi_apply

        pred_score = multi_apply(
                      fuse_single_level,
                      mlvl_cls_scores,
                      mlvl_bbox_preds,
                      mlvl_anchors)
        mlvl_pred_score = pred_score[0]
        assert len(mlvl_cls_scores) == len(mlvl_pred_score)
        return mlvl_pred_score #list[(bxNxc)]

    def loss(self, mlvl_pred_score=None,
                   mlvl_det_3d=None,
                   gt_bboxes_3d=None,
                   gt_labels_3d=None,
                   img_metas=None):
        """loss function.

        Args:
            pred_clses (list[torch.Tensor]): Output of pred class. Multi-level class scores.
            mlvl_det_3d
            -cls_scores (list[torch.Tensor]): Multi-level class scores.
            -bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            -dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]):
                Ground truth 3D boxes. Defaults to None.
        """
        mlvl_cls_scores, mlvl_bbox_preds, mlvl_dir_cls_preds = mlvl_det_3d
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds)
        assert len(mlvl_cls_scores) == len(mlvl_dir_cls_preds)
        assert len(mlvl_cls_scores) == len(mlvl_pred_score)
        num_levels = len(mlvl_cls_scores)
        featmap_sizes = [mlvl_cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = mlvl_cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        mlvl_anchors = [anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors]

        def loss_single_sample(pred_score, bbox_pred, anchors, gt_bboxes, gt_labels, img_meta):
            pred_score = pred_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)
            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            bboxes = img_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
            overlaps = BaseInstance3DBoxes.overlaps(gt_bboxes.to(device), bboxes) #gXN
            iou_amax = torch.amax(overlaps, dim=0) #N,
            iou_amax_ind = torch.argmax(overlaps, dim=0) #N,
            # create one hot of ground truth labels
            gt_scores = nn.functional.one_hot(gt_labels.to(device), num_classes=self.num_classes) #gXc
            # take N, indices for iou_amax out of gt_scores
            target_scores = gt_scores[iou_amax_ind] #NXc
            target_scale = ((iou_amax >= 0.7)*1).reshape(-1, 1) #NX1, 1 > 0.7 else 0
            target_scores = target_scores * target_scale
            positives = ((iou_amax >= 0.7)*1).reshape(-1, 1).type(iou_amax.dtype).to(device) #NX1
            negatives = ((iou_amax <= 0.5)*1).reshape(-1, 1).type(iou_amax.dtype).to(device) #NX1
            cls_weights = negatives + positives #NX1
            pos_normalizer = positives.sum(1, keepdim=True).type(iou_amax.dtype) #1X1
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            sample_loss = self.focal_loss._compute_loss(pred_score, target_scores, cls_weights).sum()
            return sample_loss

        def loss_single_level(slvl_pred_scores, slvl_bbox_preds, slvl_anchors):
          slvl_loss_sum = 0
          for i in range(len(img_metas)):
            sample_loss = loss_single_sample(slvl_pred_scores[i],
                                   slvl_bbox_preds[i],
                                   slvl_anchors,
                                   gt_bboxes_3d[i],
                                   gt_labels_3d[i],
                                   img_metas[i])
            slvl_loss_sum += sample_loss
          return [slvl_loss_sum/len(img_metas)]

        result = multi_apply(
                    loss_single_level,
                    mlvl_pred_score,
                    mlvl_bbox_preds,
                    mlvl_anchors)
        return sum(_loss.mean() for _loss in result[0])        