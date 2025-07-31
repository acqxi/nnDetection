"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Tuple, Optional, TypeVar
from abc import abstractmethod

from nndet.core.boxes import BoxCoderND
from nndet.core.boxes.sampler import AbstractSampler
from nndet.arch.heads.classifier import Classifier
from nndet.arch.heads.regressor import Regressor
from nndet.arch.heads.instance_segmenter import InstanceSegmenter


class AbstractHead(nn.Module):
    """
    Provides an abstract interface for an module which takes
    inputs and computed its own loss
    """

    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute forward pass
        
        Args
            x: feature maps
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess_for_inference(
        self,
        prediction: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head
            List[torch.Tensor]: anchors per image
        """
        raise NotImplementedError


class DetectionHead(AbstractHead):

    def __init__(
        self,
        classifier: Classifier,
        regressor: Regressor,
        coder: BoxCoderND,
    ):
        """
        Detection head with classifier and regression module
        
        Args:
            classifier: classifier module
            regressor: regression module
        """
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor
        self.coder = coder

    def forward(
        self,
        fmaps: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through head modules

        Args:
            fmaps: list of feature maps for head module

        Returns:
            Dict[str, torch.Tensor]: predictions
                `box_deltas`(Tensor): bounding box offsets
                    [Num_Anchors_Batch, (dim * 2)]
                `box_logits`(Tensor): classification logits
                    [Num_Anchors_Batch, (num_classes)]
                `inst_masks`(Tensor): instance mask logits (if instance_segmenter exists)
                    [Num_Anchors_Batch, (7 * 28 * 28)]
        """
        logits, offsets = [], []
        inst_masks = []

        for level, p in enumerate(fmaps):
            logits.append(self.classifier(p, level=level))
            offsets.append(self.regressor(p, level=level))
            if self.instance_segmenter is not None:
                inst_masks.append(
                    self.instance_segmenter.forward(p, level=level))

        sdim = fmaps[0].ndim - 2
        box_deltas = torch.cat(offsets, dim=1).reshape(-1, sdim * 2)
        box_logits = torch.cat(logits, dim=1).flatten(0, -2)

        result = {"box_deltas": box_deltas, "box_logits": box_logits}

        if self.instance_segmenter is not None:
            inst_masks = torch.cat(inst_masks, dim=1).flatten(0, -2)
            result["inst_masks"] = inst_masks

        return result

    @abstractmethod
    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation
                `box_logits`: classification logits for each anchor [N]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels: target labels for each anchor (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        raise NotImplementedError

    def postprocess_for_inference(
        self,
        prediction: Dict[str, torch.Tensor],
        anchors: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess predictions for inference e.g. ocnvert logits to probs

        Args:
            Dict[str, torch.Tensor]: predictions from this head
                `box_logits`: classification logits for each anchor [N]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            List[torch.Tensor]: anchors per image
        """
        postprocess_predictions = {
            "pred_boxes":
            self.coder.decode(prediction["box_deltas"], anchors),
            "pred_probs":
            self.classifier.box_logits_to_probs(prediction["box_logits"]),
        }
        return postprocess_predictions


class DetectionHeadHNM(DetectionHead):

    def __init__(
        self,
        classifier: Classifier,
        regressor: Regressor,
        coder: BoxCoderND,
        sampler: AbstractSampler,
        instance_segmenter: Optional[InstanceSegmenter] = None,
        log_num_anchors: Optional[str] = "mllogger",
    ):
        """
        Detection head with classifier and regression module. Uses hard negative
        example mining to compute loss

        Args:
            classifier: classifier module
            regressor: regression module
            sampler (AbstractSampler): sampler for select positive and
                negative examples
            instance_segmenter: instance segmentation module (optional)
            log_num_anchors (str): name of logger to use; if None, no logging
                will be performed
        """
        super().__init__(classifier=classifier,
                         regressor=regressor,
                         coder=coder)

        self.logger = None  # get_logger(log_num_anchors) if log_num_anchors is not None else None
        self.fg_bg_sampler = sampler
        self.instance_segmenter = instance_segmenter

    def prepare_gt_masks_for_loss(
            self,
            sampled_pos_inds: Tensor,
            target_labels: List[Tensor],
            matched_gt_boxes: List[Tensor],
            anchors: List[Tensor],
            target_instance_seg: List[Tensor],
            output_size: Tuple[int, int, int] = (7, 28, 28),
    ) -> Tensor:
        """
        Prepare ground truth masks for positive anchors

        Args:
            sampled_pos_inds: positive sample indices
            target_labels: target labels per image
            matched_gt_boxes: matched gt boxes per image
            anchors: anchors per image
            target_instance_seg: instance segmentation maps per image
            output_size: output mask size (z, y, x)

        Returns:
            Tensor: ground truth masks [num_positive_samples, 7*28*28]
        """
        if sampled_pos_inds.numel() == 0:
            return torch.empty(
                (0, output_size[0] * output_size[1] * output_size[2]),
                device=sampled_pos_inds.device,
                dtype=torch.float32)

        # Concatenate all data
        batch_target_labels = torch.cat(target_labels, dim=0)
        batch_matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)

        # Get positive samples
        pos_target_labels = batch_target_labels[sampled_pos_inds]
        pos_matched_gt_boxes = batch_matched_gt_boxes[sampled_pos_inds]

        num_pos = sampled_pos_inds.numel()
        mask_size = output_size[0] * output_size[1] * output_size[2]
        target_masks = torch.zeros((num_pos, mask_size),
                                   device=sampled_pos_inds.device,
                                   dtype=torch.float32)

        # Track which image each positive sample belongs to
        img_idx = 0
        anchor_count = 0

        for i, (gt_boxes_per_img, anchors_per_img,
                inst_seg_per_img) in enumerate(
                    zip(matched_gt_boxes, anchors, target_instance_seg)):

            next_anchor_count = anchor_count + len(anchors_per_img)

            # Find positive samples in this image
            pos_in_img_mask = (sampled_pos_inds >= anchor_count) & (
                sampled_pos_inds < next_anchor_count)
            pos_in_img_indices = sampled_pos_inds[
                pos_in_img_mask] - anchor_count

            if pos_in_img_indices.numel() > 0:
                # Get matched GT boxes and labels for positive samples in this image
                matched_boxes = gt_boxes_per_img[pos_in_img_indices]
                matched_labels = batch_target_labels[
                    sampled_pos_inds[pos_in_img_mask]]

                # Create masks for each positive sample
                for j, (box,
                        label) in enumerate(zip(matched_boxes,
                                                matched_labels)):
                    if label > 0:  # foreground
                        # Extract box coordinates
                        if box.numel() == 4:  # 2D case
                            x1, y1, x2, y2 = box.int()
                            # Handle 3D by taking the entire z-range
                            z1, z2 = torch.tensor(0), torch.tensor(
                                inst_seg_per_img.shape[0])
                        else:  # 3D case
                            x1, y1, z1, x2, y2, z2 = box.int()

                        # Convert to integers and ensure coordinates are within bounds
                        z1 = max(0, int(z1.item()))
                        z2 = min(inst_seg_per_img.shape[0], int(z2.item()))
                        y1 = max(0, int(y1.item()))
                        y2 = min(inst_seg_per_img.shape[1], int(y2.item()))
                        x1 = max(0, int(x1.item()))
                        x2 = min(inst_seg_per_img.shape[2], int(x2.item()))

                        if z2 > z1 and y2 > y1 and x2 > x1:
                            # Extract the region
                            region = inst_seg_per_img[z1:z2, y1:y2, x1:x2]

                            # Create binary mask for this instance (label value)
                            instance_mask = (region == label.item()).float()

                            # Resize to target size
                            if instance_mask.numel() > 0:
                                # Add batch dimension for interpolation
                                instance_mask = instance_mask.unsqueeze(
                                    0).unsqueeze(0)
                                resized_mask = F.interpolate(
                                    instance_mask,
                                    size=output_size,
                                    mode='trilinear'
                                    if len(output_size) == 3 else 'bilinear',
                                    align_corners=False)
                                # Remove batch and channel dimensions and flatten
                                mask_idx = torch.where(pos_in_img_mask)[0][j]
                                target_masks[mask_idx] = resized_mask.squeeze(
                                ).flatten()

            anchor_count = next_anchor_count

        return target_masks

    def forward(
        self,
        fmaps: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward feature maps through head modules

        Args:
            fmaps: list of feature maps for head module

        Returns:
            Dict[str, torch.Tensor]: predictions
                `box_deltas`(Tensor): bounding box offsets
                    [Num_Anchors_Batch, (dim * 2)]
                `box_logits`(Tensor): classification logits
                    [Num_Anchors_Batch, (num_classes)]
                `inst_masks`(Tensor): instance mask logits (if instance_segmenter exists)
                    [Num_Anchors_Batch, (7 * 28 * 28)]
        """
        logits, offsets = [], []
        inst_masks = []

        for level, p in enumerate(fmaps):
            logits.append(self.classifier(p, level=level))
            offsets.append(self.regressor(p, level=level))
            if self.instance_segmenter is not None:
                inst_masks.append(self.instance_segmenter(p, level=level))

        sdim = fmaps[0].ndim - 2
        box_deltas = torch.cat(offsets, dim=1).reshape(-1, sdim * 2)
        box_logits = torch.cat(logits, dim=1).flatten(0, -2)

        result = {"box_deltas": box_deltas, "box_logits": box_logits}

        if self.instance_segmenter is not None:
            inst_masks = torch.cat(inst_masks, dim=1).flatten(0, -2)
            result["inst_masks"] = inst_masks

        return result

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
        target_instance_seg: Optional[List[Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
                inst_masks (Tensor): instance mask logits (optional)
                    [N, 7 * 28 * 28]
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]
            target_instance_seg: instance segmentation maps per image (optional)

        Returns:
            Tensor: dict with losses (reg for regression loss, cls
                for classification loss, inst_seg for instance segmentation loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction[
            "box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(
            target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels_cat = torch.cat(target_labels, dim=0)

        with torch.no_grad():
            batch_matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
            batch_anchors = torch.cat(anchors, dim=0)
            target_deltas_sampled = self.coder.encode_single(
                batch_matched_gt_boxes[sampled_pos_inds],
                batch_anchors[sampled_pos_inds],
            )

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                box_deltas[sampled_pos_inds],
                target_deltas_sampled,
            ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels_cat[sampled_inds])

        # Add instance segmentation loss
        if (self.instance_segmenter is not None and "inst_masks" in prediction
                and target_instance_seg is not None
                and sampled_pos_inds.numel() > 0):

            pred_masks = prediction["inst_masks"][sampled_pos_inds]
            target_masks = self.prepare_gt_masks_for_loss(
                sampled_pos_inds,
                target_labels,
                matched_gt_boxes,
                anchors,
                target_instance_seg,
                output_size=(7, 28, 28),
            )

            # Ensure pred_masks and target_masks have same shape
            pred_masks = pred_masks.view(-1, 7 * 28 * 28)
            target_masks = target_masks.view(-1, 7 * 28 * 28)

            losses["inst_seg"] = self.instance_segmenter.compute_loss(
                pred_masks, target_masks)

        return losses, sampled_pos_inds, sampled_neg_inds

    def select_indices(
        self,
        target_labels: List[Tensor],
        boxes_scores: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample positive and negative anchors from target labels

        Args:
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            boxes_scores (Tensor): classification logits for each anchor
                [N, num_classes]

        Returns:
            Tensor: sampled positive indices [R]
            Tensor: sampled negative indices [R]
        """
        boxes_max_fg_probs = self.classifier.box_logits_to_probs(boxes_scores)
        boxes_max_fg_probs = boxes_max_fg_probs.max(
            dim=1)[0]  # search max of fg probs

        # positive and negative anchor indices per image
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(
            target_labels, boxes_max_fg_probs)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # if self.logger:
        #     self.logger.add_scalar("train/num_pos", sampled_pos_inds.numel())
        #     self.logger.add_scalar("train/num_neg", sampled_neg_inds.numel())

        return sampled_pos_inds, sampled_neg_inds


class BoxHeadNoSampler(DetectionHead):

    def __init__(self,
                 classifier: Classifier,
                 regressor: Regressor,
                 coder: BoxCoderND,
                 log_num_anchors: Optional[str] = "mllogger",
                 **kwargs):
        """
        Detection head with classifier and regression module. Uses all
        foreground anchors for regression an passes all anchors to classifier

        Args:
            classifier: classifier module
            regressor: regression module
            log_num_anchors (str): name of logger to use; if None, no
                logging will be performed
        """
        super().__init__(classifier=classifier,
                         regressor=regressor,
                         coder=coder)
        self.logger = None  # get_logger(log_num_anchors) if log_num_anchors is not None else None

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels: target labels for each anchor (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction[
            "box_deltas"]

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes = self.coder.decode_single(box_deltas, batch_anchors)
        target_boxes = torch.cat(matched_gt_boxes, dim=0)

        sampled_inds = torch.where(target_labels >= 0)[0]
        sampled_pos_inds = torch.where(target_labels >= 1)[0]

        losses = {}
        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes[sampled_pos_inds],
                target_boxes[sampled_pos_inds],
            ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds],
            target_labels[sampled_inds],
        ) / max(1, sampled_pos_inds.numel())
        return losses, sampled_pos_inds, None


class DetectionHeadHNMNative(DetectionHeadHNM):

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        This head decodes the relative offsets from the networks and computes
        the regression loss directly on the bounding boxes (e.g. for GIoU loss)

        Args:
            prediction: detection predictions for loss computation
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction[
            "box_deltas"]

        with torch.no_grad():
            losses = {}
            sampled_pos_inds, sampled_neg_inds = self.select_indices(
                target_labels, box_logits)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds],
                                     dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)
        pred_boxes_sampled = self.coder.decode_single(
            box_deltas[sampled_pos_inds], batch_anchors[sampled_pos_inds])

        target_boxes_sampled = torch.cat(matched_gt_boxes,
                                         dim=0)[sampled_pos_inds]

        if sampled_pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes_sampled,
                target_boxes_sampled,
            ) / max(1, sampled_pos_inds.numel())

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])
        return losses, sampled_pos_inds, sampled_neg_inds


class DetectionHeadHNMNativeRegAll(DetectionHeadHNM):

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        This head decodes the relative offsets from the networks and computes
        the regression loss directly on the bounding boxes (e.g. for GIoU loss)

        Args:
            prediction: detection predictions for loss computation
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls for
                classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction[
            "box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(
            target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        target_labels = torch.cat(target_labels, dim=0)
        batch_anchors = torch.cat(anchors, dim=0)

        assert len(batch_anchors) == len(box_deltas)
        assert len(batch_anchors) == len(box_logits)
        assert len(batch_anchors) == len(target_labels)

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        pred_boxes = self.coder.decode_single(box_deltas[pos_inds],
                                              batch_anchors[pos_inds])
        target_boxes = torch.cat(matched_gt_boxes, dim=0)[pos_inds]

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                pred_boxes,
                target_boxes,
            ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


class DetectionHeadHNMRegAll(DetectionHeadHNM):

    def compute_loss(
        self,
        prediction: Dict[str, Tensor],
        target_labels: List[Tensor],
        matched_gt_boxes: List[Tensor],
        anchors: List[Tensor],
    ) -> Tuple[Dict[str, Tensor], torch.Tensor, torch.Tensor]:
        """
        Compute regression and classification loss
        N anchors over all images; M anchors per image => sum(M) = N

        Args:
            prediction: detection predictions for loss computation
                box_logits (Tensor): classification logits for each anchor
                    [N, num_classes]
                box_deltas (Tensor): offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            target_labels (List[Tensor]): target labels for each anchor
                (per image) [M]
            matched_gt_boxes: matched gt box for each anchor
                List[[N, dim *  2]], N=number of anchors per image
            anchors: anchors per image List[[N, dim *  2]]

        Returns:
            Tensor: dict with losses (reg for regression loss, cls
                for classification loss)
            Tensor: sampled positive indices of anchors (after concatenation)
            Tensor: sampled negative indices of anchors (after concatenation)
        """
        box_logits, box_deltas = prediction["box_logits"], prediction[
            "box_deltas"]

        losses = {}
        sampled_pos_inds, sampled_neg_inds = self.select_indices(
            target_labels, box_logits)
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        target_labels = torch.cat(target_labels, dim=0)

        losses["cls"] = self.classifier.compute_loss(
            box_logits[sampled_inds], target_labels[sampled_inds])

        pos_inds = torch.where(target_labels >= 1)[0]
        with torch.no_grad():
            batch_matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)
            batch_anchors = torch.cat(anchors, dim=0)
            target_deltas_sampled = self.coder.encode_single(
                batch_matched_gt_boxes[pos_inds],
                batch_anchors[pos_inds],
            )

        assert len(batch_anchors) == len(batch_matched_gt_boxes)
        assert len(batch_anchors) == len(box_deltas)
        assert len(batch_anchors) == len(box_logits)
        assert len(batch_anchors) == len(target_labels)

        if pos_inds.numel() > 0:
            losses["reg"] = self.regressor.compute_loss(
                box_deltas[pos_inds],
                target_deltas_sampled,
            ) / max(1, pos_inds.numel())

        return losses, sampled_pos_inds, sampled_neg_inds


HeadType = TypeVar('HeadType', bound=AbstractHead)
