# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .ops import resize


class DepthBaseDecodeHead(nn.Module):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (List): Input channels.
        channels (int): Channels after modules, before conv_depth.
        loss_decode (dict): Config of decode loss.
            Default: ().
        sampler (dict|None): The config of depth map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        min_depth (int): Min depth in dataset setting.
            Default: 1e-3.
        max_depth (int): Max depth in dataset setting.
            Default: None.
        norm_cfg (dict|None): Config of norm layers.
            Default: None.
        classify (bool): Whether predict depth in a cls.-reg. manner.
            Default: False.
        n_bins (int): The number of bins used in cls. step.
            Default: 256.
        bins_strategy (str): The discrete strategy used in cls. step.
            Default: 'UD'.
        norm_strategy (str): The norm strategy on cls. probability
            distribution. Default: 'linear'
        scale_up (str): Whether predict depth in a scale-up manner.
            Default: False.
    """

    def __init__(
        self,
        in_channels,
        channels=96,
        loss_decode=(),
        sampler=None,
        align_corners=False,
        min_depth=1e-3,
        max_depth=None,
        norm_cfg=None,
        classify=False,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
        scale_up=False,
    ):
        super(DepthBaseDecodeHead, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        self.align_corners = align_corners
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_cfg = norm_cfg
        self.classify = classify
        self.n_bins = n_bins
        self.scale_up = scale_up

        if self.classify:
            assert bins_strategy in ["UD", "SID"], "Support bins_strategy: UD, SID"
            assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

            self.bins_strategy = bins_strategy
            self.norm_strategy = norm_strategy
            self.softmax = nn.Softmax(dim=1)
            self.conv_depth = nn.Conv2d(channels, n_bins, kernel_size=3, padding=1, stride=1)
        else:
            self.conv_depth = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, img_metas):
        """Placeholder of forward function."""
        pass

    def forward_train(self, img, inputs, img_metas, depth_gt):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        depth_pred = self.forward(inputs, img_metas)
        losses = self.losses(depth_pred, depth_gt)

        log_imgs = self.log_images(img[0], depth_pred[0], depth_gt[0], img_metas[0])
        losses.update(**log_imgs)

        return losses

    def forward_test(self, inputs, img_metas):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.

        Returns:
            Tensor: Output depth map.
        """
        return self.forward(inputs, img_metas)

    def depth_pred(self, feat):
        """Prediction each pixel."""
        if self.classify:
            logit = self.conv_depth(feat)

            if self.bins_strategy == "UD":
                bins = torch.linspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)
            elif self.bins_strategy == "SID":
                bins = torch.logspace(self.min_depth, self.max_depth, self.n_bins, device=feat.device)

            # following Adabins, default linear
            if self.norm_strategy == "linear":
                logit = torch.relu(logit)
                eps = 0.1
                logit = logit + eps
                logit = logit / logit.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logit = torch.softmax(logit, dim=1)
            elif self.norm_strategy == "sigmoid":
                logit = torch.sigmoid(logit)
                logit = logit / logit.sum(dim=1, keepdim=True)

            output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)

        else:
            if self.scale_up:
                output = self.sigmoid(self.conv_depth(feat)) * self.max_depth
            else:
                output = self.relu(self.conv_depth(feat)) + self.min_depth
        return output

    def losses(self, depth_pred, depth_gt):
        """Compute depth loss."""
        loss = dict()
        depth_pred = resize(
            input=depth_pred, size=depth_gt.shape[2:], mode="bilinear", align_corners=self.align_corners, warning=False
        )
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(depth_pred, depth_gt)
            else:
                loss[loss_decode.loss_name] += loss_decode(depth_pred, depth_gt)
        return loss

    def log_images(self, img_path, depth_pred, depth_gt, img_meta):
        show_img = copy.deepcopy(img_path.detach().cpu().permute(1, 2, 0))
        show_img = show_img.numpy().astype(np.float32)
        show_img = mmcv.imdenormalize(
            show_img,
            img_meta["img_norm_cfg"]["mean"],
            img_meta["img_norm_cfg"]["std"],
            img_meta["img_norm_cfg"]["to_rgb"],
        )
        show_img = np.clip(show_img, 0, 255)
        show_img = show_img.astype(np.uint8)
        show_img = show_img[:, :, ::-1]
        show_img = show_img.transpose(0, 2, 1)
        show_img = show_img.transpose(1, 0, 2)

        depth_pred = depth_pred / torch.max(depth_pred)
        depth_gt = depth_gt / torch.max(depth_gt)

        depth_pred_color = copy.deepcopy(depth_pred.detach().cpu())
        depth_gt_color = copy.deepcopy(depth_gt.detach().cpu())

        return {"img_rgb": show_img, "img_depth_pred": depth_pred_color, "img_depth_gt": depth_gt_color}


class BNHead(DepthBaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self,
                 input_transform="resize_concat",
                 in_index=(0, 1, 2, 3),
                 upsample=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_transform = input_transform
        self.in_index = in_index
        self.upsample = upsample
        # self.bn = nn.SyncBatchNorm(self.in_channels)
        if self.classify:
            self.conv_depth = nn.Conv2d(self.channels, self.n_bins, kernel_size=1, padding=0, stride=1)
        else:
            self.conv_depth = nn.Conv2d(self.channels, 1, kernel_size=1, padding=0, stride=1)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if "concat" in self.input_transform:
            inputs = [inputs[i] for i in self.in_index]
            if "resize" in self.input_transform:
                inputs = [
                    resize(
                        input=x,
                        size=[s * self.upsample for s in inputs[0].shape[2:]],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
            inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs, img_metas=None, **kwargs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if len(x) == 2:
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        # feats = self.bn(x)
        return x

    def forward(self, inputs, img_metas=None, **kwargs):
        """Forward function."""
        output = self._forward_feature(inputs, img_metas=img_metas, **kwargs)
        output = self.depth_pred(output)
        return output
