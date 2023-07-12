# two_stage_pixel_stage5.py
import torch
import torchvision
import torch.nn as nn
from PIL import Image as pil_image
import torchvision.transforms as transforms

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .rdn import RDN, VGGLoss, Vgg19

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 recon=None):
        super(TwoStageDetector, self).__init__()
        self.recon_scale = recon['scale']
        self.recon = RDN(scale_factor=recon['scale'],
                num_channels=recon['num_channels'],
                num_features=recon['num_features'],
                growth_rate=recon['growth_rate'],
                num_blocks=recon['num_blocks'],
                num_layers=recon['num_layers'])

        for param in self.recon.parameters():
            param.requires_grad = False
        
        self.VGGLoss = VGGLoss(vgg_requires_grad=False)

        self.backbone = build_backbone(backbone)
        
        self.recon_error_scale = torch.tensor([3.5455, 4.5596, 3.0827]).reshape((1,3,1,1))
        self.recon_error_shift = torch.tensor([-1.6310, -1.4675, -1.3786]).reshape((1,3,1,1))


        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained, rdn_pretrained=recon['rdn_pretrained'])



    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None, rdn_pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStageDetector, self).init_weights(pretrained)
        if rdn_pretrained is not None:
            self.recon.load_state_dict(torch.load(rdn_pretrained, map_location='cpu'))
        if self.backbone.patch_embed.in_chans == 6 and pretrained is not None:
            checkpoint = torch.load(pretrained, map_location='cpu')
            weight = checkpoint['model']['patch_embed.proj.weight']
            new_weight = torch.cat((weight, weight), 1)
            checkpoint['model']['patch_embed.proj.weight'] = new_weight
            torch.save(checkpoint, './pretrained/pretrained_in_chans_6.pth')
            pretrained = 'pretrained/pretrained_in_chans_6.pth'

        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img, stage5_error):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img, stage5_error)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        real_face_mask = torch.zeros(img.shape[0], 1, img.shape[2], img.shape[3]).cuda().type(img.type())
        for img_idx in range(img.shape[0]):
            real_face_bbox = torch.ceil(gt_bboxes[img_idx][gt_labels[img_idx]==0,:])
            for idx in range(real_face_bbox.shape[0]):
                real_face_mask[img_idx,:,int(real_face_bbox[idx,1]):int(real_face_bbox[idx,3]),int(real_face_bbox[idx,0]):int(real_face_bbox[idx,2])] = 1
    
        loss_recon, img_recon = self.recon.forward_train(img[:,:,int(self.recon_scale/2)::self.recon_scale,int(self.recon_scale/2)::self.recon_scale], img, real_face_mask.expand(img.shape))
        recon_error = torch.abs(img_recon - img)

        real_face_mask_stage = []
        real_face_mask_stage.append(real_face_mask.expand(real_face_mask.shape[0], 64, real_face_mask.shape[2], real_face_mask.shape[3]))
        real_face_mask_stage.append(real_face_mask[:, :, 1::2, 1::2].expand(real_face_mask[:, :, 1::2, 1::2].shape[0], 128, real_face_mask[:, :, 1::2, 1::2].shape[2], real_face_mask[:, :, 1::2, 1::2].shape[3]))
        real_face_mask_stage.append(real_face_mask[:, :, 2::4, 2::4].expand(real_face_mask[:, :, 2::4, 2::4].shape[0], 256, real_face_mask[:, :, 2::4, 2::4].shape[2], real_face_mask[:, :, 2::4, 2::4].shape[3]))
        real_face_mask_stage.append(real_face_mask[:, :, 4::8, 4::8].expand(real_face_mask[:, :, 4::8, 4::8].shape[0], 512, real_face_mask[:, :, 4::8, 4::8].shape[2], real_face_mask[:, :, 4::8, 4::8].shape[3]))
        real_face_mask_stage.append(real_face_mask[:, :, 8::16, 8::16].expand(real_face_mask[:, :, 8::16, 8::16].shape[0], 512, real_face_mask[:, :, 8::16, 8::16].shape[2], real_face_mask[:, :, 8::16, 8::16].shape[3]))
        vgg_loss, img_recon_vgg_output, img_vgg_output = self.VGGLoss.forward_train(img_recon, img, real_face_mask_stage)
        
        stage5_error = torch.abs(img_recon_vgg_output[4] - img_vgg_output[4])
        recon_error = recon_error * self.recon_error_scale.cuda() + self.recon_error_shift.cuda()

        img_recon_error = torch.cat((img, recon_error),1)
        x = self.extract_feat(img_recon_error, stage5_error)


        losses = dict()
        losses.update(loss_recon)
        losses.update(vgg_loss)
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        img_recon = self.recon(img[:,:,int(self.recon_scale/2)::self.recon_scale,int(self.recon_scale/2)::self.recon_scale])
        img_recon_vgg_output, img_vgg_output = self.VGGLoss(img_recon, img)
        recon_error = torch.abs(img_recon - img)
        stage5_error = torch.abs(img_recon_vgg_output[4] - img_vgg_output[4])

        recon_error = recon_error * self.recon_error_scale.cuda() + self.recon_error_shift.cuda()
        img_recon_error = torch.cat((img, recon_error),1)
        x = self.extract_feat(img_recon_error, stage5_error)


        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'


        img_recon = self.recon(img[:,:,int(self.recon_scale/2)::self.recon_scale,int(self.recon_scale/2)::self.recon_scale])
        recon_error = torch.abs(img_recon - img)
        img_recon_vgg_output, img_vgg_output = self.VGGLoss(img_recon, img)
        stage5_error = torch.abs(img_recon_vgg_output[4] - img_vgg_output[4])
        recon_error = recon_error * self.recon_error_scale.cuda() + self.recon_error_shift.cuda()
        img_recon_error = torch.cat((img, recon_error),1)
        x = self.extract_feat(img_recon_error, stage5_error)



        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        img_recon = self.recon(imgs[:,:,int(self.recon_scale/2)::self.recon_scale,int(self.recon_scale/2)::self.recon_scale])
        recon_error = torch.abs(img_recon - imgs)
        recon_error = recon_error * self.recon_error_scale.cuda() + self.recon_error_shift.cuda()
        x = torch.cat((imgs, recon_error),1)
        x = self.extract_feat(x)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)