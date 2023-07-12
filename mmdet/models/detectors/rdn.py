import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers, requires_grad=True):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward_train(self, x, gt, mask):
        losses = dict()
        
        pred = self.forward(x)
        if mask.sum() == 0:
            # losses['super_resolution_loss'] = mask.sum() + 1e-9
            losses['super_resolution'] = mask.sum() + 1e-9
        else:
            # losses['super_resolution_loss'] = (torch.abs(pred - gt) * mask).sum() / (mask.sum()+1e-9)
            losses['super_resolution'] = (torch.abs(pred - gt) * mask).sum() / (mask.sum()+1e-9)
        
        return losses, pred

    # def forward_train(self, img, img_metas, gt_bboxes, gt_labels):
    #     losses = dict()
    #     real_face_mask = torch.zeros(img.shape).cuda().type(img.type())
    #     real_face_bbox = torch.ceil(gt_bboxes[0][gt_labels[0]==0,:])
    #     for idx in range(real_face_bbox.shape[0]):
    #         real_face_mask[:,:,int(real_face_bbox[idx,1]):int(real_face_bbox[idx,3]),int(real_face_bbox[idx,0]):int(real_face_bbox[idx,2])] = 1
        
    #     x = img[:,:,int(self.recon_scale/2)::self.recon_scale,int(self.recon_scale/2)::self.recon_scale]
    #     gt = img
    #     mask = real_face_mask

    #     pred = self.forward(x)
        
    #     losses['super_resolution'] = (torch.abs(pred - gt) * mask).sum() / (mask.sum()+1e-9)
        
    #     return losses, pred
        
    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x
    

class VGGLoss(nn.Module):
    # def __init__(self, gpu_ids):
    def __init__(self, vgg_requires_grad = False):
        super(VGGLoss, self).__init__()        
        # self.vgg = Vgg19().cuda(device = gpu_ids)
        # self.vgg = Vgg19().cuda()
        # self.vgg = vgg_model
        self.vgg = Vgg19(requires_grad=vgg_requires_grad)
        #self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # loss = 0
        # for i in range(len(x_vgg)):
        #     loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return x_vgg, y_vgg
    
    def forward_train(self, x, y, mask):
        losses = dict()              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i] * mask[i], y_vgg[i].detach() * mask[i])  
        if mask[0].sum() == 0:
            # losses['vgg_loss'] = loss + 1e-9
            losses['vgg'] = loss + 1e-9
        else:
            # losses['vgg_loss'] = loss
            losses['vgg'] = loss
        return losses, x_vgg, y_vgg
    # def forward_train(self, x, y):
    #     losses = dict()              
    #     x_vgg, y_vgg = self.vgg(x), self.vgg(y)
    #     loss = 0
    #     for i in range(len(x_vgg)):
    #         loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())  
    #     losses['vgg_loss'] = loss 
    #     return losses, x_vgg, y_vgg

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
