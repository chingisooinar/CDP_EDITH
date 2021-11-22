# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:26:14 2021

@author: user
"""

from api.AiModels import *
    
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels , 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )
        self.apply(weights_init_normal)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)
    
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x
    
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
def detect_edges(img):
    """
    input:
        numpy array(image)
    Why: this function is needed to extract edge maps (sketch images)
    
    output:
        edge map
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # first applies bilateral filter to remove noise but preserve edges
    img_gray = cv2.bilateralFilter(img_gray, 5, 50, 50)
    # next runs Canny edge detector to extract significant edges
    img_gray_edges = cv2.Canny(img_gray, 65, 110)
    # invert black/white
    img_gray_edges = cv2.bitwise_not(img_gray_edges) 
    # convert to rgb
    img_edges = cv2.cvtColor(img_gray_edges, cv2.COLOR_GRAY2RGB)
    return img_edges

def _draw_color_circles_on_src_img(img_src, img_target):
    """
    input:
        img_src: bw image
        img_target: ground truth color image
    Why: for demo puproses mainly. To leave color marks extracted from ground truth image
    
    Output: None. it leaves color marks inplace
    
    """
    # get non-white coordinates to leave meaningful color marks
    non_white_coords = _get_non_white_coordinates(img_target)
    # graw color circles using extracted coordinates
    for center_y, center_x in non_white_coords:
        _draw_color_circle_on_src_img(img_src, img_target, center_y, center_x)

def _get_non_white_coordinates( img):
    non_white_mask = np.sum(img, axis=-1) < 2.75
    non_white_y, non_white_x = np.nonzero(non_white_mask)
    # randomly sample non-white coordinates
    choices = [600]
    n_non_white = len(non_white_y)
    n_color_points = min(n_non_white, random.choice(choices))
    idxs = np.random.choice(n_non_white, n_color_points, replace=False)
    non_white_coords = list(zip(non_white_y[idxs], non_white_x[idxs]))
    return non_white_coords

def _draw_color_circle_on_src_img( img_src, img_target, center_y, center_x):
    assert img_src.shape == img_target.shape, "Image source and target must have same shape."
    y0, y1, x0, x1 = _get_color_point_bbox_coords(center_y, center_x)
    color = np.mean(img_target[y0:y1, x0:x1], axis=(0, 1))
    img_src[y0:y1, x0:x1] = color

def _get_color_point_bbox_coords(center_y, center_x):
    radius = 2
    y0 = max(0, center_y-radius+1)
    y1 = min(256, center_y+radius)
    x0 = max(0, center_x-radius+1)
    x1 = min(256, center_x+radius)
    return y0, y1, x0, x1

def mark(img_gray, image):
    """
    input:
        img_gray: bw image
        image: ground truth color image
    Why: Preprocesses images and calls the main function
    
    Output: marked bw image
    
    """

    img_gray = (img_gray - 127.5)/127.5
    image = (image - 127.5)/127.5
    _draw_color_circles_on_src_img(img_gray, image)
    img_gray = (img_gray * 127.5) + 127.5
    image = (image * 127.5) + 127.5

    return img_gray.astype('uint8')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)