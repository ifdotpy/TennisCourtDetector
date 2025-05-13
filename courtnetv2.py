import torch
import torch.nn as nn
import torchvision.models as models


class CourtFinderNetHeatmap(nn.Module):
    def __init__(self, num_keypoints=14):
        super(CourtFinderNetHeatmap, self).__init__()
        self.num_keypoints = num_keypoints

        mobilenet = models.mobilenet_v3_small(pretrained=True)

        self.backbone = mobilenet.features

        # Heatmap head
        self.conv_head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, self.num_keypoints, kernel_size=1)
        )

        # Upsample to original image size
        #self.upsample = nn.Upsample(size=(360, 640), mode='bilinear', align_corners=False)
        self.upsample = nn.Upsample(size=(360, 640), mode='bicubic', align_corners=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_head(x)
        x = self.upsample(x)
        return x


# Example usage

if __name__ == '__main__':
    # Example usage
    model = CourtFinderNetHeatmap()
    input_tensor = torch.randn(1, 1, 640, 360)
    output = model(input_tensor)
    print(output.shape)
