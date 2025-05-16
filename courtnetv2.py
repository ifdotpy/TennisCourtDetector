import torch
import torch.nn as nn
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub


class CourtFinderNetHeatmap(nn.Module):
    def __init__(self, num_keypoints=14):
        super(CourtFinderNetHeatmap, self).__init__()
        self.num_keypoints = num_keypoints
        # QuantStub converts tensors from floating point to quantized.
        self.quant = QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        self.dequant = DeQuantStub()

        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        self.backbone = mobilenet.features

        # Heatmap head - Keep floating point for now, can be quantized later
        # Or ensure layers are QAT compatible. Conv2d/ReLU/BatchNorm2d usually are.
        self.conv_head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, self.num_keypoints, kernel_size=1)
        )

        # Upsample might need careful handling or replacement depending on qconfig backend support
        self.upsample = nn.Upsample(size=(360, 640), mode='bicubic', align_corners=False)

    def forward(self, x):
        # Manually specify where tensors will be converted from floating
        # point to quantized in the quantized model.
        x = self.quant(x)
        
        x = self.backbone(x)
        x = self.conv_head(x)
        
        # Manually specify where tensors will be converted from quantized
        # to floating point in the quantized model.
        x = self.dequant(x)

        # Upsample and sigmoid are often done in float after dequantization,
        # unless the specific quantization backend supports them well.
        x = self.upsample(x)
        x = torch.sigmoid(x) # Sigmoid might also affect quantization if done before dequant
        return x

    # Fuse layers for better performance and accuracy
    def fuse_model(self):
        # Fuse conv/bn/relu modules in backbone
        for module_name, module in self.backbone.named_modules():
             if isinstance(module, nn.Sequential) and len(module) == 3: # Example check
                 # Check for Conv-BN-ReLU pattern - This needs to be adapted based on MobileNetV3 structure
                 # Simplified example: fuse first Conv-BN-ReLU block if identifiable
                 # You'll need to inspect the actual MobileNetV3 structure and fuse accordingly.
                 # torch.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True) # If module[0]=Conv, module[1]=BN, module[2]=ReLU
                 pass # Placeholder: Implement actual fusion based on MobileNet structure

        # Fuse conv/bn/relu in the head if applicable
        # Example for the first Conv-BN-ReLU block in conv_head:
        # Assuming the structure is Conv2d -> ReLU -> BatchNorm2d
        # Note: ReLU is often fused *after* BN. Check your layer order.
        # If order is Conv -> BN -> ReLU:
        # torch.quantization.fuse_modules(self.conv_head, ['0', '2', '1'], inplace=True) # Fuse Conv(0), BN(2), ReLU(1)
        # If order is Conv -> ReLU -> BN: (less common but possible)
        torch.quantization.fuse_modules(self.conv_head, ['0', '1', '2'], inplace=True) # Fuse Conv(0), ReLU(1), BN(2) - Adjust indices as needed!
        
        # Fuse the final Conv in the head if needed (no BN/ReLU follows)
        # No fusion needed for self.conv_head[3] (Conv2d) by itself typically


# Example usage

if __name__ == '__main__':
    # Example usage
    model = CourtFinderNetHeatmap()
    input_tensor = torch.randn(1, 1, 640, 360)
    output = model(input_tensor)
    print(output.shape)
