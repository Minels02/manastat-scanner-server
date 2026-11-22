import torch
import torchvision.models as models


class MobileNetV3Embedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = m.features
        self.avgpool = m.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


model = MobileNetV3Embedder().eval()
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "mobilenet_v3_small.onnx",
    input_names=["input"],
    output_names=["embedding"],
    opset_version=17,
)
print("Esportato mobilenet_v3_small.onnx con successo!")