import sys
import torch
sys.path.append(".")
from modules.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)





encoder_layer = TransformerEncoderLayer(d_model=256, nhead=4)
model = TransformerEncoder(encoder_layer, num_layers=2)

dummy_input = torch.randn(5, 10, 256)

output = model(dummy_input)
print("✅ Pass")
