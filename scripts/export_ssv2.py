import os
import torch
from transformers import AutoModelForVideoClassification

REPO = 'facebook/vjepa2-vitl-fpc16-256-ssv2'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, 'models', 'model.onnx')

print('loading', REPO)
model = AutoModelForVideoClassification.from_pretrained(REPO, dtype=torch.float32)
model.eval()

cfg = model.config
num_frames = cfg.frames_per_clip
size = cfg.crop_size
print(f'config: frames={num_frames} size={size} num_labels={cfg.num_labels}')

dummy = torch.zeros(1, num_frames, 3, size, size, dtype=torch.float32)

class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pixel_values_videos):
        return self.m(pixel_values_videos=pixel_values_videos).logits

w = Wrap(model)
w.eval()

print('exporting to', DST)
os.makedirs(os.path.dirname(DST), exist_ok=True)
torch.onnx.export(
    w, (dummy,), DST,
    input_names=['pixel_values_videos'],
    output_names=['logits'],
    opset_version=17,
    dynamic_axes={'pixel_values_videos': {0: 'batch'}, 'logits': {0: 'batch'}},
    do_constant_folding=True,
)
print('done', os.path.getsize(DST))
