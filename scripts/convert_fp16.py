import os, sys, onnx
from onnxconverter_common import float16

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'models', 'model.onnx')
DST = os.path.join(ROOT, 'models', 'model.fp16.onnx')

if not os.path.isfile(SRC):
    sys.exit(f'missing {SRC}')

print('loading', SRC)
m = onnx.load(SRC)
print('converting to fp16…')
m16 = float16.convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=True)
print('saving', DST)
onnx.save(m16, DST, save_as_external_data=False)
print('done', os.path.getsize(DST))
