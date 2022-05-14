
import onnxruntime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

onnx_path = os.path.join(Config.model_path,"model.onnx"
sess = onnxruntime.InferenceSession(onnx_path)


output = sess.run(['output'], {'input' : mnist_data[0].numpy()})

out = np.array(output)
out = np.squeeze(out)
print(out.shape)
print(np.argmax(out, 1))