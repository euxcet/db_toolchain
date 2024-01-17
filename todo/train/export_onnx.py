import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import torch
from model.imu_gesture_model import GestureNetCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureNetCNN(num_classes=17)
model.load_state_dict(torch.load('best.pth', map_location=device))
model.eval()
torch.onnx.export(model, (torch.randn((1, 6, 1, 200))), 'best.onnx', input_names=['input_0'],
                  output_names=['output_0'], dynamic_axes={'input_0': [0], 'output_0': [0]})