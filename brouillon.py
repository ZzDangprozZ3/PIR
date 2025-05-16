import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
# organ_dict = {
#     1247: "trachea",
#     1302: "right lung",
#     1326: "left lung",
#     170: "pancreas",
#     187: "gallbladder",
#     237: "urinary bladder",
#     2473: "sternum",
#     29193: "first lumbar vertebra",
#     29662: "right kidney",
#     29663: "left kidney",
#     30324: "right adrenal gland",
#     30325: "left adrenal gland",
#     32248: "right psoas major",
#     32249: "left psoas major",
#     40357: "muscle body of right rectus abdominis",
#     40358: "muscle body of left rectus abdominis",
#     480: "aorta",
#     58: "liver",
#     7578: "thyroid gland",
#     86: "spleen",
#     0: "background",
#     1: "body envelope",
#     2: "thorax-abdomen"
# }

import matplotlib.pyplot as plt
import numpy as np

# Danh sách cơ quan
organs = [
    "liver", "spleen", "pancreas", "gallbladder", "urinary bladder",
    "aorta", "trachea", "right lung", "left lung", "sternum",
    "thyroid gland", "first lumbar vertebra", "right kidney", "left kidney",
    "right adrenal gland", "left adrenal gland", "right psoas major",
    "left psoas major", "right rectus abdominis", "left rectus abdominis"
]

# Dữ liệu lần 1
means1 = [
    62.66, 109.96, 72.64, 97.23, 196.80,
    48.02, 312.85, 103.31, 98.95, 225.39,
    303.48, 46.81, 73.60, 81.71,
    99.95, 110.19, 82.59,
    78.23, 146.73, 109.17
]

stds1 = [
    15.47, 13.04, 18.28, 21.19, 33.62,
    10.62, 38.92, 16.76, 14.50, 18.58,
    22.16, 15.20, 8.32, 7.17,
    16.25, 18.60, 17.66,
    18.79, 13.48, 18.09
]

# Dữ liệu lần 2
means2 = [
    63.04, 110.23, 72.96, 96.85, 196.61,
    47.58, 313.15, 102.60, 99.06, 226.30,
    303.91, 45.90, 73.59, 82.19,
    100.85, 109.13, 81.73,
    77.67, 145.67, 109.17
]

stds2 = [
    15.25, 12.73, 17.93, 20.71, 32.95,
    10.25, 38.29, 16.40, 14.54, 18.37,
    22.13, 15.37, 8.23, 7.16,
    16.38, 18.16, 17.49,
    18.54, 13.51, 18.44
]

# Thiết lập vị trí và độ rộng cột
x = np.arange(len(organs))
width = 0.4

# Vẽ biểu đồ
plt.figure(figsize=(16, 10))
plt.barh(x - width/2, means1, height=width, xerr=stds1, label='Run 1', color='skyblue', edgecolor='black')
plt.barh(x + width/2, means2, height=width, xerr=stds2, label='Run 2', color='salmon', edgecolor='black')

plt.yticks(x, organs)
plt.xlabel('Distance Error (mm)')
plt.title('Comparison of Distance Error by Organ (Run 1 vs Run 2)')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Study statistics: 
#   Number of finished trials:  50
#   Number of pruned trials:  32
#   Number of complete trials:  18
# Best trial:
#   Value:  0.2786792604145335
#   Params:
#     n_conv_layers: 2
#     conv_out_channels_0: 32
#     conv_out_channels_1: 32
#     n_linear_layers: 1
#     linear_units_0: 561
#     dropout_0: 0.43269891871173555
#     optimizer: RMSprop
#     lr: 1.0082099500553373e-06


# Study statistics: 
#   Number of finished trials:  10
#   Number of pruned trials:  2
#   Number of complete trials:  8
# Best trial:
#   Value:  0.78125
#   Params:
#     n_conv_layers: 1
#     conv_out_channels_0: 32
#     n_linear_layers: 3
#     linear_units_0: 327
#     dropout_0: 0.4445930744441742
#     linear_units_1: 263
#     dropout_1: 0.4631803048206481
#     linear_units_2: 207
#     dropout_2: 0.23263017415281362
#     optimizer: RMSprop
#     lr: 0.011924133285215112




# Study statistics: 
#   Number of finished trials:  10
#   Number of pruned trials:  1
#   Number of complete trials:  9
# Best trial:
#   Value:  0.43859649122807015
#   Params:
#     n_conv_layers: 1
#     conv_out_channels_0: 32
#     n_linear_layers: 2
#     linear_units_0: 292
#     dropout_0: 0.45772055868604666
#     linear_units_1: 822
#     dropout_1: 0.4482467933734694
#     optimizer: RMSprop
#     lr: 0.006352284420524667