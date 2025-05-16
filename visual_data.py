import numpy as np
import csv
import torch
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
# 1247    # trachea
# 1302    # right lung
# 1326    # left lung
# 170     # pancreas
# 187     # gallbladder
# 237     # urinary bladder
# 2473    # sternum
# 29193   # first lumbar vertebra
# 29662   # right kidney
# 29663   # left kidney
# 30324   # right adrenal gland
# 30325   # left adrenal gland
# 32248   # right psoas major
# 32249   # left psoas major
# 40357   # muscle body of right rectus abdominis
# 40358   # muscle body of left rectus abdominis
# 480     # aorta
# 58      # liver
# 7578    # thyroid gland
# 86      # spleen
# 0       # background
# 1       # body envelope
# 2       # thorax-abdomen
original_labels = [
    0, 1, 2, 58, 86, 170, 187, 237, 480, 1247, 1302, 1326, 2473,
    29193, 29662, 29663, 30324, 30325, 32248, 32249, 40357, 40358, 7578
]
label_map = {label: idx for idx, label in enumerate(sorted(original_labels))}  #{0:0 ; 1:1; 2:2; 58:3;...} Đổi các lớp thành thứ tự
print(label_map)
# csv_files = sorted(glob.glob("CTce_ThAb_b33x33_n1000_8bit/*.csv"))
# # names = [os.path.splitext(os.path.basename(path))[0] for path in csv_files]
# train_files, test_files = train_test_split(csv_files, test_size=0.2, random_state=42)
# # print(names)
# train_list = []
# test_list = []
# data_list= []
# def csv_to_data(file,label_map, liste):
#     with open(file, newline='') as csvfile:
#         counter = 0
#         for line in csvfile:
#             counter += 1 
#             numbers = list(map(float, line.strip().split(",")))
#             label = int(numbers[0])
#             pixels = numbers[1:]
#             imagette = torch.tensor(pixels).reshape(1, 33, 33)  # 1 channel, 33*33
#             liste.append((imagette, label_map[label]))
# for file in csv_files:
#     with open(file, newline='') as csvfile:
#         counter = 0
#         for line in csvfile:
#             numbers = list(map(float, line.strip().split(",")))
#             label = int(numbers[0])
#             pixels = numbers[1:]
#             imagette = torch.tensor(pixels).reshape(1, 33, 33)  # 1 channel, 33*33
#             basename = os.path.splitext(os.path.basename(file))[0]
#             data_list.append((imagette, label_map[label], basename))
# # # for file in train_files:
# # #     csv_to_data(file,label_map, train_list)


# # # for file in test_files:
# # #     csv_to_data(file,label_map, test_list)

# # # for file in csv_files: 
# # #     csv_to_data(file,label_map, data_list)
# # # torch.save(train_list, "traindata.path")
# # # torch.save(test_list, "testdata.path")
# torch.save(data_list, "data_all.path")
