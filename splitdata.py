from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch

# Giả sử bạn có nhãn thật và nhãn dự đoán:
y_true = [0, 1, 2, 2, 2, 1, 4, 2, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0,1,2]

# # Tính F1 macro:
# f1 = f1_score(y_true, y_pred, average='macro')
# print("F1 macro:", f1)

# # Ngoài ra có thể tính từng loại khác:
# f1_micro = f1_score(y_true, y_pred, average='micro')
# f1_weighted = f1_score(y_true, y_pred, average='weighted')

# # Nếu muốn xem F1 cho từng lớp:
# f1_per_class = f1_score(y_true, y_pred, average=None)
# print("F1 từng lớp:", f1_per_class)
tensor1 = torch.tensor(y_true).reshape(1,3 , 3)
tensor2 = torch.tensor(y_pred).reshape(1,3 , 3)
k = []
k.append(tensor1)
k.append(tensor2)
data = DataLoader(k, batch_size=2, shuffle=True) 
for X in data:
    print(X)
    break
    
