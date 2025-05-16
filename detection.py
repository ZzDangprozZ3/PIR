import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import f1_score
import torch.nn.functional as F
from sklearn.model_selection import KFold
import glob
import os
import nibabel as nib
import numpy as np
from skimage.util import view_as_windows
from model_training import CNN, train, test
import matplotlib.pyplot as plt


organ_dict = {
    1247: "trachea",
    1302: "right lung",
    1326: "left lung",
    170: "pancreas",
    187: "gallbladder",
    237: "urinary bladder",
    2473: "sternum",
    29193: "first lumbar vertebra",
    29662: "right kidney",
    29663: "left kidney",
    30324: "right adrenal gland",
    30325: "left adrenal gland",
    32248: "right psoas major",
    32249: "left psoas major",
    40357: "muscle body of right rectus abdominis",
    40358: "muscle body of left rectus abdominis",
    480: "aorta",
    58: "liver",
    7578: "thyroid gland",
    86: "spleen",
    0: "background",
    1: "body envelope",
    2: "thorax-abdomen"
}
label_map = {0: 0, 1: 1, 2: 2, 3: 58, 4: 86, 5: 170, 6: 187, 7: 237, 8: 480,
            9: 1247, 10: 1302, 11: 1326, 12: 2473, 13: 7578, 14: 29193, 15: 29662,
            16: 29663, 17: 30324, 18: 30325, 19: 32248, 20: 32249, 21: 40357, 22: 40358}
all_files = sorted(glob.glob("CTce_ThAb_b33x33_n1000_8bit/*.csv"))
all_files_basename = [os.path.splitext(os.path.basename(path))[0] for path in all_files]
kf = KFold(n_splits=4, shuffle=True, random_state=42)
data_list = torch.load("data_all.path")
batch_size = 128
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
def normalize_to_255(volume):
    volume = np.clip(volume, -1023, 2976)
    volume = (volume + 1023) / (2976 + 1023) * 255
    return volume.astype(np.uint8)
def voxel_to_ras(voxel_coord, affine):
    """
    Chuyển từ voxel (i, j, k) sang RAS+ bằng affine matrix.
    """
    voxel_coord = np.append(voxel_coord, 1)  # Thêm phần đồng nhất
    ras_coord = affine @ voxel_coord
    return ras_coord[:3]
def coordinator_reader(basename, organ_id,label_map):
    filepath = f"centers/{basename}_{label_map[organ_id]}_center.csv"
    try:
        data = np.loadtxt(filepath, delimiter=",")  # shape: (n, 6)
        last_three = data[ 3:]  # lấy 3 cột cuối
        return last_three
    except FileNotFoundError:
        return np.nan
def traiter_image_3D(image_file):
    data = []
    # Cấu hình kích thước patch và stride
    patch_size = 33  # nên là số lẻ
    stride = 16
    half_patch = patch_size // 2
    # for file in image_files:
    basename = os.path.splitext(os.path.basename(image_file))[0]
    img = nib.load(image_file)
    
    volume = img.get_fdata()
    # Chuẩn hóa dữ liệu CT về khoảng [0, 255]
    volume_norm = normalize_to_255(volume)
    # Duyệt qua từng lát cắt axial (theo trục Z)
    for z in range(volume_norm.shape[2]):
        slice_2d = volume_norm[:, :, z]

        # Cắt các patches từ lát cắt hiện tại
        if slice_2d.shape[0] < patch_size or slice_2d.shape[1] < patch_size:
            continue  # Bỏ qua nếu lát cắt quá nhỏ

        slice_patches = view_as_windows(slice_2d, (patch_size, patch_size), step=stride)
        h, w, _, _ = slice_patches.shape

        for i in range(h):
            for j in range(w):
                patch = slice_patches[i, j]
                imagette = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
                x = i * stride + half_patch
                y = j * stride + half_patch
                coordinate = torch.tensor([x, y, z], dtype=torch.float32)
                data.append((imagette, coordinate, basename))  # Lưu patch và tọa độ trung tâm voxel
    print(f"finished {basename}") 
    return img,data   
def valid(val_data,model,device):
    model.eval()
    num_organs = 23  # hoặc bao nhiêu lớp (organs) bạn có
    numerators = [np.zeros(3) for _ in range(num_organs)]
    denominators = [0.0 for _ in range(num_organs)]
    dataloader = DataLoader(val_data,batch_size=128, shuffle=False)
    threshold = 0.9999
    with torch.no_grad():
        for imagette, coordinate, basename in dataloader:
            imagette = imagette.to(device)
            pred = model(imagette)               
            pred = torch.softmax(pred, dim=1)  # Áp dụng softmax trên chiều 1 (cho mỗi organ)
            # pred_numpy = pred.cpu().numpy()  # nếu đang ở GPU, check xem đã được softmax chưa, trả về [P^0, P^1, ..., P^22]

            for organ_id in range(num_organs):
                # p: shape (B,)
                p = pred[:, organ_id]
                coordinate = coordinate.to(p.device)

                mask = (p <= threshold)  # (B,) — boolean mask
                    
                p_masked = p[mask]# (B',)
                
                coord_masked = coordinate[mask]  # (B', 3)
                if mask.sum() == 0:
                    continue
                # weighted_coords: shape (B, 3)
                weighted_coords = coord_masked * p_masked.unsqueeze(1)
                numerators[organ_id] += weighted_coords.sum(dim=0).cpu().numpy()
                denominators[organ_id] += p_masked.sum().item() 
    esperances = []
    for i in range(num_organs):
        if denominators[i] > 0:
            esp = numerators[i] / denominators[i]
        else:
            esp = np.nan
        esperances.append(esp)
    return esperances

def compute_errors(esperance, ground_truth):
    errors = []
    for pred, true in zip(esperance, ground_truth):
        if isinstance(pred, float):  # Trường hợp nếu dự đoán là NaN
            errors.append(np.nan)
            continue
        error = np.linalg.norm(np.array(pred) - np.array(true))  # Khoảng cách Euclid
        errors.append(error)
    return errors   

def predict_point_from_volume(image_path, x, y, z, model, device):
    patch_size = 33
    half_patch = patch_size // 2
    # Load ảnh 3D
    img = nib.load(image_path)
    volume = img.get_fdata()
    volume_norm = normalize_to_255(volume)
    # Cắt patch 2D tại lát z
    patch_2d = volume_norm[x - half_patch : x + half_patch + 1,
                           y - half_patch : y + half_patch + 1,
                           z]
    # Đưa vào model
    imagette = torch.tensor(patch_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(imagette)
        print(pred.cpu().numpy())
        prob = torch.softmax(pred, dim=1).squeeze().cpu().numpy()
    return prob  # vector xác suất cho từng organ

model = CNN().to(device)
model.load_state_dict(torch.load("modelCNNnew.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=1.0082099500553373e-6)
if __name__ == "__main__":
    # K-Fold Training-Dectecting
    error_moyenne_list =[]
    ecart_type_moyenne_list = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files_basename)):
        
        train_files = [all_files_basename[i] for i in train_idx]
        train_data = [item[:2] for item in data_list if item[2] in train_files]
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_files = [all_files_basename[i] for i in val_idx]
        errors = []
        print(f"Fold {fold+1}: train={len(train_data)}")
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer,device)
        for image in val_files:
                image_zip = f"CTce_ThAb/{image}.nii.gz"
                img, val_data = traiter_image_3D(image_zip)
                affine = img.affine
                esperance = valid(val_data,model,device)
                esperance_RAS = [voxel_to_ras(voxel, affine) for voxel in esperance]
                print("--------")
                centre = []                
                for organ_id in range(3,23):
                    centre.append(coordinator_reader(image,organ_id,label_map))                                       
                error = compute_errors(esperance_RAS[3:], centre)
                errors.append(error)
        # average_esperance = [np.mean(arrays, axis=0) for arrays in zip(*esperance_all)]
        mean_error = np.nanmean(errors, axis=0)
        error_moyenne_list.append(mean_error)
        print(f"Moyenne error : {mean_error}")
        ecart_type = np.nanstd(errors, axis=0)
        ecart_type_moyenne_list.append(ecart_type)
        print(f"Ecart-Type:{ecart_type}")
    error_definitive= np.nanmean(error_moyenne_list, axis=0)
    ecart_type_definitive=np.nanmean(ecart_type_moyenne_list, axis=0)
    resultat = {}
    print("Distance error :")
    for organ_id in range(3,23): 
        print(f"{organ_dict[label_map[organ_id]]} : {error_definitive[organ_id-3]:.2f} ± {ecart_type_definitive[organ_id-3]:.2f}")

print("Done!")



# torch.save(model.state_dict(), "modelCNNnew.pth")
# print("Saved PyTorch Model State to modelCNNnew.pth")



