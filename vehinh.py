import matplotlib.pyplot as plt
import nibabel as nib

# Load ảnh NIfTI
epi_img = nib.load("CTce_ThAb/10000100_1_CTce_ThAb.nii.gz")
epi_img_data = epi_img.get_fdata()

# Lấy lát cắt tại y = 239
slice_1 = epi_img_data[:, :, 152]

def show_slices_with_points(slice_2d, points):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(slice_2d.T, cmap="gray", origin="lower")

    colors = ['red', 'cyan', 'magenta', 'orange', 'lime']

    # Sắp xếp points theo z tăng dần
    points_sorted = sorted(points, key=lambda p: p[1])

    for i, (x, z, label) in enumerate(points_sorted):
        color = colors[i % len(colors)]

        ax.scatter(x, z, c=color, s=50)

        label_x = slice_2d.shape[0] + 10
        # Đặt label cách nhau 20 pixel theo thứ tự z
        label_z = points_sorted[0][1] + i * 20

        ax.annotate(
            label,
            xy=(x, z),
            xytext=(label_x, label_z),
            textcoords='data',
            arrowprops=dict(arrowstyle="->", color=color),
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec=color, alpha=0.6)
        )

    ax.set_xlim(0, slice_2d.shape[0] + 50)
    ax.set_title("Prédiction de la foie")
    plt.tight_layout()
    plt.show()

points = [
    (313, 227, "vrai centre de la foie"),
    (280, 225, "RF"),
    (295, 227, "KNN"),
    (290, 210, "CNN"),
    (275, 232, "SVM")
]

show_slices_with_points(slice_1, points)
