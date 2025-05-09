import numpy as np
import matplotlib.pyplot as plt

npz_path = "/home/mitsui/Desktop/LC-WM/pydreamer/mlruns/0/55a08ecc50594bca9e999b464ac9881c/artifacts/d2_wm_closed/0000001.npz"
data = np.load(npz_path)

# キー一覧を表示
print(data.files)

# 各キーのデータの形状を確認
for key in data.files:
    print(f"{key}: {data[key].shape}")

# print(data['latent'].shape)
# print(data['latent'][:5])  # 最初の5つを表示

# 再構成画像の確認
plt.imshow(data['image_rec'][0].reshape(224, 224, 3))
plt.title("Reconstructed Image")
plt.show()

# 再構成画像のピクセル値の統計情報
print("Image Rec Pixel Stats:")
print(f"Min: {data['image_rec'].min()}")
print(f"Max: {data['image_rec'].max()}")
print(f"Mean: {data['image_rec'].mean()}")
print(f"Unique Values: {np.unique(data['image_rec'])}")


plt.imshow(data['image_pred'][0].reshape(224, 224, 3))
plt.title("Predicted Image")
plt.show()