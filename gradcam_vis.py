import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from PIL import Image
import numpy as np

# === 設定 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "datasets/Barn_Swallow_0016_130678.jpg"  # 可視化したい画像
model_path = "resnet50_waterbirds.pth"
target_class = 0  # 例: landbird
layer_name = "layer4"

# === モデル準備 ===
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device).eval()

# === レイヤー取得 ===
target_layer = dict(model.named_modules())[layer_name]
gradcam = LayerGradCam(model, target_layer)

# === 入力画像前処理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_image = Image.open(image_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# === Grad-CAM算出 & 可視化 ===
original_np = np.array(input_image.resize((224, 224))) / 255.0  # PIL画像を[0,1]のnumpy配列に変換
attributions = gradcam.attribute(input_tensor, target=target_class)
upsampled_attr = LayerAttribution.interpolate(attributions, input_tensor.shape[2:])

# 可視化設定を改善
plt.figure(figsize=(15, 5))

# 1. 元の画像を表示
plt.subplot(1, 3, 1)
plt.imshow(original_np)
plt.title("Original Image")
plt.axis('off')

# 2. Grad-CAMのみを表示
plt.subplot(1, 3, 2)
heatmap = upsampled_attr[0].cpu().detach().numpy().transpose(1, 2, 0)
plt.imshow(heatmap.squeeze(), cmap='jet')
plt.title("Grad-CAM Heatmap")
plt.colorbar()
plt.axis('off')

# 3. オーバーレイ表示（元画像+ヒートマップ）
plt.subplot(1, 3, 3)
# visualize_image_attrの戻り値を処理せず、直接plt内で表示設定
heatmap = upsampled_attr[0].cpu().detach().numpy().squeeze()
plt.imshow(original_np)
plt.imshow(heatmap, cmap='jet', alpha=0.5)  # 透明度0.5でヒートマップを重ねる
plt.title("Overlaid Visualization")
plt.colorbar()
plt.axis('off')

# 結果を保存
plt.tight_layout()
plt.savefig("results/gradcam_vis.png", dpi=300, bbox_inches='tight')
plt.show()  # 明示的に表示
