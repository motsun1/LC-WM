import torch
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from captum.concept import TCAV, Concept, ConceptInterpreter
from captum.concept._utils.data_iterator import dataset_to_dataloader

import os
from PIL import Image
from glob import glob
from torch.utils.data import Dataset

# device設定 - すべてCPUで実行するよう変更
device = torch.device("cpu")
print(f"Using device: {device} (すべての処理をCPUで実行します)")

# モデル準備（評価モード）
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_waterbirds.pth", weights_only=True, map_location=device))  # CPUにマップ
model = model.to(device).eval()

# 対象クラス（例: 0 = landbird）
target_class = 0

# === カスタム画像Dataset ===
class ConceptImgDataset(Dataset):
    def __init__(self, image_paths, transform, device):
        self.image_paths = image_paths
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = default_loader(self.image_paths[idx])
        # デバイスへの移動はここではなく、データローダーから取得後に行う
        return self.transform(img)

# === Transform定義 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === 各概念フォルダから画像読込 ===
def load_concept_images(concept_dir):
    return ConceptImgDataset(
        image_paths=glob(os.path.join(concept_dir, "*.jpg")),
        transform=transform,
        device=device
    )

concepts_root = "datasets/waterbirds/concepts"

# 概念データセットを準備
concept_datasets = [
    load_concept_images(f"{concepts_root}/background_land"),
    load_concept_images(f"{concepts_root}/background_water"),
    load_concept_images(f"{concepts_root}/random_concept")
]

# 概念データを適切なフォーマットで準備する関数
def prepare_concept_data(dataset):
    """データセットからデータローダーを作成"""
    # 通常のデータローダーを作成
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True
    )

# 概念データを準備
print("Preparing concept data...")

try:
    # 各概念のデータローダーを準備
    land_loader = prepare_concept_data(concept_datasets[0])
    water_loader = prepare_concept_data(concept_datasets[1])
    random_loader = prepare_concept_data(concept_datasets[2])
    
    print(f"Land concept dataloader created with {len(concept_datasets[0])} samples")
    print(f"Water concept dataloader created with {len(concept_datasets[1])} samples")
    print(f"Random concept dataloader created with {len(concept_datasets[2])} samples")
    
    # 概念の定義
    # Conceptオブジェクトには、データローダーをdata_iterとして渡す
    land_concept = Concept(
        id=0,
        name="background_land", 
        data_iter=land_loader
    )
    
    water_concept = Concept(
        id=1,
        name="background_water", 
        data_iter=water_loader
    )
    
    random_concept = Concept(
        id=2,
        name="random", 
        data_iter=random_loader
    )
    
    # 概念のリスト
    concepts = [land_concept, water_concept, random_concept]
    
except Exception as e:
    print(f"Error preparing concept data: {e}")
    raise

# === テスト画像群（分類対象） ===
# ※ここではtest splitの画像を使う（正解ラベル: target_class）
from train_resnet_waterbirds import WaterbirdsDataset  # 自前のDatasetクラスを使う

test_set = WaterbirdsDataset(
    root_dir='datasets/waterbirds/waterbird_complete95_forest2water2',
    metadata_path='datasets/waterbirds/waterbird_complete95_forest2water2/metadata.csv',
    split=2,
    transform=transform
)

# target_classに該当するデータだけをフィルタリングして取得
filtered_dataset = [(img, label) for img, label in test_set if label == target_class]
if not filtered_dataset:
    raise ValueError(f"No samples found for target class {target_class}")

print(f"Found {len(filtered_dataset)} samples for target class {target_class}")

# バッチサイズを指定してデータローダーを作成
batch_size = 16
target_loader = torch.utils.data.DataLoader(
    [img for img, _ in filtered_dataset],  # 画像のみを含むデータセット
    batch_size=batch_size
)

# TCAVの入力用に適切な形式で提供
# バッチ次元を保持したままデータをデバイスに移動
target_batches = []
for batch in target_loader:
    # バッチをデバイスに移動
    target_batches.append(batch.to(device))

# 念のためにターゲットが適切なデバイスにあるか確認
print(f"Target batch device: {target_batches[0].device}")
print(f"Model device: {next(model.parameters()).device}")

# === TCAV 実行 ===
# レイヤー名を正確に指定（ResNet50のレイヤー名）
layer_name = "layer4"

# キャッシュ関連のディレクトリを一度削除（必要に応じて）
import shutil
if os.path.exists("cav"):
    print("Removing cached activations and CAVs...")
    shutil.rmtree("cav")
    os.makedirs("cav/av/default_model_id", exist_ok=True)
    os.makedirs("cav/default_model_id", exist_ok=True)

# CaptumのTCAVでは文字列としてレイヤー名を指定する必要がある
interpreter = TCAV(
    model=model,
    layers=[layer_name],   # レイヤー名の文字列のリスト（layer属性は使用しない）
    save_path="./cav/",    # 明示的にキャッシュパスを指定
    attribute_to_layer_input=False  # 出力側の属性を使用
)

print("Starting TCAV interpretation...")

# 実験セットを定義
# 1. background_land vs. random (背景が陸地である概念の重要性)
# 2. background_water vs. random (背景が水である概念の重要性)
# 3. background_land vs. background_water (背景の種類による対照実験)
experimental_sets = [
    [concepts[0], concepts[2]],  # land vs. random
    [concepts[1], concepts[2]],  # water vs. random
    [concepts[0], concepts[1]]   # land vs. water
]

# 実験セットの説明を出力
print("Experimental sets:")
print("1. background_land vs. random")
print("2. background_water vs. random")
print("3. background_land vs. background_water")

# Captum v0.8.0のTCAV APIに合わせて結果処理を修正
tcav_scores = interpreter.interpret(
    inputs=target_batches[0],  # 最初のバッチだけを使用
    experimental_sets=experimental_sets,
    target=target_class
)

# === 結果出力 ===
print("\nTCAV Scores:")
print("============")

# 返却値のフォーマット例：
# {
#   '0-2': {  # 実験セット (land vs. random)
#     'layer4': {  # レイヤー名
#       'sign_count': tensor([0.6, 0.4]),  # 正の値の数の割合
#       'magnitude': tensor([0.7, 0.3])    # 正の値の大きさの割合
#     }
#   },
#   '1-2': { ... },  # 実験セット (water vs. random)
#   '0-1': { ... }   # 実験セット (land vs. water)
# }

exp_names = [
    "Land vs. Random",
    "Water vs. Random", 
    "Land vs. Water"
]

# 実験セットごとに結果を処理
for i, (exp_key, exp_results) in enumerate(tcav_scores.items()):
    print(f"\n{exp_names[i]} ({exp_key}):")
    
    # レイヤーごとの結果
    for layer_name, layer_results in exp_results.items():
        print(f"  Layer: {layer_name}")
        
        # sign_count (正の値の割合)
        sign_counts = layer_results["sign_count"]
        print(f"    Sign count: {sign_counts[0].item():.4f} / {sign_counts[1].item():.4f}")
        
        # magnitude (正の値の大きさの割合)
        magnitudes = layer_results["magnitude"]
        print(f"    Magnitude: {magnitudes[0].item():.4f} / {magnitudes[1].item():.4f}")
        
        # 正の値が多いほど、その概念とターゲットクラスの相関が強い
        concept1_idx = exp_key.split('-')[0]
        concept2_idx = exp_key.split('-')[1]
        
        concept1_name = concepts[int(concept1_idx)].name
        concept2_name = concepts[int(concept2_idx)].name
        
        # 最初の概念のスコアが高いほど、その概念がモデルの予測に重要
        if sign_counts[0] > sign_counts[1]:
            print(f"    ✅ Concept '{concept1_name}' is more important for class {target_class}")
        else:
            print(f"    ✅ Concept '{concept2_name}' is more important for class {target_class}")