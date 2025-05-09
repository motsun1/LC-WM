import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
import pandas as pd
import json
import mlflow
import random

def create_waterbirds_episodes(data_dir, output_dir, split="train", max_episodes=1000):
    """
    Waterbirdsデータセットを、PyDreamerのエピソードフォーマットに変換する
    
    Args:
        data_dir: Waterbirdsデータセットのルートディレクトリ
        output_dir: 出力ディレクトリ
        split: 'train', 'val', 'test'のいずれか
        max_episodes: 最大エピソード数
    """
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # メタデータ読み込み
    metadata_path = os.path.join(data_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    
    # splitでフィルタリング (0:train, 1:val, 2:test)
    split_map = {"train": 0, "val": 1, "test": 2}
    split_idx = split_map[split]
    filtered_data = metadata[metadata['split'] == split_idx]
    
    # 出力変換に使用するトランスフォーム
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # エピソード生成
    episode_idx = 0
    
    # サンプルをバイナリクラスでグループ化
    class0_samples = filtered_data[filtered_data['y'] == 0]  # landbird
    class1_samples = filtered_data[filtered_data['y'] == 1]  # waterbird
    
    # 各クラスから同数のサンプルを選択（バランス調整）
    n_per_class = min(len(class0_samples), len(class1_samples), max_episodes // 2)
    
    # ランダムにサンプル選択
    class0_samples = class0_samples.sample(n_per_class)
    class1_samples = class1_samples.sample(n_per_class)
    
    # 全サンプルを結合してシャッフル
    all_samples = pd.concat([class0_samples, class1_samples]).sample(frac=1)
    
    print(f"Creating {len(all_samples)} episodes for {split} split...")
    
    for idx, row in tqdm(all_samples.iterrows(), total=len(all_samples)):
        # 画像パス
        img_path = os.path.join(data_dir, row['img_filename'])
        
        # クラスとバックグラウンド情報
        y_class = row['y']  # 0: landbird, 1: waterbird
        place = row['place']  # 0: land, 1: water
        
        # 画像読み込み
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).numpy()  # (C, H, W)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        
        # エピソードデータ作成
        # PyDreamerのフォーマットに合わせた疑似エピソードを作成
        # T=1のシングルステップエピソードとしてマッピング
        
        # 画像をCHW→HWC形式に変換
        img_hwc = img_tensor.transpose(1, 2, 0)  # (H, W, C)
        
        # メタデータ
        metadata_dict = {
            "class": int(y_class),         # 0: landbird, 1: waterbird
            "background": int(place),      # 0: land, 1: water
            "img_path": row['img_filename'],
            "split": split
        }
        
        # メタデータを別ファイルに保存するオプション
        metadata_path = os.path.join(output_dir, f"metadata_{episode_idx:06d}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f)
        
        # エピソードデータ - PyDreamerの期待する形式に合わせる
        episode_data = {
            "image": img_hwc[np.newaxis, ...],    # (T, H, W, C) = (1, 224, 224, 3)
            "action": np.zeros((1, 1), dtype=np.float32),  # 学習では使用しない簡略化した形式
            "reward": np.zeros(1, dtype=np.float32),       # 学習では使用しない簡略化した形式
            "reset": np.array([True]),            # エピソード開始フラグ
            "terminal": np.array([True]),         # エピソード終了フラグ
            # メタデータを埋め込む場合（オプション）
            # "metadata": json.dumps(metadata_dict)
        }
        
        # 原データ（教師ラベル）も保存しておく（オプション）
        episode_data["y_class"] = np.array([y_class], dtype=np.int32)
        episode_data["place"] = np.array([place], dtype=np.int32)
        
        # 保存
        episode_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.npz")
        np.savez_compressed(episode_path, **episode_data)
        episode_idx += 1
    
    print(f"Created {episode_idx} episodes in {output_dir}")
    return episode_idx

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Waterbirdsデータセットを PyDreamer用のエピソードに変換')
    parser.add_argument('--data_dir', type=str, default='datasets/waterbirds/waterbird_complete95_forest2water2',
                      help='Waterbirdsデータセットのルートディレクトリ')
    parser.add_argument('--output_dir', type=str, default='datasets/waterbirds/waterbirds_episodes',
                      help='出力ディレクトリ')
    parser.add_argument('--max_episodes', type=int, default=10000,
                      help='最大エピソード数')
    args = parser.parse_args()
    
    # 出力ディレクトリ
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    test_dir = os.path.join(args.output_dir, 'test')
    
    # エピソード作成
    n_train = create_waterbirds_episodes(args.data_dir, train_dir, "train", args.max_episodes)
    n_val = create_waterbirds_episodes(args.data_dir, val_dir, "val", args.max_episodes // 5)
    n_test = create_waterbirds_episodes(args.data_dir, test_dir, "test", args.max_episodes // 5)
    
    print(f"Created {n_train} train, {n_val} validation, and {n_test} test episodes")