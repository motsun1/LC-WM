import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_npz_files(directory_pattern, label_filter=None, label_key='y_class', 
                  allow_pickle=True, max_files=None, batch_size=None):
    """
    directory_patternに一致する.npzファイルを一括で読み込み、必要な潜在表現を抽出する
    
    Args:
        directory_pattern: glob形式のパターン (例: "d2_wm_closed/*/*.npz")
        label_filter: 特定のラベルに対応するデータだけを抽出する場合に指定 (例: 0 or 1)
        label_key: ラベルの情報が格納されているキー (例: 'y_class', 'label')
        allow_pickle: numpyのピクルオブジェクトを許可するかどうか
        max_files: 処理する最大ファイル数（None=全て）
        batch_size: バッチ処理する場合のバッチサイズ（None=一括処理）
    
    Returns:
        dict: キーが.npzファイル内の変数名、値がそれらの配列を連結したものの辞書
    """
    print(f"Loading .npz files from: {directory_pattern}")
    file_paths = sorted(glob(directory_pattern))
    
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {directory_pattern}")
    
    if max_files:
        file_paths = file_paths[:max_files]
    
    print(f"Found {len(file_paths)} files. Processing {'in batches' if batch_size else 'all at once'}.")
    
    # 最初のファイルからデータ構造を確認
    sample_data = np.load(file_paths[0], allow_pickle=allow_pickle)
    print(f"Keys in .npz files: {sample_data.files}")
    
    # 結果を格納する辞書を初期化
    all_data = {key: [] for key in sample_data.files}
    
    # メタデータを保存用に抽出
    metadata = []
    
    # バッチ処理または全ファイル一括処理
    if batch_size:
        # バッチ処理の実装
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            process_batch(batch_files, all_data, metadata, label_filter, label_key, allow_pickle)
            print(f"Processed batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
    else:
        # 全ファイルをロード
        for file_path in tqdm(file_paths):
            try:
                data = np.load(file_path, allow_pickle=allow_pickle)
                
                # ラベルでフィルタリング（label情報がある場合）
                if label_filter is not None and label_key in data:
                    labels = data[label_key]
                    mask = labels == label_filter
                    
                    # マスクが全てFalseなら次のファイルへ
                    if not np.any(mask):
                        continue
                        
                    # ラベルでフィルタリングしたデータを追加
                    for key in all_data.keys():
                        if key in data:
                            filtered_data = data[key][mask] if data[key].shape[0] == labels.shape[0] else data[key]
                            all_data[key].append(filtered_data)
                else:
                    # フィルタリングなしで全データを追加
                    for key in all_data.keys():
                        if key in data:
                            all_data[key].append(data[key])
                
                # メタデータに追加
                metadata.append(os.path.basename(file_path))
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    # 各キーごとにnumpy配列を結合
    for key in all_data.keys():
        if all_data[key]:  # リストが空でない場合
            try:
                all_data[key] = np.concatenate(all_data[key], axis=0)
                print(f"{key}: Shape = {all_data[key].shape}, Type = {all_data[key].dtype}")
            except Exception as e:
                print(f"Warning: Could not concatenate {key}: {e}")
                all_data[key] = all_data[key]  # リストのまま保持
    
    all_data['_metadata'] = metadata
    return all_data

def process_batch(batch_files, all_data, metadata, label_filter, label_key, allow_pickle):
    """バッチ処理用のヘルパー関数"""
    for file_path in batch_files:
        try:
            data = np.load(file_path, allow_pickle=allow_pickle)
            
            # ラベルでフィルタリング
            if label_filter is not None and label_key in data:
                labels = data[label_key]
                mask = labels == label_filter
                
                if not np.any(mask):
                    continue
                    
                for key in all_data.keys():
                    if key in data:
                        filtered_data = data[key][mask] if data[key].shape[0] == labels.shape[0] else data[key]
                        all_data[key].append(filtered_data)
            else:
                for key in all_data.keys():
                    if key in data:
                        all_data[key].append(data[key])
            
            metadata.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def extract_features(data_dict, feature_key='features'):
    """潜在表現（特徴ベクトル）を抽出する"""
    if feature_key not in data_dict:
        available_keys = list(data_dict.keys())
        raise KeyError(f"Key '{feature_key}' not found in data. Available keys: {available_keys}")
    
    features = data_dict[feature_key]
    print(f"Extracted features shape: {features.shape}, dtype: {features.dtype}")
    return features

def visualize_reconstructions(data_dict, num_samples=5, 
                             orig_key='obs', recon_key='image_rec',
                             is_hwc=True):
    """元画像と再構成画像を可視化する
    
    Args:
        data_dict: データ辞書
        num_samples: 表示するサンプル数
        orig_key: 元画像のキー名（'obs', 'image'など）
        recon_key: 再構成画像のキー名（'image_rec', 'recon'など）
        is_hwc: 画像がHWC形式の場合True、CHW形式の場合False
    """
    if orig_key not in data_dict or recon_key not in data_dict:
        print(f"Image keys not found. Available keys: {list(data_dict.keys())}")
        return
    
    orig_images = data_dict[orig_key][:num_samples]
    recon_images = data_dict[recon_key][:num_samples]
    
    # 形状チェック
    print(f"Original images shape: {orig_images.shape}")
    print(f"Reconstructed images shape: {recon_images.shape}")
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))
    
    for i in range(num_samples):
        # 元画像
        img_orig = orig_images[i]
        if not is_hwc and img_orig.shape[0] == 3:  # CHW形式の場合
            img_orig = np.transpose(img_orig, (1, 2, 0))
        
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # 再構成画像
        img_recon = recon_images[i]
        if not is_hwc and img_recon.shape[0] == 3:  # CHW形式の場合
            img_recon = np.transpose(img_recon, (1, 2, 0))
        
        axes[1, i].imshow(img_recon)
        axes[1, i].set_title(f'Reconstruction {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_samples.png')
    plt.show()

if __name__ == "__main__":
    # コマンドライン引数のパース
    import argparse
    parser = argparse.ArgumentParser(description='.npzファイルから特徴量を抽出し可視化する')
    parser.add_argument('--pattern', type=str, default="d2_wm_closed/*/*.npz",
                      help='glob形式のパターン')
    parser.add_argument('--label_filter', type=int, default=None,
                      help='特定のラベルだけを抽出する場合に指定')
    parser.add_argument('--label_key', type=str, default='y_class',
                      help='ラベルのキー名')
    parser.add_argument('--feature_key', type=str, default='features',
                      help='特徴量のキー名')
    parser.add_argument('--orig_key', type=str, default='obs',
                      help='元画像のキー名')
    parser.add_argument('--recon_key', type=str, default='image_rec',
                      help='再構成画像のキー名')
    parser.add_argument('--max_files', type=int, default=None,
                      help='処理する最大ファイル数')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='バッチサイズ')
    args = parser.parse_args()
    
    # データロード
    data = load_npz_files(args.pattern, 
                         label_filter=args.label_filter,
                         label_key=args.label_key,
                         max_files=args.max_files,
                         batch_size=args.batch_size)
    
    # 特徴が存在すれば抽出
    if args.feature_key in data:
        features = extract_features(data, feature_key=args.feature_key)
        print(f"Extracted features shape: {features.shape}")
    else:
        print(f"Feature key '{args.feature_key}' not found. Available keys: {list(data.keys())}")
    
    # 再構成の可視化（元画像と再構成画像の両方が存在する場合）
    if args.orig_key in data and args.recon_key in data:
        visualize_reconstructions(data, 
                               orig_key=args.orig_key, 
                               recon_key=args.recon_key,
                               is_hwc=(len(data[args.orig_key].shape) == 4 and data[args.orig_key].shape[-1] == 3))
    else:
        print("Reconstruction visualization skipped: Missing original or reconstructed images")