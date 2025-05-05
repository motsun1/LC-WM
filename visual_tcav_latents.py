import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from npz_extraction import load_npz_files, extract_features

class VisualTCAVLatents:
    def __init__(self, model_path=None, device='cpu'):
        """
        Dreamer V2の潜在空間に対して Visual-TCAV を適用するクラス
        
        Args:
            model_path: 必要であれば、ここにDreamerのモデルパスを指定
            device: 計算に使用するデバイス
        """
        self.device = device
        self.concept_vectors = {}
        self.pca_components = {}
        
        # モデルを読み込む場合はここで処理
        self.model = None
        if model_path:
            print(f"Loading model from {model_path}")
            # モデル読み込み実装（必要に応じて）
    
    def load_concept_features(self, concept_dir, concept_name):
        """概念データセットから特徴を抽出する"""
        print(f"Loading concept: {concept_name} from {concept_dir}")
        image_paths = glob(os.path.join(concept_dir, "*.jpg"))
        
        if not image_paths:
            raise ValueError(f"No images found in {concept_dir}")
        
        print(f"Found {len(image_paths)} images for concept {concept_name}")
        
        # ここでは、すでに抽出された特徴を使用する想定
        # 実際のプロジェクトでは、feature extractorを使う必要がある場合もある
        
        return np.random.randn(len(image_paths), 512)  # ダミーデータ（実際に置き換え必要）
    
    def extract_concept_direction(self, concept_a_features, concept_b_features, method='svm'):
        """
        2つの概念間の方向ベクトルを抽出する
        
        Args:
            concept_a_features: 概念Aの特徴ベクトル配列
            concept_b_features: 概念Bの特徴ベクトル配列
            method: 'svm' または 'pca'
            
        Returns:
            direction_vector: 概念軸を表す方向ベクトル
        """
        print(f"Extracting concept direction using {method}")
        
        if method == 'svm':
            # 概念AとBのラベルを準備
            X = np.vstack([concept_a_features, concept_b_features])
            y = np.hstack([np.ones(len(concept_a_features)), 
                          np.zeros(len(concept_b_features))])
            
            # SVMで分類境界を学習
            svm = LinearSVC(C=1.0, class_weight='balanced')
            svm.fit(X, y)
            
            # 分類境界の法線ベクトルを方向ベクトルとして使用
            direction_vector = svm.coef_[0]
            
            # 正規化
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            
            return direction_vector
            
        elif method == 'pca':
            # 両方の概念データを結合
            all_features = np.vstack([concept_a_features, concept_b_features])
            
            # PCAで主成分を抽出
            pca = PCA(n_components=10)
            pca.fit(all_features)
            
            # 第一主成分を方向ベクトルとして使用
            direction_vector = pca.components_[0]
            
            # 正規化
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            
            # 方向が概念Aの方を向くように調整
            mean_a = np.mean(concept_a_features, axis=0)
            mean_b = np.mean(concept_b_features, axis=0)
            concept_diff = mean_a - mean_b
            
            # 内積が正なら同じ方向、負なら逆方向
            if np.dot(direction_vector, concept_diff) < 0:
                direction_vector = -direction_vector
                
            self.pca_components[f"{concept_a_name}_{concept_b_name}"] = pca
            
            return direction_vector
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def generate_counterfactual_latents(self, z, direction_vector, alpha_values):
        """
        指定された方向ベクトルに沿って反実潜在表現を生成する
        
        Args:
            z: 元の潜在表現 (N, dim)
            direction_vector: 方向ベクトル (dim,)
            alpha_values: 移動量のリスト [alpha1, alpha2, ...]
            
        Returns:
            z_cf_list: 反実潜在表現のリスト [z_cf1, z_cf2, ...]
        """
        z_cf_list = []
        
        for alpha in alpha_values:
            # z_cf = z + alpha * v
            z_cf = z + alpha * direction_vector
            z_cf_list.append(z_cf)
            
        return z_cf_list

    def visualize_latent_space(self, features, labels, direction_vector, title="Latent Space Projection"):
        """
        潜在空間を可視化する（方向ベクトルと一緒に）
        
        Args:
            features: 潜在表現 (N, dim)
            labels: 各サンプルのラベル (N,)
            direction_vector: 方向ベクトル (dim,)
            title: グラフのタイトル
        """
        # PCAで2次元に削減
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # 方向ベクトルも同じPCAで変換
        direction_2d = pca.transform(np.expand_dims(direction_vector, 0))[0]
        
        # プロット
        plt.figure(figsize=(10, 8))
        
        # 各クラスを異なる色でプロット
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=f"Class {label}", alpha=0.6)
        
        # 方向ベクトルを矢印で表示
        vector_scale = 20  # スケーリング係数
        plt.arrow(0, 0, direction_2d[0] * vector_scale, direction_2d[1] * vector_scale,
                 color='red', width=0.5, head_width=5, head_length=7, 
                 length_includes_head=True, label='Concept Direction')
        
        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.show()
        
    def decode_from_latent(self, model, z_cf_list, batch_size=16):
        """
        反実潜在表現からDreamerのデコーダを使用して画像を再構成する
        この関数は、PyDreamerのmodel.decode_from_latentと連携する必要がある
        
        Args:
            model: Dreamerモデル（デコード機能付き）
            z_cf_list: 反実潜在表現のリスト
            batch_size: バッチサイズ
            
        Returns:
            images_list: デコードされた画像のリスト
        """
        images_list = []
        
        print("Decoding counterfactual latents...")
        
        # バッチ処理
        for z_cf in z_cf_list:
            num_samples = len(z_cf)
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            batch_images = []
            for i in tqdm(range(num_batches)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                z_batch = torch.tensor(z_cf[start_idx:end_idx], dtype=torch.float32, device=self.device)
                
                # モデルのデコード機能を呼び出す
                # ここは実際のPyDreamerのAPIに合わせて調整が必要
                with torch.no_grad():
                    if self.model:
                        # モデルがロードされている場合
                        decoded = self.model.decoder.image.forward(z_batch).cpu().numpy()
                    else:
                        # モデルがない場合はダミーデータを生成
                        decoded = np.random.rand(len(z_batch), 3, 64, 64)
                
                batch_images.append(decoded)
            
            # バッチをまとめる
            images = np.concatenate(batch_images, axis=0)
            images_list.append(images)
        
        return images_list
    
    def visualize_counterfactual_grid(self, original_images, cf_images_list, alpha_values, 
                                     num_samples=5, save_path="counterfactual_grid.png"):
        """
        元画像と生成された反実画像をグリッド形式で可視化する
        
        Args:
            original_images: 元画像 (N, C, H, W)
            cf_images_list: 反実画像のリスト [imgs1, imgs2, ...], 各imgsは (N, C, H, W)
            alpha_values: 各反実画像に対応するalpha値
            num_samples: 表示するサンプル数
            save_path: 保存先のパス
        """
        num_alphas = len(alpha_values)
        num_rows = num_samples
        num_cols = num_alphas + 1  # +1 for original
        
        plt.figure(figsize=(num_cols * 3, num_rows * 3))
        
        # サンプルをランダムに選択
        indices = np.random.choice(len(original_images), num_samples, replace=False)
        
        for row, idx in enumerate(indices):
            # 元画像
            plt.subplot(num_rows, num_cols, row * num_cols + 1)
            plt.imshow(np.transpose(original_images[idx], (1, 2, 0)))
            if row == 0:
                plt.title("Original")
            plt.axis('off')
            
            # 反実画像
            for col, (cf_images, alpha) in enumerate(zip(cf_images_list, alpha_values)):
                plt.subplot(num_rows, num_cols, row * num_cols + col + 2)
                plt.imshow(np.transpose(cf_images[idx], (1, 2, 0)))
                if row == 0:
                    plt.title(f"α = {alpha:.2f}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Dreamer V2潜在空間に対してVisual-TCAVを適用する')
    parser.add_argument('--npz_pattern', type=str, default="d2_wm_closed/*/*.npz", 
                       help='NPZファイルのglobパターン')
    parser.add_argument('--use_dummy_data', action='store_true',
                       help='NPZファイルがない場合にダミーデータを使用する')
    parser.add_argument('--concept_a', type=str, default="background_land",
                       help='概念Aの名前')
    parser.add_argument('--concept_b', type=str, default="background_water",
                       help='概念Bの名前')
    parser.add_argument('--method', type=str, default="svm", choices=['svm', 'pca'],
                       help='概念軸抽出の方法')
    args = parser.parse_args()
    
    # VisualTCAVLatentsのインスタンス化
    tcav = VisualTCAVLatents()
    
    try:
        # 1. .npzファイルからデータ読み込み
        print(f"Loading .npz files from pattern: {args.npz_pattern}")
        data = load_npz_files(args.npz_pattern)
        features = extract_features(data, feature_key='features')
        
        # ラベル情報
        if 'label' in data:
            labels = data['label']
        else:
            # ダミーラベル
            labels = np.zeros(len(features))
        
        print(f"Successfully loaded {len(features)} feature vectors of dimension {features.shape[1]}")
        
    except Exception as e:
        if args.use_dummy_data:
            print(f"Error loading .npz files: {e}")
            print("Using dummy data instead...")
            # ダミーデータ作成
            features = np.random.randn(1000, 512)  # 1000サンプル x 512次元
            labels = np.random.randint(0, 2, size=1000)  # バイナリラベル
            data = {
                'image': np.random.rand(1000, 3, 64, 64),  # ダミー画像
                'features': features
            }
        else:
            raise e
    
    # 2. 概念ベクトルの構築
    # 実データの代わりにダミーデータ
    concept_a_features = np.random.randn(100, features.shape[1])
    concept_b_features = np.random.randn(100, features.shape[1])
    
    # 3. SVM方向ベクトル取得
    direction_vector = tcav.extract_concept_direction(
        concept_a_features, concept_b_features, method=args.method)
    
    # 潜在空間の可視化
    tcav.visualize_latent_space(
        features, labels, direction_vector, 
        title=f"Latent Space with {args.concept_a} vs {args.concept_b}")
    
    # 4. 反実潜在表現の生成
    alpha_values = [-3.0, -1.5, 0.0, 1.5, 3.0]
    z_cf_list = tcav.generate_counterfactual_latents(
        features, direction_vector, alpha_values)
    
    # 5. 潜在表現からの画像生成
    # 仮のデコード処理（実際にはDreamerのモデルを使用）
    cf_images_list = tcav.decode_from_latent(None, z_cf_list)
    
    # 6. 反実画像の可視化
    original_images = data['image'] if 'image' in data else np.random.rand(len(features), 3, 64, 64)
    tcav.visualize_counterfactual_grid(
        original_images, cf_images_list, alpha_values)