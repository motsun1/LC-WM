import pandas as pd
import os
import shutil

# パス設定
metadata_path = 'datasets/waterbirds/waterbird_complete95_forest2water2/metadata.csv'
image_root = 'datasets/waterbirds/waterbird_complete95_forest2water2'
save_dir = 'concepts'
n_samples = 50

# ディレクトリ作成
os.makedirs(f'{save_dir}/background_land', exist_ok=True)
os.makedirs(f'{save_dir}/background_water', exist_ok=True)
os.makedirs(f'{save_dir}/random_concept', exist_ok=True)

# メタデータ読み込み
meta = pd.read_csv(metadata_path)

# === 背景概念セットを作成 ===
used_images = set()

for place_val, concept_name in zip([0, 1], ['background_land', 'background_water']):
    subset = meta[meta['place'] == place_val].sample(n_samples, random_state=place_val)
    for _, row in subset.iterrows():
        img_name = os.path.basename(row['img_filename'])
        src = os.path.join(image_root, row['img_filename'])
        dst = os.path.join(save_dir, concept_name, img_name)
        shutil.copyfile(src, dst)
        used_images.add(img_name)  # 使用済み画像として記録

# === ランダム画像（重複除外） ===
remaining = meta[~meta['img_filename'].apply(lambda x: os.path.basename(x) in used_images)]
random_subset = remaining.sample(n_samples, random_state=999)

for _, row in random_subset.iterrows():
    img_name = os.path.basename(row['img_filename'])
    src = os.path.join(image_root, row['img_filename'])
    dst = os.path.join(save_dir, 'random_concept', img_name)
    shutil.copyfile(src, dst)
