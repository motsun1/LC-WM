# PyDreamerを使ったVisual-TCAV実験の引き継ぎ資料

## これまでの作業内容

1. **PyDreamerモデルの学習環境準備**
   - PyDreamerフレームワークを使用してDreamerV2モデルを学習するためのセットアップ
   - Waterbirdsデータセットを用いた画像分類タスク用の設定ファイル作成
   - データセットをPyDreamerの入力形式（エピソードデータ）に変換するスクリプト作成

2. **Visual-TCAV実装**
   - 潜在空間の概念軸抽出と反実ロールアウト生成のための実装
   - `visual_tcav_latents.py`の実装（概念ベクトル構築、SVM/PCAによる方向ベクトル取得、反実潜在表現の生成）
   - 学習済みモデルから潜在表現を抽出するための`npz_extraction.py`スクリプト実装

## 現在進行中の作業

1. **PyDreamerモデルの学習実行**
   - GPUを使用した`waterbirds`設定でのDreamerV2モデル学習
   - コマンド: `python launch.py --configs defaults waterbirds`
   - 設定ファイルの微調整（generator_workersなどのパラメータ調整）

2. **オフラインデータでの学習**
   - 事前に変換したWaterbirdsエピソードデータを使用
   - データパス: `../datasets/waterbirds_episodes/train` と `../datasets/waterbirds_episodes/val`

## これからやるべきこと

1. **PyDreamerモデル学習の完了**
   - 学習が完了するまで待機（GPUを使用）
   - 学習完了後、`d2_wm_closed`ディレクトリに`.npz`ファイルが生成される

2. **Visual-TCAVの適用**
   - 学習完了後、生成された`.npz`ファイルを使用して概念軸の抽出
   - コマンド: `python visual_tcav_latents.py --npz_pattern "d2_wm_closed/*/*.npz"`
   - 背景概念（land/water）に基づく潜在空間の軸を可視化

3. **反実ロールアウトの生成と可視化**
   - 抽出した概念軸に沿って潜在空間で移動させた反実潜在表現を生成
   - DreamerV2のデコーダーを使用して潜在表現から画像を再構成
   - 生成された反実画像をグリッド形式で可視化

## 注意点・課題

1. **設定パラメータの管理**
   - PyDreamerの設定ファイルは`--configs defaults waterbirds`のように複数の設定を組み合わせる
   - 設定の継承関係に注意（`defaults`セクションが基本設定）

2. **GPUメモリ使用量**
   - バッチサイズやモデルサイズを調整して、GPUメモリに収まるようにする

3. **潜在表現の抽出と解釈**
   - 画像分類タスクに対するDreamerV2の適用は標準的でないため、潜在表現の解釈に注意

## 参考資料・コマンド

```bash
# PyDreamerモデルの学習実行
cd /home/mitsui/Desktop/LC-WM/pydreamer
python launch.py --configs defaults waterbirds

# Visual-TCAVの適用
cd /home/mitsui/Desktop/LC-WM
python visual_tcav_latents.py --npz_pattern "d2_wm_closed/*/*.npz"
```