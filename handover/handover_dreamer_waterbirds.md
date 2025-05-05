# Dreamer × Waterbirds 研究 ─ 引き継ぎドキュメント

## 0. ゴール概要

* **段階 A ― 最小構成 (batch\_length = 1) で Dreamer Encoder を事前学習し，潜在特徴を用いた画像分類器を構築・評価する。**
* **段階 C ― 概念編集アクション α を導入したシーケンス環境で世界モデルを再学習し，反実画像を生成して分類器の概念忠実性／頑健性を向上させる。**

---

## 1. データ準備

| ディレクトリ                             | 役割                                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `waterbirds/raw/{train,val,test}/` | 元画像 + `class.txt` (0=Land 1=Water)                                                          |
| `datasets/waterbirds_episodes/`    | 1 枚 = 1 ステップの `.npz` エピソード <br>`obs` (1,H,W,C), `act` (1,1)=0, `rew` (1)=0, `done` (1)=True |

**スクリプト** : `prepare_waterbirds.py`

```bash
python prepare_waterbirds.py --data_root waterbirds/raw \
                             --out_dir datasets/waterbirds_episodes
```

---

## 2. 段階 A ― Dreamer Encoder 事前学習

### 2.1 Config (`configs/waterbirds_minimal.yaml`)

* `batch_length: 1`, `action_dim: 1`, `dreamer.use_policy: false`, `dreamer.use_value: false`
* `offline_{data,eval,test}_dir` を上記エピソードパスへ
* `cnn_depth: 64`, `deter_dim: 1024`, `stoch_dim: 32` (調整可能)

### 2.2 実行

```bash
python launch.py --configs defaults waterbirds_minimal
```

* 出力: `checkpoints/world_model.ckpt` (Encoder / RSSM / Decoder)

---

## 3. 分類ヘッド学習

### 3.1 潜在特徴抽出

```python
world = DreamerWorldModel.load_from_checkpoint('world_model.ckpt').eval().cuda()
with torch.no_grad():
    z = world.encoder(img_batch.cuda())[0]  # stoch 部
```

### 3.2 ヘッド定義

```python
head = nn.Sequential(nn.Linear(latent_dim,256), nn.ReLU(), nn.Linear(256,2))
loss = nn.CrossEntropyLoss()
```

* **データ**: `train_loader` (Waterbirds train split)
* **評価**: Accuracy, F1, Confusion, ERM (spurious split毎)

### 3.3 ResNet50 ベースライン

* `torchvision.models.resnet50(weights="IMAGENET1K_V2")`
* 同データで fine‑tune → 指標を **共通表形式** で比較

---

## 4. 段階 C ― 概念編集シーケンス

| 要素                | 仕様                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------- |
| **環境**            | `WaterbirdsConceptEnv` : `reset(img)` → factual, `step(alpha)` → ConceptShifter(alpha\*d) |
| **action\_dim**   | 1 (α ∈ \[0,1])                                                                            |
| **batch\_length** | 8–16 (factual → counterfactual)                                                           |
| **報酬**            | 0 (世界モデルのみ)                                                                               |
| **config 追加**     | `time_limit: batch_length`, `collector.min_episode_length: batch_length`                  |
| **生成手順**          | `zs = model.rollout(z0, actions=a_seq)` → `decoder(zs)`                                   |

### 4.1 分類器再学習 & 評価

* Fine‑tune ResNet50 と Dreamer+Head それぞれで **TCAV, Grad‑CAM IoU, ImageNet‑C mCE** を測定
* 追加学習後の頑健性向上率をレポート

---

## 5. リポジトリ構成 (提案)
実際の現在のリポジトリ構成を考慮していないため、今後きちんと考えるので、これはあまり考慮しなくて良い。
```
project_root/
├─ configs/
│   ├─ waterbirds_minimal.yaml
│   └─ waterbirds_concept.yaml
├─ datasets/
├─ scripts/
│   ├─ prepare_waterbirds.py      # エピソード生成
│   ├─ train_dreamer.py           # launch.py ラッパ
│   ├─ extract_latent.py          # z 抽出
│   └─ train_classifier.py        # CE 学習
├─ models/                        # 保存 ckpt
└─ notebooks/                     # 解析・可視化
```

---

## 6. タスク一覧 & 優先順位

|  優先 | タスク                                                         | ステータス | 担当 |
| :-: | ----------------------------------------------------------- | ----- | -- |
|  ①  | `.npz` エピソード生成スクリプトのキー名・shape 修正                            | ☐     | –  |
|  ②  | `waterbirds_minimal.yaml` で学習が走るか確認                         | ☐     | –  |
|  ③  | `extract_latent.py` → `train_classifier.py` で Baseline 指標取得 | ☐     | –  |
|  ④  | ResNet50 fine‑tune ベースライン実装                                 | ☐     | –  |
|  ⑤  | `WaterbirdsConceptEnv` 実装 + Dreamer 再学習                     | ☐     | –  |
|  ⑥  | 反実画像生成パイプライン & 評価指標スクリプト                                    | ☐     | –  |

---

## 7. 参考リンク

* Dreamer V2 論文 (Hafner+ 2020)
* Waterbirds Data Card (Sagawa et al. 2020)
* Visual‑TCAV 実装 [https://github.com/vis\_tcav](https://github.com/vis_tcav)  (概念方向抽出)

---

**備考**

* Dreamer 実装固有の `OfflineDataset` が NHWC or CHW を要求する点に注意。
* 行動 / 報酬は段階 A ではダミー。段階 C で α 介入を行動として利用。
* ハイパパラは GPU メモリ 24 GB なら `batch_size: 32` で問題なし。

---

Copilot へのお願い

1. 上記タスク表を issue/PR ベースで管理してください。
2. 学習・評価スクリプトには Hydra / PyTorch Lightning を使用し CLI 化すると再現性が高まります。
3. エラー発生時はログ・スタックトレースをこのドキュメントに追記しながら対応してください。
