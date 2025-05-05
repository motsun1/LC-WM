# 🔁 PyDreamer × Visual-TCAV 引き継ぎ資料

## 📌 プロジェクト概要

Waterbirds データセットに対して DreamerV2 を用いて潜在表現を抽出し、Visual-TCAV によって概念軸（例：背景バイアス）を学習し、その軸方向へ反実移動 → decode により反実画像を生成する。

---

## ✅ 現状

| 項目                         | 状況                                  |
| -------------------------- | ----------------------------------- |
| Waterbirds による ResNet50 学習 | 完了                                  |
| PyDreamer による学習            | `launch.py`, `train.py` 実行可能状態      |
| 評価出力 `.npz` 形式             | あり（`d2_wm_closed/*/*.npz`）          |
| Visual-TCAV 実装             | `TCAV_evaluation.py` にResNet向けの実装あり |
| 潜在ベクトル → decode            | `image_rec` 等から復元可能であること確認済み        |

---

## 🗂 ディレクトリ構成（抜粋）

```
.
├── launch.py
├── train.py
├── config/
├── d2_wm_closed/
│   ├── train/
│   │   └── 0000100.npz
│   └── test/
│       └── 0000123_r55.npz
├── npz_extraction.py   # 潜在空間の抽出スクリプトを追加予定
├── visual_tcav_latents.py  # 概念軸抽出 & 投影スクリプトを追加予定
```

---

## 🧠 今後のToDo（Copilot向け）

1. **`.npz` の一括読み込みユーティリティ**

   * 特定のクラス（例: `label==0`）に対応する潜在ベクトルだけを取り出す

2. **概念ベクトルの構築**

   * `concept_A = land_background`
   * `concept_B = water_background or random`

3. **SVM / PCA による方向ベクトル取得**

   * `v = SVM(concept_A vs concept_B)` の法線ベクトル

4. **`z_cf = z + αv` による反実潜在生成**

   * αを変化させて中間反実を生成

5. **Dreamer の `decode(z_cf)` により RGB 画像へ再構成**

   * `image_rec`, `image_pred` の形式に揃えると良い

6. **生成画像の保存 & 可視化**

   * グリッド表示などにして比較用に保存

---

## 🔗 参考リソース

* [PyDreamer GitHub](https://github.com/jurgisp/pydreamer)
* [Captum TCAV Tutorial](https://captum.ai/tutorials)
* `tools/mlflow_log_npz` → `.npz` ログ保存箇所
* `prepare_batch_npz()` → decode後の画像をuint8で保存

---

