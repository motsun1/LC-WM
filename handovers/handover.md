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
├── cav/  # TCSVのキャッシュ関連のファイルが入ってるディレクトリ
├── datasets/waterbirds/
│   ├── concepts/  # TCAV学習用の概念（背景）ごとの画像セット 各ディレクトリ50枚ずつ画像が入っている
│   │   ├── background_land/
│   │   ├── background_water/
│   │   └── random_concept/
│   ├── waterbird_complete95_forest2water2/  # 水鳥データセットの元データ 鳥の種類ごとにディレクトリが分かれて画像が入っている。
│   │   └── metadata.csv  # ココに、各画像が陸鳥、水鳥のどちらか、train, test, val のどれかや、背景が水か陸かなどの情報がある
│   └── waterbirds_episodes/  # Dreamer 用のエピソード形式のデータ 各ディレクトリ
│        ├── test/  # 各画像に対する.npz ファイルと、metadata.json(陸鳥or水鳥、背景等の情報)がある
│        ├── train/  # 上に同じ
│        └── val/  # 上に同じ
├── handovers/
├── pydreamer/  # Dreamer V2 (pytorch版) の実行ファイルがある ソース: [PyDreamer GitHub](https://github.com/jurgisp/pydreamer)
│        └── launch.py  # pydreamer の実行ファイル
├── npz_extraction.py   # 潜在空間の抽出スクリプト
├── TCAV_evaluation.py   # TCAV 評価ファイル
├── train_resnet_waterbirds.py   # ResNet-50 の水鳥データセット用の学習ファイル
├── visual_tcav_latents.py  # 概念軸抽出 & 投影スクリプト
```

---

## 🧭 今後やること　Phase 2：概念介入による因果検証 in Dreamer潜在空間

### 🎯 目的

Dreamerの潜在空間における「概念方向への介入」が、分類という行動（たとえば鳥の種別）にどのように影響するかを**因果的に評価**する。

---

### ✅ ステップ1：Dreamerでの潜在空間学習（事前準備）

| タスク   | 詳細                                 | 備考                                                       |
| ----- | ---------------------------------- | -------------------------------------------------------- |
| ✅ 1.1 | 画像入力サイズ（224×224）対応のDreamer構造に修正    | - Encoderの最初のConv層変更<br>- RSSMのinput\_size再計算            |
| ✅ 1.2 | WaterbirdsでDreamerを教師あり分類タスクとして微調整 | - World Modelだが、「画像→潜在→Decoder→分類」の経路を試作<br>- 行動=分類として扱う |
| ✅ 1.3 | 潜在表現 \$z\$ の抽出コード整備                | - 画像ごとに\$z\$を取り出して保存（再利用性のため）                            |

---

### ✅ ステップ2：概念軸の構築と介入ベクトルの導出

| タスク   | 詳細                                | 備考                                                             |
| ----- | --------------------------------- | -------------------------------------------------------------- |
| ✅ 2.1 | Visual-TCAVで「背景」概念方向を抽出           | - positive set: 背景が水の画像<br>- negative set: 背景が草の画像             |
| ✅ 2.2 | TCAVのベクトルをDreamer潜在空間上に投影（または再学習） | - TCAVベクトルがDreamer latentにも通用するか検証<br>- 必要に応じて別途linear probe学習 |

---

### ✅ ステップ3：概念介入と因果検証

| タスク   | 詳細                                                           | 備考                                 |              |
| ----- | ------------------------------------------------------------ | ---------------------------------- | ------------ |
| ✅ 3.1 | 潜在表現 \$z\$ に対して \$z' = z + \alpha \cdot v\_{\text{背景}}\$ を適用 | - \$\alpha\$ を ±方向で変化させるインターポレーション |              |
| ✅ 3.2 | 変化後の \$z'\$ をDecoderに通して分類予測を取得                              | - \$p(y                            | z')\$ の変化を追跡 |
| ✅ 3.3 | 出力の変化を定量評価（分類確率/精度変化/信頼度）                                    | - 特定クラスへのスコア変化や出力のエントロピー           |              |

---

### ✅ ステップ4：因果的評価指標の構築（余力があれば）

| タスク    | 詳細                                                         |
| ------ | ---------------------------------------------------------- |
| 🔄 4.1 | Do-Intervention（= \$do(z+\alpha v)\$）とCF-Interventionの違い評価 |
| 🔄 4.2 | TCAV-score vs Concept Shift Sensitivity 指標の比較              |
| 🔄 4.3 | 因果グラフ視覚化（z → y）に概念方向を挿入して説明強化                              |

---

### その他詳細

1. **RSSMの224×224対応について**

   * 現状、`Conv2d`の設定を変えれば…と書いてありますが、**実際の出力次元の計算**を明示したほうが良いです。
   * 例えば、以下のような計算を行い、設計の目処をつけておくと良いでしょう：

   $$
   \text{出力サイズ} = \frac{\text{入力サイズ} - \text{カーネルサイズ} + 2 \times \text{パディング}}{\text{ストライド}} + 1
   $$

   特に最後のFlatten後のベクトルサイズが分かれば、RSSMのMLP構造を変更できます。

---

2. **潜在空間での概念介入の実装案**

   * **visual\_tcav\_latents.py** のところにある `project_to_latent()` を使うイメージだと思いますが、**Dreamerの潜在空間への対応**がまだ明確ではないので、ここを試してみる価値があります。
   * 必要であれば `linear probe` を使って、潜在空間上での「背景概念」をプローブする手法も有効です。

---

3. **因果的評価指標（ステップ4）の具体化**

   * 現状のプランは明確ですが、因果的評価の指標設計がまだ抽象的です。
   * 具体的には以下を追加してみてはどうでしょうか：

     * **ACE (Average Causal Effect):** 介入後の予測変化を平均的に評価
     * **Do-Calculus vs. Counterfactual Shift:** 直接介入 vs. 反実介入の効果差
     * **Causal Sensitivity Analysis:** 概念介入量に応じた分類スコアの感度

---

4. **今後のタスク管理**

   * 実装に取り掛かる際、以下の順序でタスクを管理すると効率的です：

     1. **Dreamerの224×224対応 → 潜在空間の抽出**
     2. **Visual-TCAVの再評価**（潜在空間への投影の確認）
     3. **概念介入の実装**（\$z + \alpha v\$ の操作）
     4. **因果評価の整備**（Do-Intervention, Counterfactual-Intervention）

---

## 🔄 **次のアクションプラン**

1. **RSSMの再設計**：224×224対応

   * Conv2dのフィルタサイズとストライドの計算
   * Flatten後の次元確認

2. **潜在空間の抽出**：学習したDreamerからの潜在出力の保存

   * npzファイルへの保存
   * Visual-TCAVと連携できるか検証

3. **概念ベクトルの投影**：Visual-TCAV結果をDreamer潜在に再利用

   * linear probeが必要か確認

4. **概念介入の設計**：\$\alpha\$の設定範囲とその影響の定量化

---

## 将来の見通し　Phase 3（多様な反実 or 長期影響）
* RSSM→Diffusion化で多様な予測へ
* 長期概念影響 or Video分類系へ応用

---

## 🔗 参考リソース

* [PyDreamer GitHub](https://github.com/jurgisp/pydreamer)
* [Captum TCAV Tutorial](https://captum.ai/tutorials)
* `tools/mlflow_log_npz` → `.npz` ログ保存箇所
* `prepare_batch_npz()` → decode後の画像をuint8で保存

---

