## 研究ハンドオーバー用サマリー  
（このメモをそのまま新しいプロジェクトの README / Notion ページなどに貼り付ければ、すぐに議論を再開できます）

---

### 1. 研究テーマ  
**Latent Counterfactual World Model (LC-WM) で概念忠実性を測定・向上させる**  
- Dreamer 系の潜在空間を Visual-TCAV で概念軸抽出  
- 概念軸に沿った反実ロールアウトを生成・追加学習  
- Grad-CAM IoU / TCAV スコア / ImageNet-C mCE で  
  **「モデルは本質概念を見ているか」＋「頑健性は上がったか」** を定量評価  

---

### 2. 技術スタック  
| カテゴリ | 推奨ツール / 実装 | 備考 |
|----------|-----------------|------|
| 世界モデル | **Dreamer V2** 公式 TF (danijar/dreamerv2) <br>or PyTorch fork (pydreamer) | 64×64 RGB 入力に対応  
| 概念アトリビューション | **Captum** TCAV / Visual-TCAV notebook | PyTorch 完結  
| サリエンシーマップ & 指標 | **xplique**（Grad-CAM, Deletion/Insertion 曲線） | pip で即導入  
| データセット | **Waterbirds**（背景バイアス）<br>HardImageNet, ImageNet-C (頑健性) | 先行研究と比較しやすい  
| GPU 要件 | RTX 30xx (≥12 GB) 1 枚で実験可能 | Dreamer 64×64 なら OK  

---

### 3. 具体的な研究マイルストーン

以下では **Phase 2 ～ Phase 3（約 6 か月）** を例に、研究マイルストーンを「タスク ⇄ 成果物 ⇄ 評価基準」でブレイクダウンしました。ガント表に落とし込めるよう **週番号** を振っています（週 0 ＝ 来週開始、Asia/Tokyo 時間基準）。GPU＝RTX 4090 ×1 を想定した所要時間も目安として記載しました。

---

## 0 – 準備フェーズ（Week 0）

| タスク                                                  | 成果物              | 評価基準                         | 目安  |
| ---------------------------------------------------- | ---------------- | ---------------------------- | --- |
| ✅ Phase 1 の再現スクリプトを `scripts/phase1_baseline.sh` に集約 | Jupyter nb + シェル | Waterbirds TCAV Score 再現±2 % | 2 日 |

---

## Phase 2 ― Dreamer + 概念介入（Week 1-12）

### Milestone 1 「MuDreamer-128」導入 (W1-W3)

| タスク                                | 成果物                      | 評価基準                 | GPU 時間 |
| ---------------------------------- | ------------------------ | -------------------- | ------ |
| 1-1 画像前処理 128×128, 3ch             | `dataset/waterbirds128/` | 全 4,795 枚変換          | 0.5h   |
| 1-2 `dreamer/models/mu_rssm.py` 実装 | Pull Request #m1         | unit-test 100 % pass | —      |
| 1-3 学習 & 早期停止ロジック                  | `runs/m1_mu128/`         | ELBO ↑/plateau判定     | 24h    |
| 1-4 Latent z → Linear Probe 分類     | notebook                 | Top-1 ≥ 80 %         | 1h     |

### Milestone 2 「概念報酬設計」(W4-W6)

| タスク                                      | 成果物               | 評価基準              | GPU 時間 |
| ---------------------------------------- | ----------------- | ----------------- | ------ |
| 2-1 TCAV バッチ評価 API (`concept_reward.py`) | モジュール             | 1エピソード < 0.3 s    | —      |
| 2-2 `dreamer/agent.py` に報酬 Hook 追加       | PR #m2            | 追加コード < +300 行    | —      |
| 2-3 RL 事前学習 (β=0.1)                      | `runs/m2_beta01/` | Grad-CAM IoU +5pp | 30h    |
| 2-4 β スイープ (0.0-1.0, 5 点)                | CSV レポート          | β\* → 最大忠実度       | 60h    |

### Milestone 3 「latent 介入 vs. CEILS 比較」(W7-W9)

| タスク                                    | 成果物               | 評価基準               |
| -------------------------------------- | ----------------- | ------------------ |
| 3-1 CEILS 実装（既存 repo fork）             | `external/ceils/` | 再現例 MNIST OK       |
| 3-2 共通 API (`intervene(z, dir, α)`) 整備 | PR #m3            | Dreamer/CEILS 両対応  |
| 3-3 介入実験 500 本 × 3 α                   | `results/intv/`   | ΔConfDrop, ΔIoU 計算 |
| 3-4 論文化図表 (Fig 3, Tab 2)               | `paper/figs/`     | 学会テンプレ合致           |

### Milestone 4 「汎化評価 & アブレーション」(W10-W12)

| タスク                           | 成果物                  | 評価基準   |
| ----------------------------- | -------------------- | ------ |
| 4-1 CelebA “Smiling/Not” で再学習 | `runs/celeba/`       | 同手順    |
| 4-2 Ablation: ✗報酬, ✗RSSM, ✗γ  | `results/abl/`       | 全組合せ完了 |
| 4-3 統合レポート草稿 (8 p)            | `paper/draft_v1.tex` | 5/末 まで |

---

## Phase 3 ― Diffusion 拡張 & 動画応用（Week 13-24）

### Milestone 5 「Diffusion Prior 接続」(W13-W16)

| タスク                           | 成果物                | 評価基準     |
| ----------------------------- | ------------------ | -------- |
| 5-1 Latent→Timestep 映写層実装     | `models/prior.py`  | FID ≤ 35 |
| 5-2 Concept Slider (LoRA) 埋込み | `models/slider.py` | α 連続制御可  |

### Milestone 6 「長期概念影響シミュレーション」(W17-W20)

| タスク                              | 成果物                | 評価基準        |
| -------------------------------- | ------------------ | ----------- |
| 6-1 時系列介入 (32 step rollout)      | `results/long_cf/` | 画像崩壊率 < 5 % |
| 6-2 Video Waterbirds (鳥飛翔シーン) 収集 | `dataset/vwb/`     | 400 クリップ    |

### Milestone 7 「最終実験＋論文仕上げ」(W21-W24)

| タスク                      | 成果物               | 評価基準      |
| ------------------------ | ----------------- | --------- |
| 7-1 全タスク再実行・seed=3 平均    | `results/final/`  | 結果確定      |
| 7-2 論文最終稿 + 補足資料         | `paper/final.pdf` | 7/31 締切   |
| 7-3 arXiv 投稿 & GitHub 公開 | DOI 取得            | CC-BY-4.0 |

---

## リスク & バックアップ

| リスク            | 対策                                           |
| -------------- | -------------------------------------------- |
| Dreamer 学習が不安定 | ①ムービング平均 KL 制御 ②再構成レス版（MuDreamer）            |
| 概念報酬がノイジー      | スムージング (EMA 0.9) + 報酬クリッピング                  |
| GPU 枯渇         | 128×128 & mixed-precision, kolmogorov スケーリング |

---
