# LC-WM
Doing the research of Latent Counterfactual World Model (LC-WM).

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

### 3. 8 週間スタータープラン

| 週 | マイルストーン | 成果物 |
|----|----------------|--------|
| 1–2 | DreamerV2 を dmc_walker で動作確認 | ログ & 生成 GIF  
| 3–4 | Waterbirds で ResNet-50 ベースライン | テスト精度, 混同行列  
| 5–6 | TCAV / Grad-CAM 計器化 | 概念スコア & ヒートマップ  
| 7–8 | 潜在反実生成 (α版) → 再学習 | 反実画像セット, 指標改善グラフ  

---

### 4. 成功判定の主要指標  
| 指標 | 目標 |
|------|------|
| TCAV スコア（背景→前景） | 背景概念 下降、前景概念 上昇 ≥ +10 pt |
| Grad-CAM IoU | +5 pt 以上向上 |
| ImageNet-C mCE（Snow/Fog） | 20 % 以上低減 |
| Waterbirds OOD テスト Acc | ベースライン比 +3 pt |

---

### 5. 今後の拡張アイデア  
1. **Diffusion WM** 版に置き換えて高精細反実を生成  
2. 概念ボトルネック層をデコーダに挿入し，概念介入テストを実施  
3. 潜在空間境界可視化ツールを OSS として公開 → 引用稼ぎ

---

### 6. 主要リソースリンク  
- DreamerV2（TF）: <https://github.com/danijar/dreamerv2>  
- DreamerV2（PyTorch）: <https://github.com/jurgisp/pydreamer>  
- Captum TCAV: <https://captum.ai/tutorials>  
- xplique Explainability: <https://github.com/deel-ai/xplique>  
- Waterbirds dataset (Kaggle): <https://www.kaggle.com/datasets>  

---

### 7. すぐ始めるチェックリスト  
- [ ] `nvidia-smi` で GPU & CUDA を確認  
- [ ] DreamerV2 を clone → 1 episode 軽く学習  
- [ ] Waterbirds を DL → `datasets/` に配置  
- [ ] Captum で TCAV notebook を実行し，背景/前景スコアが出ることを確認  

---

**このメモがあれば、次のチャットや共同ドキュメントに即時インポートして議論を継続できます。**  
追加で必要な情報や詰まったポイントが出たら、いつでも質問してください。
