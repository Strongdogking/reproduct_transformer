# Development Plan — Transformer 中英翻译复现

---

## Phase 0: 环境准备

- [x] **P0-1** 创建 conda 环境 `ds_env`（Python 3.10）
- [x] **P0-2** 安装本地依赖（PyTorch CPU/MPS、tokenizers、sacrebleu 等）
- [ ] **P0-3** 生成 `environment.yml` 和 `requirements.txt`
- [x] **P0-4** 初始化项目目录结构
- [ ] **P0-5** 配置云端环境（GPU 实例 + CUDA 依赖）

---

## Phase 1: 数据准备

- [x] **P1-1** 下载 WMT17 中英数据集（或 OPUS100 备选）→ 使用 Helsinki-NLP/opus-100 en-zh
- [x] **P1-2** 数据清洗（去重、过滤过长/过短句子、去噪）
- [x] **P1-3** 训练 BPE 分词器（共享中英词表，vocab_size=8000 debug）
- [x] **P1-4** 将数据编码为 token ID，保存为 JSON 格式
- [x] **P1-5** 实现 `Dataset` 和 `DataLoader`（支持 padding、masking）
- [x] **P1-6** 本地调试用：提取 10k 句对子集（train=9000, val=1000）

---

## Phase 2: 模型实现

- [x] **P2-1** 实现 `ScaledDotProductAttention`
- [x] **P2-2** 实现 `MultiHeadAttention`（含 Q/K/V 线性投影）
- [x] **P2-3** 实现 `PositionwiseFeedForward`
- [x] **P2-4** 实现 `PositionalEncoding`（sin/cos）
- [x] **P2-5** 实现 `TokenEmbedding` + 与 PE 的组合层
- [x] **P2-6** 实现 `EncoderLayer` 和 `Encoder`（N=6 层堆叠）
- [x] **P2-7** 实现 `DecoderLayer` 和 `Decoder`（含 masked self-attn + cross-attn）
- [x] **P2-8** 实现完整 `Transformer` 模型（Encoder + Decoder + Linear + Softmax）
- [x] **P2-9** 单元测试：验证各模块输出 shape 正确

---

## Phase 3: 训练基础设施

- [x] **P3-1** 实现 `LabelSmoothingLoss`（epsilon=0.1）
- [x] **P3-2** 实现 Warmup LR Scheduler（论文公式）
- [x] **P3-3** 实现训练循环 `Trainer`（前向、反向、梯度裁剪、optimizer step）
- [x] **P3-4** 实现 checkpoint 保存与恢复
- [x] **P3-5** 实现训练日志（loss、lr、step）
- [x] **P3-6** 本地小规模冒烟测试（10k 数据，5 epochs，Val Loss 5.26，MPS 加速）

---

## Phase 4: 评估与推理

- [x] **P4-1** 实现 Greedy Decode 推理
- [x] **P4-2** 实现 Beam Search（beam_size=4）
- [x] **P4-3** 集成 sacrebleu，计算验证集 BLEU 分数
- [x] **P4-4** 编写推理示例脚本（scripts/evaluate.py）

---

## Phase 5: 云端训练

- [ ] **P5-1** 配置云端实例（推荐 A100 40G 或 V100）
- [ ] **P5-2** 上传数据集 / 配置远程存储
- [ ] **P5-3** 启用混合精度训练（`torch.cuda.amp`）
- [ ] **P5-4** 全量数据训练（目标：newstest2017 BLEU ≥ 25）
- [ ] **P5-5** 训练曲线分析，必要时调整超参数
- [ ] **P5-6** 保存最终模型权重，推理验证

---

## Phase 6: 收尾

- [ ] **P6-1** 整理代码，补充注释
- [ ] **P6-2** 更新 README
- [ ] **P6-3** 对比论文结果，记录复现差异

---

## 依赖关系

```
P0 → P1 → P2 → P3 → P4
                ↘         ↗
                 P5 (云端，依赖 P3 完成)
P4 + P5 → P6
```
