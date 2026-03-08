# Project Status

> 最后更新：2026-03-08

---

## 当前阶段：Phase 5（云端训练）进行中

## 总体进度

| Phase | 名称 | 状态 |
|---|---|---|
| Phase 0 | 环境准备 | 已完成 |
| Phase 1 | 数据准备 | 已完成 |
| Phase 2 | 模型实现 | 已完成 |
| Phase 3 | 训练基础设施 | 已完成 |
| Phase 4 | 评估与推理 | 已完成 |
| Phase 5 | 云端训练 | **进行中** |
| Phase 6 | 收尾 | 未开始 |

---

## Phase 4 详情

| 任务 | 状态 | 备注 |
|---|---|---|
| P4-1 Greedy Decode | 已完成 | src/utils/inference.py |
| P4-2 Beam Search | 已完成 | beam_size=4，src/utils/inference.py |
| P4-3 sacrebleu BLEU 评估 | 已完成 | src/utils/bleu.py |
| P4-4 推理示例脚本 | 已完成 | scripts/evaluate.py |

---

## Phase 5 详情

| 任务 | 状态 | 备注 |
|---|---|---|
| P5-1 配置云端实例 | 已完成 | RTX 4090 24G，CUDA 可用 |
| P5-2 全量数据准备 | 已完成 | opus-100 full，885k train / 1928 val，vocab=32000，保存至 data/processed/full/ |
| P5-3 混合精度训练 | 已完成 | AMP + grad_accum_steps=2（等效 batch=128），configs/cloud.yaml |
| P5-4 全量数据训练 | **进行中** | train_cloud.py，TensorBoard → checkpoints/cloud/tb_logs/ |
| P5-5 训练曲线分析 | 未开始 | — |
| P5-6 最终权重保存与推理验证 | 未开始 | — |

---

## Debug 模型评估结果（5 epoch，9k 训练对）

| 方法 | BLEU（200 val 样本） | 备注 |
|---|---|---|
| Greedy | 0.62 | 有重复生成，符合小模型预期 |
| Beam Search (k=4) | 1.22 | 略高于 greedy |

> **注意**：低 BLEU 是预期结果。debug 模型 d_model=128、2层、仅 9k 训练数据、5 epoch。
> 真实评估须待云端 full 模型（d_model=512、6层、全量数据）。

---

## 架构关键决策记录

- **ZH/EN 分离词表**（已完成）：Encoder 专用 ZH tokenizer（4707 tokens），Decoder 专用 EN tokenizer（8000 tokens），不混用
- **中文字符化**：`tokenize_zh()` 在每个 CJK 字符两侧插空格，使 BPE 可在字符级处理中文
- **模型 vocab size 跟 tokenizer 走**：`train_local.py` 先加载 tokenizer 读取实际 vocab size，再建模型

---

## 本次 Session 关键修改记录

| 文件 | 修改内容 |
|---|---|
| `scripts/prepare_data.py` | 新增 `--full` 模式；HF 缓存改为项目内 `data/raw/hf_cache/` |
| `configs/cloud.yaml` | 新建云端配置（batch=64, grad_accum=2, AMP, max_seq_len=200）|
| `src/train/trainer.py` | 新增 AMP、梯度累积、TensorBoard SummaryWriter |
| `scripts/train_cloud.py` | 新建云端训练脚本，支持 --resume |
| `src/model/attention.py` | mask fill `-1e9` → `float('-inf')` 修复 AMP fp16 溢出 |

---

## 当前阻塞 / 待决策

无阻塞。训练正在运行中。

---

## 下一步行动

**Phase 5（进行中）：**
1. **P5-4** 等待训练完成（30 epochs）
   - 用 `tail -f` 查看日志
   - 用 `tensorboard --logdir /root/opt/transformer/reproduct_transformer/checkpoints/cloud/tb_logs` 查看曲线
2. **P5-5** 训练结束后分析 loss 曲线，判断是否需要调参
3. **P5-6** 运行 `scripts/evaluate.py --ckpt checkpoints/cloud/best.pt` 做最终 BLEU 评估
