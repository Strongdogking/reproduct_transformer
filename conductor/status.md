# Project Status

> 最后更新：2026-03-08

---

## 当前阶段：Phase 4 完成 → Phase 5（云端训练）待开始

## 总体进度

| Phase | 名称 | 状态 |
|---|---|---|
| Phase 0 | 环境准备 | 已完成 |
| Phase 1 | 数据准备 | 已完成 |
| Phase 2 | 模型实现 | 已完成 |
| Phase 3 | 训练基础设施 | 已完成 |
| Phase 4 | 评估与推理 | 已完成 |
| Phase 5 | 云端训练 | 未开始 |
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

## 当前阻塞 / 待决策

- [ ] 需要决定云端实例类型（A100 / V100）和存储方案
- [ ] 全量数据集下载方案（WMT17 or 继续用 opus-100 full split）

---

## 下一步行动

**Phase 5：云端训练**
1. **P5-1** 申请云端 GPU 实例（A100 40G 推荐）
2. **P5-2** 上传代码 + 数据，或配置数据在线下载
3. **P5-3** 用 `base.yaml`（d_model=512, 6层, vocab×32000）训练
4. **P5-4** 启用混合精度（AMP）加速训练
5. **P5-5** 目标：newstest BLEU ≥ 25
