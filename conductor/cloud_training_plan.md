# 云端训练计划（Phase 5 详细版）

> 本文档是 plan.md Phase 5 的详细展开，包含每一步的具体操作命令。

---

## 整体思路

```
本地 Mac                        云端 GPU 服务器
────────────────                ─────────────────────────────
代码（src/、scripts/）  ──rsync/git──►  相同代码
configs/base.yaml       ──────────────►  相同配置
                                         │
                        ←── 自动下载 ──── 全量数据集（opus-100 / WMT17）
                                         │
                                       训练（CUDA + AMP）
                                         │
                        ←── scp ──────── checkpoints/best.pt
                                         │
本地推理验证           ◄─────────────────┘
```

---

## P5-1：选择云端实例

### 推荐配置

| 平台 | 实例 | GPU | 显存 | 参考价格 | 推荐度 |
|---|---|---|---|---|---|
| AutoDL（国内） | RTX 4090 | 1× | 24GB | ~¥2/小时 | ★★★★★ |
| AutoDL（国内） | A100 40G | 1× | 40GB | ~¥6/小时 | ★★★★☆ |
| Vast.ai（海外） | RTX 4090 | 1× | 24GB | ~$0.3/小时 | ★★★★☆ |
| Lambda Labs | A100 80G | 1× | 80GB | ~$1.1/小时 | ★★★☆☆ |

**建议：AutoDL RTX 4090**，24GB 显存对 batch_size=64 的论文配置足够，价格低。

### 显存估算（base.yaml，d_model=512，batch=64）

```
模型参数（论文原版）：~65M 参数 × 4字节 = ~260MB
激活值（batch=64，seq=128）：~2GB
Adam优化器状态（参数×2）：~520MB
梯度：~260MB
总计：≈ 3GB
────────────────────────────────
RTX 4090 24GB → 充裕，可用 batch=128 加速
```

---

## P5-2：上传代码与数据

### 方案 A：Git + 云端下载数据（推荐）

**好处**：代码用 git 管理，数据直接在云端下载，不用上传大文件。

```bash
# 本地：初始化 git 仓库（.gitignore 已排除 data/ checkpoints/ .env）
cd /Users/yul/coding/reproduct_transformer
git init
git add .
git commit -m "initial: transformer zh-en implementation"
git remote add origin git@github.com:yourname/reproduct_transformer.git
git push -u origin main

# 云端：clone 代码
git clone git@github.com:yourname/reproduct_transformer.git
cd reproduct_transformer

# 云端：创建 conda 环境
conda create -n ds_env python=3.10 -y
pip install torch torchvision tokenizers datasets sacrebleu pyyaml tqdm

# 云端：创建 .env 文件（填入你的 HF_TOKEN）
echo "HF_TOKEN=你的token" > .env

# 云端：准备全量数据（会自动下载 opus-100 全部数据）
python scripts/prepare_data.py --full
```

### 方案 B：rsync 直接同步（本地数据已有时）

```bash
# 把本地文件同步到云端（排除 .git 和 checkpoints）
rsync -avz --exclude='.git' --exclude='checkpoints/' \
  /Users/yul/coding/reproduct_transformer/ \
  root@云端IP:/root/reproduct_transformer/
```

---

## P5-3：添加全量数据准备支持

当前 `prepare_data.py` 只有 `--debug` 模式（10k 对）。
需要新增 `--full` 模式，下载 opus-100 全部 ~1M 中英句对。

**在云端运行：**
```bash
python scripts/prepare_data.py --full \
  --src_vocab_size 32000 \
  --tgt_vocab_size 32000
```

全量数据处理步骤：
1. 下载 opus-100 en-zh 全部 split（train/validation/test）
2. 清洗（约剩 ~900k 对）
3. 训练 ZH BPE（vocab=32000）和 EN BPE（vocab=32000）
4. 编码全部数据，保存到 data/processed/full/

---

## P5-4：配置云端训练脚本

云端使用 `configs/base.yaml`（论文原始超参数）：

```yaml
model:
  d_model: 512       # 论文原版（本地是128）
  num_heads: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  d_ff: 2048
  dropout: 0.1
  max_seq_len: 128   # 建议用128而非512，覆盖95%句子且省显存

train:
  batch_size: 64     # RTX 4090 可用128
  max_epochs: 30
  warmup_steps: 4000
  label_smoothing: 0.1
  grad_clip: 1.0
```

需新建 `scripts/train_cloud.py`，在 `train_local.py` 基础上增加：
- 读取 `configs/base.yaml`
- 启用 AMP 混合精度（`torch.cuda.amp`）
- 支持断点续训（`--resume checkpoints/last.pt`）
- 定期保存 `last.pt`（每 N step，不只保存 best）

---

## P5-5：启用混合精度训练（AMP）

AMP（Automatic Mixed Precision）：用 fp16 做前向传播，fp32 更新参数。
效果：显存减半，速度提升 2-3 倍，精度几乎不变。

核心改动在 trainer.py 的训练循环：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 前向传播用 fp16
with autocast():
    logits = model(src, tgt_in)
    loss = criterion(logits, tgt_out)

# 反向传播缩放梯度（防止 fp16 下溢）
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
```

---

## P5-6：预期训练时间与 BLEU 目标

### 训练时间估算（RTX 4090，batch=64，~900k 句对）

```
每个 epoch：~900k / 64 ≈ 14000 steps
每 step 耗时（AMP）：~50ms
每 epoch 耗时：~700 秒 ≈ 12 分钟
30 epochs 总计：≈ 6 小时
费用（AutoDL ¥2/小时）：≈ ¥12
```

### BLEU 目标

| 阶段 | BLEU | 说明 |
|---|---|---|
| debug 小模型（已完成）| 0.6~1.2 | 正常，模型太小 |
| 云端 10 epoch | ~8-12 | 可见明显翻译结构 |
| 云端 30 epoch | ~20-28 | 接近论文水平（论文EN-DE是28.4）|
| 论文 EN-ZH 参考 | ~20 | ZH-EN比EN-DE难，词表利用率低 |

---

## P5-7：训练监控

建议在云端用 `tmux` 保持会话，防止 SSH 断开：

```bash
tmux new -s train
python scripts/train_cloud.py 2>&1 | tee logs/train.log
# Ctrl+B D 退出 tmux（不终止训练）
# tmux attach -t train  重新连接
```

---

## P5-8：取回结果

```bash
# 训练完成后，把 checkpoint 和日志拉回本地
scp root@云端IP:/root/reproduct_transformer/checkpoints/best.pt \
    /Users/yul/coding/reproduct_transformer/checkpoints/cloud_best.pt

scp root@云端IP:/root/reproduct_transformer/logs/train.log \
    /Users/yul/coding/reproduct_transformer/logs/

# 本地评估
python scripts/evaluate.py \
  --ckpt checkpoints/cloud_best.pt \
  --config configs/base.yaml \
  --method beam \
  --samples 1000
```

---

## 执行顺序总结

```
P5-1  选实例（AutoDL RTX 4090）
P5-2  上传代码（git push → 云端 clone）
P5-3  云端跑 prepare_data.py --full（下载+处理全量数据）
P5-4  新建 scripts/train_cloud.py（加 AMP、断点续训）
P5-5  云端启动训练（tmux 保持会话）
P5-6  等待 ~6 小时（可中途查看 train.log）
P5-7  scp 取回 best.pt
P5-8  本地跑 evaluate.py 验证 BLEU
```

---

## 需要你决定的事项

- [ ] 使用哪个云平台？（AutoDL 国内 / Vast.ai 海外 / 其他）
- [ ] 数据集用 opus-100 继续（~1M 对）还是换 WMT17（~20M 对，质量更高但更慢）
- [ ] 是否需要我现在写好 `train_cloud.py` 和 `prepare_data.py --full` 模式？
