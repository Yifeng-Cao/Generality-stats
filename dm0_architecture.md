# DM0 (Dexbotic) 模型架构详解

> 源码路径: `docs/code/dexbotic-main/dexbotic-main/`

---

## 1. 架构总览

DM0 是一个 ~2B 参数的 Embodied-Native VLA 模型，核心设计是 **VLM + Action Expert 双专家架构**，通过 **Merged Attention** 实现高效融合。

```
多视角图像 → PE_LANG_L14_728 (ViT, 23层, 1024D) → 4x Projector → [B, 2705, 1536]
                                                                        ↓
语言指令 → Qwen3-1.7B embed_tokens → [B, L, 1536]                      ↓
                                                                        ↓
                    ┌──────────────── Prefix (VLM 侧) ─────────────────┐
                    │  images + language tokens, hidden=1536            │
                    │  28 层 Qwen3, GQA(16h/8kv), head_dim=128         │
                    └──────────────────────────────────────────────────┘
                                        ↕ Merged Attention (共享 KV)
                    ┌──────────────── Suffix (Action Expert 侧) ──────┐
                    │  noisy_actions + time, hidden=action_hidden      │
                    │  28 层 Qwen3, GQA(16h/8kv), head_dim=128         │
                    └──────────────────────────────────────────────────┘
                                        ↓
                    action_out_proj → v_t → MSE(v_t, u_t) → Flow Matching Loss
```

**关键文件:**
- 模型定义: `dexbotic/model/dm0/dm0_arch.py`
- 注意力掩码: `dexbotic/model/dm0/dm0_utils.py`
- 基类: `dexbotic/model/dexbotic_arch.py`
- 视觉编码器: `dexbotic/model/modules/mm_vision/pe/`
- 实验配置: `dexbotic/exp/dm0_exp.py`

---

## 2. 模型类层次

```
DexboticConfig
  └── DM0Config                          # dm0_arch.py:35
        action_config, action_dim=32, chunk_size=50

DexboticVLMModel
  └── DM0Model                           # dm0_arch.py:63
        ├── llm (Qwen3, 继承)
        ├── mm_vision_tower (PE_LANG_L14_728, 继承)
        ├── mm_projector (MLP 2x GELU, 继承)
        ├── action_expert (Qwen3ForCausalLM, embed_tokens=None)
        ├── action_in_proj: Linear(32 → action_hidden)
        ├── action_out_proj: Linear(action_hidden → 32)
        ├── action_time_mlp_in: Linear(2*action_hidden → action_hidden)
        └── action_time_mlp_out: Linear(action_hidden → action_hidden)

DexboticForCausalLM + ActionOutputForCausalLM
  └── DM0ForCausalLM                     # dm0_arch.py:128
        ├── _compute_merged_layer()      # 核心: 单层 Merged Attention
        ├── _merged_attention_forward()  # 逐层循环
        ├── forward()                    # 训练
        └── inference_action()           # Euler 采样推理
```

---

## 3. Merged Attention 机制 (核心创新)

> `dm0_arch.py:145-268`

这是 DM0 最核心的设计。VLM 和 Action Expert **共享注意力计算，但各自保留独立的 MLP**。

### 3.1 单层计算流程

```python
# dm0_arch.py:145 - _compute_merged_layer()

# Step 1: 各自独立计算 Q, K, V
for module_idx, (layer, input_embeds) in enumerate(zip(layers, input_embeds_list)):
    prenorm_embeds = layer.input_layernorm(input_embeds)
    query = layer.self_attn.q_norm(layer.self_attn.q_proj(prenorm_embeds))  # 独立 Q
    key   = layer.self_attn.k_norm(layer.self_attn.k_proj(prenorm_embeds))  # 独立 K
    value = layer.self_attn.v_proj(prenorm_embeds)                          # 独立 V

# Step 2: 拼接 Q, K, V (序列维度)
query_states = torch.cat(query_list, dim=2)   # [B, H, P+S, D]
key_states   = torch.cat(key_list, dim=2)     # [B, H, P+S, D]
value_states = torch.cat(value_list, dim=2)   # [B, H, P+S, D]

# Step 3: 共享 RoPE
cos, sin = self.model.llm.rotary_emb(dummy_tensor, position_ids)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# Step 4: 共享注意力计算
attn_output = eager_attention_forward(query_states, key_states, value_states, attention_mask)

# Step 5: 拆分输出，各自独立 o_proj + MLP
for module_idx, (layer, input_embeds) in enumerate(zip(layers, input_embeds_list)):
    attn_embeds = attn_output[:, start_idx:start_idx+seq_len, :]
    attn_embeds = layer.self_attn.o_proj(attn_embeds)           # 独立 o_proj
    residual = input_embeds + attn_embeds
    mlp_embeds = layer.mlp(layer.post_attention_layernorm(residual))  # 独立 MLP
    output = residual + mlp_embeds
```

### 3.2 共享 vs 独立

| 组件 | VLM | Action Expert | 共享? |
|------|-----|---------------|-------|
| input_layernorm | ✓ | ✓ | 独立 |
| q_proj, k_proj, v_proj | ✓ | ✓ | 独立 |
| q_norm, k_norm | ✓ | ✓ | 独立 |
| RoPE | ✓ | ✓ | **共享** (VLM 的 rotary_emb) |
| Attention 计算 | ✓ | ✓ | **共享** (拼接后统一计算) |
| o_proj | ✓ | ✓ | 独立 |
| MLP (gate_proj, up_proj, down_proj) | ✓ | ✓ | 独立 |
| post_attention_layernorm | ✓ | ✓ | 独立 |

### 3.3 与 pi0 的关键区别

- **pi0**: Action Expert 完全独立，仅通过 blockwise causal attention 的 KV 拼接交互
- **DM0**: Q, K, V 全部拼接后做统一 attention，信息流更充分
- **效率**: DM0 只需一次 attention 计算，pi0 需要分别计算

---

## 4. 注意力掩码设计

> `dm0_utils.py:12-92`

### 4.1 Cumsum 机制

```python
# dm0_utils.py:12
def make_attn_mask_2d(padding_mask, attn_mask):
    cumsum = torch.cumsum(attn_mask, dim=1)
    # token i 能 attend to token j 当且仅当 cumsum[j] <= cumsum[i]
    attn_mask_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    return attn_mask_2d & padding_mask_2d
```

### 4.2 掩码结构

```
Prefix (images + language):  attn_mask = [1, 1, 1, ..., 1, 1, 1, ..., 1]
                                          ↑ 每个 image view 开头为 1
Suffix (actions):            attn_mask = [1, 0, 0, ..., 0]
                                          ↑ 第一个 action token 为 1

结果:
- Prefix 内部: 双向注意力 (所有 cumsum 相同)
- Suffix 内部: 双向注意力 (cumsum 都等于 prefix_cumsum + 1)
- Suffix → Prefix: 可以 attend (prefix cumsum < suffix cumsum)
- Prefix → Suffix: 不能 attend (prefix cumsum < suffix cumsum)
```

**注意**: 这里 suffix 内部实际上是**双向**的（所有 action token 的 cumsum 相同），不是 causal。这与 pi0 的 blockwise causal 不同。

---

## 5. 时间步注入 (Flow Matching)

> `dm0_arch.py:355-404`

```python
# 1. 正弦余弦时间编码
time_embeddings = posemb_sincos(time, dim, min_period=4e-3, max_period=4.0)
# time: [B] → time_embeddings: [B, action_hidden]

# 2. 动作投影
action_hidden_states = action_in_proj(noisy_actions)  # [B, T, action_hidden]

# 3. 拼接 + MLP 融合
fused = torch.cat([action_hidden_states, time_embeddings.expand(...)], dim=2)  # [B, T, 2*action_hidden]
x = action_time_mlp_in(fused)   # [B, T, action_hidden]
x = F.silu(x)
hidden_states = action_time_mlp_out(x)  # [B, T, action_hidden]
```

**关键**: DM0 不使用 adaLN，而是通过 **concat + MLP** 将 timestep 融入 action embedding。这与 pi0 的做法一致。

---

## 6. 训练 Forward Pass

> `dm0_arch.py:406-511`

```
输入: images [B, num_views, C, H, W], input_ids [B, L], actions [B, chunk_size, 32]

1. 采样噪声和时间
   noise ~ N(0, 1), shape = actions.shape
   time ~ Beta(1.5, 1.0) * 0.999 + 0.001  → [0.001, 0.999]
   x_t = time * noise + (1-time) * actions
   u_t = noise - actions  (目标速度场)

2. 编码 Prefix
   每个视角: image → PE_LANG_L14_728 → mm_projector → [B, 2705, 1536]
   语言: input_ids → embed_tokens → [B, L, 1536]
   拼接: prefix = [img_view1; img_view2; ...; language]

3. 编码 Suffix
   noisy_actions → action_in_proj → [B, chunk_size, action_hidden]
   time → posemb_sincos → [B, action_hidden]
   concat + MLP → suffix = [B, chunk_size, action_hidden]

4. 构建注意力掩码
   cumsum 机制: prefix 双向, suffix 双向, suffix→prefix 可见, prefix→suffix 不可见

5. Merged Attention Forward (28 层)
   每层: 独立 QKV → 拼接 → 共享 attention → 拆分 → 独立 MLP

6. 计算损失
   v_t = action_out_proj(suffix_out[:, -chunk_size:])
   loss = MSE(v_t, u_t)
```

---

## 7. 推理 Forward Pass (Euler 采样)

> `dm0_arch.py:513-583`

```
1. 初始化: x_t ~ N(0,1), time=1.0, dt=-1/diffusion_steps

2. 编码 Prefix + 缓存 KV
   prefix → merged_attention_forward(use_cache=True) → kv_cache

3. Euler 循环 (默认 10 步):
   while time >= -dt/2:
     suffix = encode_suffix(x_t, time)
     suffix_out = merged_attention_forward(suffix, past_key_values=kv_cache)
     v_t = action_out_proj(suffix_out)
     x_t = x_t + v_t * dt
     time = time + dt

4. 返回 x_t (去归一化后的动作)
```

**KV Cache 策略**: Prefix 只编码一次，后续 Euler 步只重新编码 suffix。

---

## 8. 视觉编码器 (PE_LANG_L14_728)

> `dexbotic/model/modules/mm_vision/pe/pe_configuration.py`

| 参数 | 值 |
|------|-----|
| 输入分辨率 | 728×728 |
| Patch Size | 14×14 |
| Grid | 52×52 = 2704 patches (+1 CLS = 2705) |
| Hidden Dim | 1024 |
| 层数 | 23 |
| 注意力头 | 16 |
| MLP Ratio | 4.0 |
| 位置编码 | RoPE2D |
| LayerScale | init=0.1 |

**Projector**: `Linear(1024→1536) → GELU → Linear(1536→1536)`

---

## 9. 梯度隔离

> `dm0_arch.py:108-126`, `dexbotic/exp/dm0_exp.py`

DM0 的梯度隔离通过 `requires_grad` 控制:
- **Embodied 数据训练时**: VLM 参数 `requires_grad=False`，仅更新 Action Expert + 投影层
- **非 embodied 数据时**: VLM 正常训练

精度策略: 大部分参数 bfloat16，但 `conv1`, `positional_embedding`, `layernorm`, `model.norm` 保持 float32。

---

## 10. 关键超参数

| 参数 | 值 | 来源 |
|------|-----|------|
| action_dim | 32 | DM0Config |
| chunk_size | 50 | DM0Config |
| VLM hidden | 1536 (Qwen3-1.7B) | llm_config |
| VLM layers | 28 | llm_config |
| Action Expert hidden | action_config.hidden_size | action_config |
| Action Expert layers | 28 | action_config |
| GQA | 16 query heads, 8 kv heads | Qwen3 |
| head_dim | 128 | Qwen3 |
| 噪声分布 | Beta(1.5, 1.0) | forward() |
| 推理步数 | 10 (默认) | inference_action() |
| 归一化 | q01/q99 分位数 | dm0_exp.py |

---

## 11. 与当前系统的对比

| 维度 | DM0 | 当前系统 |
|------|-----|---------|
| VLM→Expert 融合 | **Merged Attention** (共享 KV + 独立 MLP) | Cross-Attention (prefix 固定) |
| 信息流 | **双向** (VLM ↔ Expert) | 单向 (VLM → DiT) |
| Timestep 注入 | concat + MLP (无 adaLN) | adaLN |
| Action Expert 架构 | Qwen3 (同 VLM 架构) | DiT (独立架构) |
| 注意力掩码 | cumsum 机制 (suffix 双向) | interleaved cross/self |
| 梯度隔离 | requires_grad 控制 | 无 |
| 损失 | 纯 Flow Matching MSE | Flow Matching MSE |
