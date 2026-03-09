# LingBot-VLA 模型架构详解

> 源码路径: `docs/code/lingbot-vla-main/lingbot-vla-main/`

---

## 1. 架构总览

LingBot-VLA 是一个工程导向的 VLA 框架，采用 **MoT (Mixture-of-Transformers)** 架构：Qwen2.5-VL 作为 VLM backbone + Qwen2 作为 Action Expert，两者**共享 self-attention + 独立 MLPs**。额外支持 LingBot-Depth 深度蒸馏模块。

```
多视角图像 → Qwen2.5-VL ViT (32层, 1280D) → spatial_merge(2x2) → [B, n*patch, 2048]
语言指令 → Qwen2.5-VL embed_tokens → [B, L, 2048]
(可选) Depth Align Tokens → learnable queries → [B, 3*num_task_tokens, 2048]
                                    ↓
                    ┌──── Prefix (VLM 侧, Qwen2.5-VL) ────────────┐
                    │  images + (align_tokens) + language           │
                    │  36 层, hidden=2048, GQA(16h/2kv)            │
                    └──────────────────────────────────────────────┘
                                    ↕ MoT: 共享 Attention, 独立 MLP
                    ┌──── Suffix (Action Expert, Qwen2) ──────────┐
                    │  state + noisy_actions (+ time via AdaRMSNorm)│
                    │  36 层, hidden=768, GQA(16h/2kv), head_dim=128│
                    └──────────────────────────────────────────────┘
                                    ↓
                    action_out_proj → v_t → Loss(v_t, u_t)
                    (可选) depth_align_head → depth_pred → depth_loss
```

**关键文件:**
- 主模型: `lingbotvla/models/vla/pi0/modeling_lingbot_vla.py`
- Qwen2.5-VL 集成: `lingbotvla/models/vla/pi0/qwenvl_in_vla.py`
- Flex Attention: `lingbotvla/models/vla/pi0/flex_attention.py`
- 工具函数: `lingbotvla/models/vla/pi0/utils.py`
- 深度头: `lingbotvla/models/vla/vision_models/align_heads/depth_head.py`
- Resampler: `lingbotvla/models/vla/vision_models/align_heads/resampler.py`
- Flow Matching: `lingbotvla/schedulers/flow_match.py`
- 配置: `configs/vla/robotwin_load20000h.yaml`

---

## 2. 模型类层次

```
PretrainedConfig
  └── QwenvlWithExpertConfig             # modeling_lingbot_vla.py:919
        ├── qwenvl_config (Qwen2.5-VL)
        ├── qwen_expert_config (Qwen2)
        ├── freeze_vision_encoder: bool
        ├── train_expert_only: bool
        ├── attention_implementation: "eager"/"flex"
        ├── adanorm_time: bool
        └── split_gate_liner: bool

PreTrainedModel
  └── QwenvlWithExpertModel              # modeling_lingbot_vla.py:1234
        ├── qwenvl: Qwen2_5_VLForConditionalGeneration
        ├── qwen_expert: Qwen2ForCausalLM (无 embed_tokens)
        ├── (可选) expert_visual: DINOv3
        └── attention_interface: flex/eager

PI0Config → QwenVLA_Config
PreTrainedPolicy
  └── LingbotVlaPolicy                   # modeling_lingbot_vla.py:1500
        └── model: FlowMatching

nn.Module
  └── FlowMatching                       # modeling_lingbot_vla.py:1563
        ├── qwenvl_with_expert: QwenvlWithExpertModel
        ├── state_proj: Linear(max_state_dim → 768)
        ├── action_in_proj: Linear(max_action_dim → 768)
        ├── action_out_proj: Linear(768 → max_action_dim)
        ├── action_time_mlp_in: Linear(768*2 → 768)
        ├── action_time_mlp_out: Linear(768 → 768)
        └── (可选) depth_align_head: TaskTokenDepthHead
```

---

## 3. VLM Backbone (Qwen2.5-VL)

> `modeling_lingbot_vla.py:945-1003`

### 3.1 配置

| 参数 | 值 |
|------|-----|
| hidden_size | 2048 |
| num_hidden_layers | 36 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 (GQA 8:1) |
| intermediate_size | 11008 |
| head_dim | 128 |
| max_position_embeddings | 128000 |
| sliding_window | 32768 |
| RoPE | mRoPE (16, 24, 24) |

### 3.2 Vision Encoder

```
ViT: 32 层, hidden=1280, 16 heads
Patch: 14×14, spatial_merge_size=2
Window Attention: window_size=112, full attention at layers [7, 15, 23, 31]
输出: spatial_merge 后 token 数 = (H/14/2) × (W/14/2)
```

### 3.3 梯度隔离

> `modeling_lingbot_vla.py:1268-1288`

```python
def set_requires_grad(self):
    if self.config.freeze_vision_encoder:
        self.qwenvl.visual.eval()
        for params in self.qwenvl.visual.parameters():
            params.requires_grad = False

    if self.config.train_expert_only:
        self.qwenvl.eval()
        for params in self.qwenvl.parameters():
            params.requires_grad = False
```

**关键**: `train_expert_only=True` 时，整个 Qwen2.5-VL 冻结，仅训练 Action Expert + 投影层。`use_ki=True` 时还会 `.detach()` VLM 的 QKV 输出，彻底切断梯度。

---

## 4. Action Expert (Qwen2)

> `modeling_lingbot_vla.py:1011-1050`

| 参数 | 值 |
|------|-----|
| hidden_size | 768 |
| num_hidden_layers | 36 (与 VLM 相同) |
| num_attention_heads | 16 |
| num_key_value_heads | 2 (GQA 8:1) |
| intermediate_size | 2752 |
| head_dim | 128 |

**注意**: Action Expert 与 VLM **层数相同 (36层)**，但 hidden 维度小得多 (768 vs 2048)。`embed_tokens` 被删除，因为 action 通过独立投影层编码。

---

## 5. MoT (Mixture-of-Transformers) 机制 (核心)

> `modeling_lingbot_vla.py:1365-1477`

### 5.1 Forward 流程

```python
def forward(self, attention_mask, position_ids, inputs_embeds=[prefix_embs, suffix_embs], ada_cond=None):
    models = [self.qwenvl.model, self.qwen_expert.model]
    num_layers = 36

    for layer_idx in range(num_layers):
        # ===== Phase 1: 各自独立计算 Q, K, V =====
        query_states, key_states, value_states = [], [], []
        gates = []

        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            if i == 1:  # Action Expert (可能带 AdaRMSNorm + gate)
                q, k, v, gate = models[i].layers[layer_idx](
                    hidden_states, compute_kqv=True, ada_cond=ada_cond)
            else:       # VLM
                q, k, v = models[i].layers[layer_idx](
                    hidden_states, compute_kqv=True)
                gate = None
                if use_ki:  # 梯度隔离: detach VLM 的 QKV
                    q, k, v = q.detach(), k.detach(), v.detach()

            query_states.append(q)
            key_states.append(k)
            value_states.append(v)
            gates.append(gate)

        # ===== Phase 2: 拼接 + 共享 Attention =====
        query_states = torch.cat(query_states, dim=1)   # [B, L_vlm+L_action, H, D]
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)

        query_states = apply_rope(query_states, position_ids)
        key_states = apply_rope(key_states, position_ids)

        # KV Cache 处理
        key_states, value_states, past_key_values = self.handle_kv_cache(...)

        # 统一 attention 计算 (flex / eager)
        att_output = self.attention_interface(
            query_states, key_states, value_states, attention_mask)

        # ===== Phase 3: 拆分 + 各自独立 MLP =====
        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                end = start + hidden_states.shape[1]
                if i == 1:  # Action Expert: 带 gate
                    out = models[i].layers[layer_idx](
                        hidden_states, att_output, start, end,
                        output_atten=True, ada_cond=ada_cond, gate=gates[i])
                else:       # VLM
                    out = models[i].layers[layer_idx](
                        hidden_states, att_output, start, end, output_atten=True)
                outputs_embeds.append(out)
                start = end

        inputs_embeds = outputs_embeds

    # Final Norm (各自独立)
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is not None:
            outputs_embeds[i] = models[i].norm(hidden_states)
```

### 5.2 共享 vs 独立

| 组件 | VLM (Qwen2.5-VL) | Action Expert (Qwen2) | 共享? |
|------|-------------------|----------------------|-------|
| input_layernorm | RMSNorm | AdaRMSNorm (可选) | 独立 |
| q_proj, k_proj, v_proj | ✓ | ✓ | 独立 |
| RoPE | ✓ | ✓ | **共享** (统一 position_ids) |
| Attention 计算 | ✓ | ✓ | **共享** (拼接后统一计算) |
| o_proj | ✓ | ✓ | 独立 |
| MLP (gate/up/down_proj) | ✓ | ✓ | 独立 |
| post_attention_layernorm | RMSNorm | AdaRMSNorm (可选) | 独立 |
| Final norm | RMSNorm | AdaRMSNorm (可选) | 独立 |

### 5.3 与 DM0 Merged Attention 的对比

两者本质相同，都是 "共享 attention + 独立 MLP"，但实现细节不同:

| 维度 | LingBot MoT | DM0 Merged Attention |
|------|-------------|---------------------|
| VLM | Qwen2.5-VL (36层, 2048D) | Qwen3 (28层, 1536D) |
| Expert | Qwen2 (36层, 768D) | Qwen3 (28层, action_hidden) |
| Timestep 注入 | AdaRMSNorm (FiLM) | concat + MLP |
| Gate 机制 | **有** (split_gate_liner) | 无 |
| Attention 实现 | flex_attention / eager | eager |
| 梯度隔离 | requires_grad + detach | requires_grad |

---

## 6. AdaRMSNorm + Gate 机制

> `modeling_lingbot_vla.py:1096-1148`

LingBot 的 Action Expert 可选使用 **AdaRMSNorm** 替代标准 RMSNorm，实现 timestep 条件注入:

```python
class AdaRMSNorm(nn.Module):
    """RMSNorm + FiLM (Feature-wise Linear Modulation)"""
    def __init__(self, hidden_size, cond_dim, split_gate_liner, ...):
        self.gamma = nn.Linear(cond_dim, hidden_size)  # scale
        self.beta = nn.Linear(cond_dim, hidden_size)   # shift
        if split_gate_liner:
            self.gate = nn.Linear(cond_dim, hidden_size)  # gate
            nn.init.zeros_(self.gate.weight)  # 零初始化 → 初始 gate=0

        # DiT 风格初始化: gamma=0, beta=0 → 初始为 identity
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, hidden_states, cond):
        # 标准 RMSNorm
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)

        # FiLM modulation
        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(cond).unsqueeze(1)
        hidden_states = (1 + gamma) * hidden_states + beta

        if self.use_gate:
            gate = self.gate(cond).unsqueeze(1)
        return hidden_states, gate
```

**Gate 的使用**: 在 decoder layer 的 output_atten 阶段，gate 作为残差连接的缩放因子:
```python
# 伪代码
attn_out = o_proj(att_output[:, start:end])
if gate is not None:
    hidden_states = hidden_states + gate * attn_out  # gated residual
else:
    hidden_states = hidden_states + attn_out
```

**替换机制** (`replace_lnorm_with_adanorm`): 遍历 Action Expert 的所有 `Qwen2RMSNorm`，替换为 `AdaRMSNorm`，条件输入为 time embedding。

---

## 7. 注意力掩码

> `utils.py:65-95`

### 7.1 Cumsum 机制 (与 DM0 相同)

```python
def make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks
```

### 7.2 掩码结构

```
Prefix:
  img_emb: att_masks = [0, 0, ..., 0]  (双向)
  (align_tokens: att_masks = [0, 0, ..., 0])
  lang_emb: att_masks = [0, 0, ..., 0]  (双向)
  → 所有 prefix 内部双向

Suffix:
  state: att_masks = [True]   ← 新 block 开始
  action_0: att_masks = [True] ← 新 block 开始
  action_1..N: att_masks = [False, ..., False]  (双向)

结果:
  - Prefix 内部: 双向
  - Suffix: state 独立 block, action tokens 双向
  - Suffix → Prefix: 可见
  - Prefix → Suffix: 不可见
  - State → Action: 不可见 (state cumsum < action cumsum)
```

**注意**: `att_masks[:, :2] = True` 表示 state 和第一个 action token 各自开启新 block。State 不能看到 action tokens。

### 7.3 Depth Align Tokens 的特殊掩码

> `modeling_lingbot_vla.py:1979-1998`

当使用 depth alignment 时，align tokens 有特殊的注意力规则:
- 每组 align tokens 只能看到对应视角的 image tokens + 自身
- Image tokens 和 language tokens 不能看到 align tokens
- 3 组 align tokens 互相不可见

---

## 8. Flow Matching

### 8.1 噪声采样

> `modeling_lingbot_vla.py:1666-1669`

```python
def sample_time(self, bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)  # Beta(1.5, 1.0)
    time = time_beta * 0.999 + 0.001  # → [0.001, 0.999]
    return time
```

### 8.2 训练 Forward

> `modeling_lingbot_vla.py:1807-1892`

```python
def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, ...):
    noise = torch.randn(actions.shape)
    time = self.sample_time(bsize, device)

    # Flow matching interpolation
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    u_t = noise - actions  # target velocity

    # Embed prefix & suffix
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, ...)
    time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

    # Build attention mask
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

    # MoT forward
    (outputs_embeds, suffix_out), _ = self.qwenvl_with_expert.forward(
        attention_mask=att_2d_masks,
        inputs_embeds=[prefix_embs, suffix_embs],
        ada_cond=time_embs if adanorm_time else None,
    )

    # Depth loss (可选)
    if self.use_depth_align:
        loss_depth, depth_preds = self.depth_emb_forward(outputs_embeds, depth_targets, img_masks)

    # Action loss
    suffix_out = suffix_out[:, -n_action_steps:]
    v_t = self.action_out_proj(suffix_out)
    losses = F.mse_loss(u_t, v_t, reduction="none")  # 或 L1

    return losses, loss_depth, depth_preds
```

### 8.3 推理 (Euler 采样)

> `modeling_lingbot_vla.py:1894-1943`

```python
def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, ...):
    # 1. Encode prefix + cache KV
    _, past_key_values = self.qwenvl_with_expert.forward(
        inputs_embeds=[prefix_embs, None], fill_kv_cache=True)

    # 2. Euler loop
    x_t = noise
    time = 1.0
    dt = -1.0 / num_steps
    while time >= -dt/2:
        v_t = self.predict_velocity(state, prefix_pad_masks, past_key_values, x_t, time)
        x_t += dt * v_t
        time += dt

    return x_t
```

---

## 9. LingBot-Depth 深度蒸馏

### 9.1 架构

> `depth_head.py:44-66`, `resampler.py:163-202`

```
LLM hidden states (image tokens) → TaskTokenResampler → depth embeddings
                                         ↓
                                    PerceiverAttention (多层 cross-attention)
                                         ↓
                                    depth_pred → L1 loss vs depth_target
```

### 9.2 TaskTokenResampler

```python
class TaskTokenResampler(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_head, dim_out, num_layers, num_queries, num_heads, ff_mult):
        self.proj_in1 = nn.Linear(dim_in, dim_mid)   # LLM features
        self.proj_in2 = nn.Linear(dim_in, dim_mid)   # Query tokens
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)
        self.layers = nn.ModuleList([
            [PerceiverAttention(dim_mid, dim_head, num_heads),
             FeedForward(dim_mid, mult=ff_mult)]
            for _ in range(num_layers)
        ])

    def forward(self, x, queries):
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)
        for attn, ff in self.layers:
            queries = attn(x, queries) + queries  # cross-attention + residual
            queries = ff(queries) + queries
        return self.norm_out(self.proj_out(queries))
```

### 9.3 PerceiverAttention

```python
class PerceiverAttention(nn.Module):
    def forward(self, x, latents):
        q = self.to_q(latents)                          # queries attend
        kv_input = torch.cat((x, latents), dim=-2)      # to image features + self
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        weight = softmax((q * scale) @ (k * scale).T)
        return self.to_out(weight @ v)
```

### 9.4 Depth Loss

> `modeling_lingbot_vla.py:2000-2050`

```python
# Query 模式
align_embs = hidden_states[:, chunk_size*3 : chunk_size*3 + num_task_tokens*3, :]
image_embs = hidden_states[:, :chunk_size*3, :]
align_embs = torch.cat([image_embs, align_embs], dim=1)
depth_preds = self.depth_align_head(align_embs, depth_preds)

# Loss: Smooth L1
loss = F.smooth_l1_loss(depth_preds, depth_targets.detach())
```

### 9.5 Depth 配置

```yaml
align_params:
  mode: 'query'
  num_task_tokens: 8
  depth_loss_weight: 0.004
  depth:
    model_type: MoRGBD
    num_layers: 1
    num_heads: 4
    dim_head: 32
    num_backbone_tokens: 256
    token_size: 16
    dim_out: 1024
```

---

## 10. Flex Attention 优化

> `flex_attention.py:32-148`

```python
def flex_attention_forward(query_states, key_states, value_states, attention_mask):
    block_size = 128

    # GQA 展开
    key_states = repeat(key_states, "b l h d -> b l (h g) d", g=num_kv_groups)
    value_states = repeat(value_states, "b l h d -> b l (h g) d", g=num_kv_groups)

    # Pad 到 block_size 的倍数
    q_len_rounded = round_up(q_len, block_size)
    kv_len_rounded = round_up(kv_len, block_size)

    # 创建 block mask
    block_mask = create_block_mask(mask_mod_fn, B, H, q_len_rounded, kv_len_rounded, block_size)

    # Flex attention (torch 2.5+)
    attn_output = flex_attention(query, key, value, block_mask=block_mask, enable_gqa=True)
```

**优势**: 利用 PyTorch 2.5+ 的 `torch.nn.attention.flex_attention`，支持任意注意力掩码的高效计算，避免了 flash attention 对 causal mask 的限制。

---

## 11. 完整训练 Forward Pass

```
输入: images [B, 3, C, H, W], lang_tokens [B, L], state [B, max_state_dim],
      actions [B, n_action_steps, max_action_dim]

1. 采样噪声和时间
   noise ~ N(0, 1)
   time ~ Beta(1.5, 1.0) * 0.999 + 0.001
   x_t = time * noise + (1-time) * actions
   u_t = noise - actions

2. 编码 Prefix (VLM 输入)
   images → Qwen2.5-VL ViT → spatial_merge → [B, n*patch, 2048]
   lang_tokens → embed_tokens → [B, L, 2048]
   (可选) depth_align_embs → [B, 3*num_task_tokens, 2048]
   prefix = cat([img_emb, (align_embs), lang_emb])

3. 编码 Suffix (Action Expert 输入)
   state → state_proj → [B, 1, 768]
   x_t → action_in_proj → [B, n_action_steps, 768]
   time → sinusoidal_pos_embedding → [B, 768]
   action_time = cat([action_emb, time_emb], dim=-1) → MLP → [B, n_action_steps, 768]
   suffix = cat([state, action_time])

4. 构建注意力掩码
   cumsum 机制: prefix 双向, state 独立 block, action 双向
   (可选) depth align tokens 特殊掩码

5. MoT Forward (36 层)
   每层:
     a. VLM: RMSNorm → Q,K,V (独立)
     b. Expert: AdaRMSNorm(time) → Q,K,V,gate (独立)
     c. 拼接 Q,K,V → RoPE → 共享 Attention
     d. 拆分 → VLM: o_proj + MLP (独立)
                Expert: o_proj * gate + MLP (独立)

6. 计算损失
   v_t = action_out_proj(suffix_out[:, -n_action_steps:])
   fm_loss = MSE(v_t, u_t) 或 L1(v_t, u_t)
   (可选) depth_loss = smooth_L1(depth_pred, depth_target) * 0.004
   total_loss = fm_loss + depth_loss
```

---

## 12. 关键超参数

| 参数 | 值 | 来源 |
|------|-----|------|
| VLM hidden | 2048 | Qwen2.5-VL |
| VLM layers | 36 | Qwen2.5-VL |
| Expert hidden | 768 | Qwen2 config |
| Expert layers | 36 | Qwen2 config |
| Expert intermediate | 2752 | Qwen2 config |
| GQA | 16 query / 2 kv heads | 两者相同 |
| head_dim | 128 | 两者相同 |
| max_action_dim | 75 | PI0Config |
| action_dim | 14 | 实际机器人 |
| n_action_steps | 50 | chunk size |
| max_state_dim | 75 | PI0Config |
| 噪声分布 | Beta(1.5, 1.0) | sample_time() |
| 推理步数 | 10 (默认) | config.num_steps |
| depth_loss_weight | 0.004 | align_params |
| num_task_tokens | 8 | align_params |

---

## 13. 三模型对比总结

| 维度 | DM0 | DreamZero | LingBot |
|------|-----|-----------|---------|
| **架构范式** | VLM + Expert (Merged Attn) | 统一 DiT (视频+动作) | VLM + Expert (MoT) |
| **VLM** | Qwen3-1.7B | 无 (UMT5+CLIP) | Qwen2.5-VL |
| **Expert** | Qwen3 (28L, action_hidden) | 同一 DiT | Qwen2 (36L, 768D) |
| **总参数** | ~2B | 14B | ~4B |
| **融合方式** | 共享 KV + 独立 MLP | 同一 transformer | 共享 Attn + 独立 MLP |
| **Timestep** | concat + MLP | DiT modulation (6p/block) | AdaRMSNorm (FiLM) |
| **Gate** | 无 | DiT gate (modulation) | split_gate_liner |
| **视频预测** | 无 | 有 (核心) | 无 |
| **深度融合** | 无 | 无 | LingBot-Depth |
| **注意力掩码** | cumsum (suffix 双向) | blockwise causal flash | cumsum (state 独立) |
| **训练策略** | 从头训练 | LoRA 微调 | 冻结 VLM |
| **推理优化** | KV Cache | DiT Cache + CFG | KV Cache + Flex Attn |
| **Action dim** | 32 | 32 | 75 (padded) |
| **Chunk size** | 50 | 32 | 50 |

