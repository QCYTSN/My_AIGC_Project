import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from typing import Optional, Union, List, Dict, Any

class MaskedCrossAttentionProcessor:
    """
    ⚔️ [终极修复版] 带 FP32 安全锁 + 维度修正的 Mask Attention 处理器
    """
    def __init__(self, token_idx_to_mask: Dict[int, torch.Tensor], scale: float = 1.0):
        self.token_idx_to_mask = token_idx_to_mask
        self.scale = scale

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # === 1. 获取 Q, K, V ===
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 转换到多头维度 [Batch*Heads, SeqLen, Dim]
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # === 🚨 关键修复区 ===
        # 1. 转 FP32 防止溢出
        original_dtype = query.dtype
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        # 2. 计算 Attention Scores
        # ⚠️ 之前的错误：这里多包了一层 attn.batch_to_head_dim，导致维度被压缩，现已删除
        attention_scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=attn.scale,
        )
        
        # === 2. 注入 Mask 控制 ===
        current_res = int(sequence_length ** 0.5) 
        
        for token_idx, mask in self.token_idx_to_mask.items():
            # 缩放 Mask
            resized_mask = F.interpolate(mask, size=(current_res, current_res), mode="nearest")
            flat_mask = resized_mask.reshape(1, -1)
            
            # 制造惩罚项 (Mask 为 0 的地方扣 10000 分)
            # FP32 下非常安全
            penalty = (1 - flat_mask) * -50.0
            
            # 施加惩罚
            attention_scores[:, :, token_idx] = attention_scores[:, :, token_idx] + penalty.to(attention_scores.device)

        # === 3. 收尾 ===
        # Softmax (FP32)
        attention_probs = attention_scores.softmax(dim=-1)
        
        # 转回 FP16 (如果原模型是FP16)
        attention_probs = attention_probs.to(original_dtype)
        
        # 计算输出
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # 线性投射输出
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class MaskStableDiffusionPipeline(StableDiffusionPipeline):
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        mask_config: Dict[str, torch.Tensor] = None, 
        **kwargs,
    ):
        if mask_config is not None:
            print(f">>> 🛡️ 正在注入 Mask 控制: {list(mask_config.keys())}")
            self.register_attention_control(prompt, mask_config)
        
        return super().__call__(prompt=prompt, **kwargs)

    def register_attention_control(self, prompt, mask_config):
        # 获取 Token ID
        input_ids = self.tokenizer(prompt).input_ids
        token_idx_to_mask = {}
        
        print(">>> 🔍 Token 映射:")
        words = prompt.split()
        for word, mask in mask_config.items():
            try:
                found = False
                for i, token_id in enumerate(input_ids):
                    decoded_word = self.tokenizer.decode([token_id]).strip()
                    # 简单匹配逻辑
                    if word in decoded_word:
                        print(f"    - '{word}' 对应 Token ID: {token_id} (位置 {i})")
                        token_idx_to_mask[i] = mask.to(self.device, dtype=torch.float32)
                        found = True
                if not found:
                     print(f"⚠️ 警告: Prompt 中未找到单词 '{word}' 的 Token")
            except ValueError:
                pass

        # 替换处理器
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            if name.endswith("attn2.processor"): 
                attn_procs[name] = MaskedCrossAttentionProcessor(token_idx_to_mask)
            else:
                attn_procs[name] = AttnProcessor2_0()
        
        self.unet.set_attn_processor(attn_procs)