"""
模型配置文件 - 用于切换模型和调整参数
"""
import os

# ==================== 模型选择 ====================
# 注意：在 CPU 上运行，建议使用更小的模型或量化模型以提高速度
# 如果使用 GPU，可以使用更大的模型

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-0.5B-Instruct")  # 默认使用 Qwen2-0.5B（非常小，适合CPU）

# 其他可选模型（按参数大小和速度从快到慢）：
# === 超小模型（推荐用于CPU，速度快）===
# "Qwen/Qwen2-0.5B-Instruct" - Qwen2 0.5B（推荐，最小最快，兼容旧版本 transformers）
# "Qwen/Qwen2-1.5B-Instruct" - Qwen2 1.5B（速度快）
# "Qwen/Qwen2-3B-Instruct" - Qwen2 3B（平衡）
# "Qwen/Qwen2.5-0.5B-Instruct" - Qwen2.5 0.5B（需要 transformers>=4.37）
# "Qwen/Qwen2.5-1.5B-Instruct" - Qwen2.5 1.5B（需要 transformers>=4.37）
# "Qwen/Qwen2.5-3B-Instruct" - Qwen2.5 3B（需要 transformers>=4.37）
# "THUDM/chatglm2-6b" - ChatGLM2 6B（中等速度）
# === 中等模型（需要较好CPU或GPU）===
# "THUDM/chatglm3-6b" - ChatGLM3 6B（功能更强但更慢）
# "Qwen/Qwen2.5-7B-Instruct" - Qwen2.5 7B（需要GPU）
# === 大模型（需要GPU）===
# "Qwen/Qwen2.5-14B-Instruct" - Qwen2.5 14B（需要GPU）
# "baichuan-inc/Baichuan2-7B-Chat" - 百川 7B（需要GPU）


# ==================== 量化配置 ====================
# 量化可以大幅减少内存占用和提高速度（特别是CPU模式）

USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"  # 是否使用量化
QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "8"))  # 量化位数：4 或 8（8-bit 更稳定，4-bit 更快但可能损失精度）
# 注意：4-bit 量化需要 GPU 和 bitsandbytes，CPU 模式建议使用 8-bit 动态量化


# ==================== 生成参数 ====================
# 小模型（如 0.5B-3B）可以生成更多 tokens，大模型建议减少

GENERATION_CONFIG = {
    "max_new_tokens": 512,  # 小模型可以使用 512，大模型建议 256
    "temperature": 0.8,     # 平衡情感表达和稳定性（0.8 既有人情味又不会太随机）
    "top_p": 0.9,           # 保持表达的丰富性，同时略微减少随机性
    "do_sample": True,      # 使用采样以产生更自然的表达
    "repetition_penalty": 1.2,  # 提高重复惩罚，避免机械化重复，让表达更自然
}

# 高级生成参数（可选，取消注释以使用）
# GENERATION_CONFIG.update({
#     "top_k": 50,              # Top-K 采样：只考虑概率最高的 k 个 tokens
#     "num_beams": 1,           # Beam search 的 beam 数量（>1 启用 beam search）
#     "length_penalty": 1.0,    # 长度惩罚：>1.0 鼓励更长输出，<1.0 鼓励更短输出
#     "early_stopping": False,  # 是否早停（beam search 时）
# })


# ==================== 对话配置 ====================

MAX_HISTORY_LENGTH = 10  # 最大历史对话轮数（保留多少轮历史对话）
MAX_CONTEXT_LENGTH = 2048  # 最大上下文长度（tokens）
SAVE_ONLY_USER_MESSAGES = os.getenv("SAVE_ONLY_USER_MESSAGES", "true").lower() == "true"  # 是否仅保存用户问题（不保存助手回复）


# ==================== 设备优化配置 ====================

# CPU 模式下的自动调整（如果为 True，会根据设备自动调整参数）
AUTO_ADJUST_FOR_CPU = True

# CPU 模式下的 max_new_tokens 限制（如果 AUTO_ADJUST_FOR_CPU 为 True）
CPU_MAX_NEW_TOKENS_LIMIT = 128  # 大模型在 CPU 上的限制
CPU_SMALL_MODEL_MAX_TOKENS = 256  # 小模型（<3B）在 CPU 上的限制

# GPU 模式下的默认配置
GPU_MAX_NEW_TOKENS = 1024  # GPU 模式下可以使用更多 tokens


# ==================== 模型加载配置 ====================

# 模型加载时的数据类型
MODEL_DTYPE = "auto"  # "auto", "float16", "float32" - auto 会根据设备自动选择

# 是否使用 trust_remote_code（某些模型需要）
TRUST_REMOTE_CODE = True

# 是否使用 device_map="auto"（GPU 模式下的自动设备映射）
USE_AUTO_DEVICE_MAP = True

