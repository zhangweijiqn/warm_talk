"""
模型管理器 - 负责加载和运行中文对话模型
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import logging
from typing import Optional, List, Dict, Tuple
from config.model_config import (
    MODEL_NAME,
    GENERATION_CONFIG,
    USE_QUANTIZATION,
    QUANTIZATION_BITS,
    AUTO_ADJUST_FOR_CPU,
    CPU_MAX_NEW_TOKENS_LIMIT,
    CPU_SMALL_MODEL_MAX_TOKENS,
)
import threading
import time
from config.server_config import LOG_LEVEL

# 根据配置设置日志级别
log_level = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器，支持加载和推理"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_progress = None  # 存储当前生成进度信息
        logger.info(f"使用设备: {self.device}")
    
    def _estimate_model_size(self) -> float:
        """估算模型大小（B = 十亿参数）"""
        model_name_lower = self.model_name.lower()
        # 提取模型大小信息
        if "0.5b" in model_name_lower or "0.5-b" in model_name_lower:
            return 0.5
        elif "1.5b" in model_name_lower or "1.5-b" in model_name_lower or "1.8b" in model_name_lower:
            return 1.5
        elif "3b" in model_name_lower or "3-b" in model_name_lower:
            return 3.0
        elif "6b" in model_name_lower or "6-b" in model_name_lower:
            return 6.0
        elif "7b" in model_name_lower or "7-b" in model_name_lower:
            return 7.0
        elif "14b" in model_name_lower or "14-b" in model_name_lower:
            return 14.0
        elif "32b" in model_name_lower or "32-b" in model_name_lower:
            return 32.0
        else:
            # 默认假设是中等模型
            return 6.0
    
    def load_model(self):
        """加载模型和分词器"""
        import time
        try:
            start_time = time.time()
            logger.info("=" * 60)
            logger.info(f"开始加载模型: {self.model_name}")
            logger.info(f"使用设备: {self.device}")
            logger.info(f"CUDA 可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA 设备数量: {torch.cuda.device_count()}")
                logger.info(f"当前 CUDA 设备: {torch.cuda.current_device()}")
                logger.info(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")
            
            # 加载分词器
            logger.info("正在加载分词器...")
            tokenizer_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            tokenizer_time = time.time() - tokenizer_start
            logger.info(f"分词器加载完成，耗时: {tokenizer_time:.2f} 秒")
            
            # 加载模型
            logger.info("正在加载模型（这可能需要几分钟）...")
            model_start = time.time()
            
            # 准备量化配置
            quantization_config = None
            if USE_QUANTIZATION:
                if self.device == "cuda":
                    # GPU 模式：尝试使用 bitsandbytes 进行量化
                    try:
                        from transformers import BitsAndBytesConfig
                        if QUANTIZATION_BITS == 4:
                            logger.info("使用 4-bit 量化（需要 bitsandbytes）...")
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                        elif QUANTIZATION_BITS == 8:
                            logger.info("使用 8-bit 量化（需要 bitsandbytes）...")
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True
                            )
                    except ImportError:
                        logger.warning("⚠️ bitsandbytes 未安装，无法使用 GPU 量化。安装命令: pip install bitsandbytes")
                        quantization_config = None
                else:
                    # CPU 模式：使用动态量化（不需要额外库）
                    logger.info(f"CPU 模式：将在加载后应用 {QUANTIZATION_BITS}-bit 动态量化...")
            
            # 根据模型类型选择不同的加载方式
            if "chatglm" in self.model_name.lower():
                # ChatGLM 系列使用 AutoModel
                logger.info("检测到 ChatGLM 模型，使用 AutoModel")
                load_kwargs = {
                    "trust_remote_code": True,
                }
                
                if quantization_config:
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                    load_kwargs["device_map"] = "auto" if self.device == "cuda" else None
                
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **load_kwargs
                )
            else:
                # 其他模型使用 AutoModelForCausalLM
                logger.info("使用 AutoModelForCausalLM")
                load_kwargs = {
                    "trust_remote_code": True,
                }
                
                if quantization_config:
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                    load_kwargs["device_map"] = "auto" if self.device == "cuda" else None
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_kwargs
                )
            
            model_time = time.time() - model_start
            logger.info(f"模型加载完成，耗时: {model_time:.2f} 秒")
            
            if self.device == "cpu":
                logger.info("将模型移动到 CPU...")
                self.model = self.model.to(self.device)
            
            logger.info("设置模型为评估模式...")
            self.model.eval()
            
            # CPU 模式的动态量化（如果启用且未使用 bitsandbytes）
            if self.device == "cpu" and USE_QUANTIZATION and quantization_config is None:
                logger.info(f"应用 {QUANTIZATION_BITS}-bit 动态量化（CPU 模式）...")
                try:
                    if QUANTIZATION_BITS == 8:
                        # 8-bit 动态量化
                        self.model = torch.quantization.quantize_dynamic(
                            self.model,
                            {torch.nn.Linear},  # 量化线性层
                            dtype=torch.qint8
                        )
                        logger.info("✅ 8-bit 动态量化完成")
                    elif QUANTIZATION_BITS == 4:
                        # 4-bit 在 CPU 上不支持动态量化，回退到 8-bit
                        logger.warning("⚠️ CPU 模式不支持 4-bit 量化，自动使用 8-bit")
                        self.model = torch.quantization.quantize_dynamic(
                            self.model,
                            {torch.nn.Linear},
                            dtype=torch.qint8
                        )
                        logger.info("✅ 8-bit 动态量化完成（4-bit 回退）")
                except Exception as e:
                    logger.warning(f"⚠️ 量化失败: {str(e)}，使用原始模型")
            
            # 确保模型在正确的设备上
            if self.device == "cpu" and not USE_QUANTIZATION:
                logger.info("确保模型在 CPU 上...")
                self.model = self.model.cpu()
                # 对于 CPU，使用 float32 可能更稳定
                if hasattr(self.model, 'half'):
                    logger.info("CPU 模式：使用 float32 精度")
            
            # 检查模型状态
            logger.info(f"模型训练模式: {self.model.training}")
            if hasattr(self.model, 'device'):
                logger.info(f"模型所在设备: {self.model.device}")
            
            total_time = time.time() - start_time
            logger.info(f"模型加载总耗时: {total_time:.2f} 秒")
            logger.info("=" * 60)
            
        except Exception as e:
            import traceback
            logger.error(f"模型加载失败: {str(e)}")
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        history: Optional[List[Tuple[str, str]]] = None,
        use_chain_of_thought: bool = True,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        生成回复
        
        Args:
            prompt: 用户输入或完整提示词
            history: 对话历史 [(用户消息, 助手回复), ...]
            use_chain_of_thought: 是否使用思考链路
            system_prompt: 系统提示词
            
        Returns:
            模型生成的回复
        """
        import time
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        try:
            logger.debug(f"[模型推理] 开始生成回复...")
            logger.debug(f"[模型推理] 提示词长度: {len(prompt)} 字符")
            logger.debug(f"[模型推理] 历史对话轮数: {len(history) if history else 0}")
            
            # 输出对话历史详情
            if history:
                logger.debug("=" * 60)
                logger.debug("[模型推理] 【对话历史详情】")
                for i, (user_msg, assistant_msg) in enumerate(history, 1):
                    logger.debug(f"  第 {i} 轮:")
                    logger.debug(f"    用户: {user_msg[:150]}..." if len(user_msg) > 150 else f"    用户: {user_msg}")
                    if assistant_msg:
                        logger.debug(f"    助手: {assistant_msg[:150]}..." if len(assistant_msg) > 150 else f"    助手: {assistant_msg}")
                    else:
                        logger.debug(f"    助手: (未保存)")
                logger.debug("=" * 60)
            
            # 构建完整的输入
            if "chatglm" in self.model_name.lower():
                # ChatGLM 系列
                logger.debug("[模型推理] 使用 ChatGLM 推理模式")
                # 如果有系统提示词，需要特殊处理
                if system_prompt:
                    # 将系统提示词添加到第一轮对话
                    full_prompt = f"{system_prompt}\n\n用户：{prompt}\n助手："
                    logger.debug("[模型推理] 已添加系统提示词")
                else:
                    full_prompt = prompt
                
                logger.debug("=" * 60)
                logger.debug("[模型推理] 【当前 Prompt（不包含历史对话）】")
                logger.debug(full_prompt)
                if history:
                    logger.debug(f"[模型推理] 注意：历史对话将通过 history 参数传递（{len(history)} 轮）")
                    logger.debug("[模型推理] ChatGLM 模型会将历史对话与当前 prompt 合并处理")
                logger.debug("=" * 60)
                
                # 根据设备类型优化生成参数
                chat_config = GENERATION_CONFIG.copy()
                # 确保不包含 max_length，避免与 max_new_tokens 冲突
                chat_config.pop("max_length", None)
                
                if self.device == "cpu" and AUTO_ADJUST_FOR_CPU:
                    # CPU 上根据模型大小调整生成长度
                    # 小模型（<3B）可以使用更多 tokens，大模型需要减少
                    model_size = self._estimate_model_size()
                    if model_size < 3:
                        # 小模型：可以使用更多 tokens
                        chat_config["max_new_tokens"] = min(
                            chat_config.get("max_new_tokens", 512),
                            CPU_SMALL_MODEL_MAX_TOKENS
                        )
                        logger.info(f"[模型推理] 小模型（{model_size}B），CPU 模式 max_new_tokens 设为 {chat_config['max_new_tokens']}")
                    else:
                        # 大模型：大幅减少
                        chat_config["max_new_tokens"] = min(
                            chat_config.get("max_new_tokens", 256),
                            CPU_MAX_NEW_TOKENS_LIMIT
                        )
                        logger.warning(f"[模型推理] ⚠️ 大模型（{model_size}B），CPU 模式：已调整 max_new_tokens 为 {chat_config['max_new_tokens']}（建议使用 GPU 或更小的模型）")
                
                logger.debug(f"[模型推理] 准备调用 model.chat()...")
                logger.debug(f"[模型推理] 历史对话轮数: {len(history) if history else 0}")
                logger.debug(f"[模型推理] 生成参数: {chat_config}")
                logger.debug(f"[模型推理] 使用设备: {self.device}")
                
                # 检查模型状态
                if hasattr(self.model, 'training'):
                    logger.debug(f"[模型推理] 模型训练模式: {self.model.training}")
                
                inference_start = time.time()
                max_new_tokens = chat_config.get("max_new_tokens", 1024)
                logger.debug(f"[模型推理] 开始调用 model.chat()（最大生成 {max_new_tokens} tokens）...")
                
                # 进度统计
                progress_info = {
                    "start_time": inference_start,
                    "last_update_time": inference_start,
                    "last_token_count": 0,
                    "current_token_count": 0,
                    "max_new_tokens": max_new_tokens,
                    "generated_text": "",
                    "tokens_per_second": 0.0,
                    "estimated_remaining": 0.0,
                    "progress_percent": 0.0,
                    "elapsed_time": 0.0,
                    "is_generating": True
                }
                # 存储到实例变量，供外部查询
                self.current_progress = progress_info
                
                # 添加进度监控线程
                progress_stop = threading.Event()
                def progress_monitor():
                    """定期输出进度日志，包括速度和预估时间"""
                    update_interval = 5  # 每5秒更新一次
                    while not progress_stop.is_set():
                        time.sleep(update_interval)
                        if not progress_stop.is_set():
                            elapsed = time.time() - progress_info["start_time"]
                            current_tokens = progress_info["current_token_count"]
                            
                            # 计算生成速度
                            if elapsed > 0:
                                progress_info["tokens_per_second"] = current_tokens / elapsed
                            
                            # 预估剩余时间
                            if progress_info["tokens_per_second"] > 0 and max_new_tokens > current_tokens:
                                remaining_tokens = max_new_tokens - current_tokens
                                progress_info["estimated_remaining"] = remaining_tokens / progress_info["tokens_per_second"]
                                remaining_str = f"预计剩余 {progress_info['estimated_remaining']:.1f} 秒"
                            else:
                                remaining_str = "计算中..."
                            
                            # 计算进度百分比
                            progress_pct = min(100, (current_tokens / max_new_tokens) * 100) if max_new_tokens > 0 else 0
                            progress_info["progress_percent"] = progress_pct
                            progress_info["elapsed_time"] = elapsed
                            
                            logger.debug(
                                f"[模型推理] ⏳ 进度: {progress_pct:.1f}% | "
                                f"已生成: {current_tokens}/{max_new_tokens} tokens | "
                                f"速度: {progress_info['tokens_per_second']:.2f} tokens/秒 | "
                                f"已用时: {elapsed:.1f}秒 | {remaining_str}"
                            )
                
                progress_thread = threading.Thread(target=progress_monitor, daemon=True)
                progress_thread.start()
                
                try:
                    # 尝试使用流式生成以获取进度
                    use_streaming = hasattr(self.model, 'stream_chat')
                    
                    if use_streaming:
                        logger.debug("[模型推理] 使用流式生成模式（可显示实时进度）")
                        last_response = ""
                        
                        # 抑制 ChatGLM 内部的警告（其日志格式化有 bug）
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            
                            with torch.no_grad():
                                # ChatGLM: 如果有历史对话，使用 prompt（不包含系统提示词），否则使用 full_prompt
                                # history 参数会将历史对话与当前 prompt 合并
                                chat_prompt = prompt if history else full_prompt
                                logger.debug(f"[模型推理] ChatGLM stream_chat: prompt长度={len(chat_prompt)}, history轮数={len(history) if history else 0}")
                                for response, _ in self.model.stream_chat(
                                    self.tokenizer,
                                    chat_prompt,
                                    history=history or [],
                                    **chat_config
                                ):
                                    # stream_chat 返回的是完整的累积文本，不是增量
                                    if response and response != last_response:
                                        # 计算新增的文本
                                        new_text = response[len(last_response):] if last_response else response
                                        
                                        # 使用tokenizer计算新增的token数
                                        if new_text:
                                            try:
                                                # 计算新增文本的token数
                                                new_tokens = self.tokenizer.encode(new_text, add_special_tokens=False)
                                                progress_info["current_token_count"] += len(new_tokens)
                                            except:
                                                # 如果tokenizer失败，使用字符数估算
                                                progress_info["current_token_count"] += int(len(new_text) / 2)
                                        
                                        progress_info["generated_text"] = response
                                        last_response = response
                        
                        response = last_response if last_response else ""
                    else:
                        # 回退到非流式生成
                        logger.debug("[模型推理] 使用标准生成模式（无法显示实时进度）")
                        # 抑制 ChatGLM 内部的警告（其日志格式化有 bug）
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            
                            with torch.no_grad():
                                # ChatGLM: 如果有历史对话，使用 prompt（不包含系统提示词），否则使用 full_prompt
                                # history 参数会将历史对话与当前 prompt 合并
                                chat_prompt = prompt if history else full_prompt
                                logger.debug(f"[模型推理] ChatGLM chat: prompt长度={len(chat_prompt)}, history轮数={len(history) if history else 0}")
                                response, _ = self.model.chat(
                                    self.tokenizer,
                                    chat_prompt,
                                    history=history or [],
                                    **chat_config
                                )
                        
                        # 估算生成的token数
                        try:
                            tokens = self.tokenizer.encode(response, add_special_tokens=False)
                            progress_info["current_token_count"] = len(tokens)
                        except:
                            progress_info["current_token_count"] = int(len(response) / 2)
                    
                    progress_stop.set()  # 停止进度监控
                    inference_time = time.time() - inference_start
                    
                    # 计算最终统计
                    final_tokens = progress_info["current_token_count"]
                    final_speed = final_tokens / inference_time if inference_time > 0 else 0
                    
                    logger.info(f"[模型推理] ✅ ChatGLM 推理完成，耗时: {inference_time:.2f} 秒")
                    logger.debug(f"[模型推理] 生成 tokens: {final_tokens}")
                    logger.debug(f"[模型推理] 平均速度: {final_speed:.2f} tokens/秒")
                    logger.debug(f"[模型推理] 生成回复长度: {len(response)} 字符")
                    logger.debug(f"[模型推理] 回复预览: {response[:200]}..." if len(response) > 200 else f"[模型推理] 回复: {response}")
                    return response
                except Exception as chat_error:
                    progress_stop.set()  # 停止进度监控
                    inference_time = time.time() - inference_start
                    logger.error(f"[模型推理] ❌ model.chat() 调用失败，耗时: {inference_time:.2f} 秒")
                    logger.error(f"[模型推理] 错误信息: {str(chat_error)}")
                    import traceback
                    logger.error(f"[模型推理] 错误堆栈:\n{traceback.format_exc()}")
                    raise
            else:
                # 其他模型（如 Qwen, Baichuan）
                logger.debug("[模型推理] 使用标准模型推理模式")
                # 构建对话格式
                messages = []
                
                # 添加系统提示词
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                    logger.debug("[模型推理] 已添加系统提示词")
                
                # 添加历史对话
                if history:
                    logger.debug(f"[模型推理] 添加 {len(history)} 轮历史对话")
                    logger.debug("=" * 60)
                    logger.debug("[模型推理] 【添加到 Messages 的对话历史】")
                    for i, (user_msg, assistant_msg) in enumerate(history, 1):
                        logger.debug(f"  第 {i} 轮:")
                        logger.debug(f"    user: {user_msg[:150]}..." if len(user_msg) > 150 else f"    user: {user_msg}")
                        if assistant_msg:
                            logger.debug(f"    assistant: {assistant_msg[:150]}..." if len(assistant_msg) > 150 else f"    assistant: {assistant_msg}")
                        else:
                            logger.debug(f"    assistant: (空)")
                        messages.append({"role": "user", "content": user_msg})
                        messages.append({"role": "assistant", "content": assistant_msg})
                    logger.debug("=" * 60)
                
                # 添加当前用户输入
                messages.append({"role": "user", "content": prompt})
                
                logger.debug("=" * 60)
                logger.debug("[模型推理] 【最终发送给模型的完整 Messages（包含历史对话）】")
                logger.debug(f"消息总数: {len(messages)}")
                for i, msg in enumerate(messages, 1):
                    content_preview = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                    logger.debug(f"[消息 {i}] {msg['role']}: {content_preview}")
                logger.debug("=" * 60)
                
                # 应用对话模板
                logger.debug("[模型推理] 应用对话模板...")
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                logger.debug(f"[模型推理] 模板文本长度: {len(text)} 字符")
                logger.debug("=" * 60)
                logger.debug("[模型推理] 【应用模板后的完整 Prompt（包含历史对话）】")
                logger.debug(f"Prompt 长度: {len(text)} 字符")
                logger.debug(text)
                logger.debug("=" * 60)
                
                # 编码
                logger.debug("[模型推理] 编码输入...")
                encode_start = time.time()
                inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                encode_time = time.time() - encode_start
                logger.debug(f"[模型推理] 编码完成，耗时: {encode_time:.2f} 秒，输入长度: {inputs.shape[1]} tokens")
                
                # 生成
                logger.debug("[模型推理] 开始生成（这可能需要一些时间）...")
                logger.debug(f"[模型推理] 生成参数: {GENERATION_CONFIG}")
                gen_start = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        **GENERATION_CONFIG
                    )
                gen_time = time.time() - gen_start
                logger.info(f"[模型推理] ✅ 生成完成，耗时: {gen_time:.2f} 秒")
                logger.debug(f"[模型推理] 输出 tokens 数量: {outputs.shape[1] - inputs.shape[1]}")
                
                # 解码
                logger.debug("[模型推理] 解码输出...")
                decode_start = time.time()
                response = self.tokenizer.decode(
                    outputs[0][inputs.shape[1]:],
                    skip_special_tokens=True
                )
                decode_time = time.time() - decode_start
                logger.debug(f"[模型推理] 解码完成，耗时: {decode_time:.2f} 秒")
                logger.debug(f"[模型推理] 最终回复长度: {len(response)} 字符")
                
                # 生成完成后清除进度信息
                if self.current_progress:
                    self.current_progress["is_generating"] = False
                
                return response.strip()
                
        except Exception as e:
            # 出错时也清除进度信息
            if self.current_progress:
                self.current_progress["is_generating"] = False
            import traceback
            logger.error(f"[模型推理] 生成回复时出错: {str(e)}")
            logger.error(f"[模型推理] 错误堆栈:\n{traceback.format_exc()}")
            return f"抱歉，生成回复时出现错误: {str(e)}"
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.tokenizer is not None

