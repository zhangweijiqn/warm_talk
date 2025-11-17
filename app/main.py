"""
主应用 - FastAPI 后端服务
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
import logging
import uuid
import os
import sys
import json
import asyncio

from app.model_manager import ModelManager
from app.conversation_manager import ConversationManager
from app.prompt_builder import PromptBuilder
from config import HOST, PORT, HEALTH_CHECK_INTERVAL, LOG_LEVEL
import re

# 根据配置设置日志级别
log_level = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info(f"日志级别设置为: {LOG_LEVEL}")


def extract_thinking(response: str) -> Optional[str]:
    """
    从模型回复中提取思考过程
    
    支持多种格式：
    1. **思考过程：** ... **正式回答：**
    2. 思考过程： ... 正式回答：
    3. 思考： ... 回答：
    4. [思考] ... [回答]
    5. 其他变体
    """
    if not response:
        return None
    
    # 模式1: **思考过程：** ... **正式回答：**（最标准格式）
    # 使用更精确的匹配，确保不会包含正式回答的内容
    pattern1 = r'\*\*思考过程：?\*\*[\s\n]*(.*?)(?=[\s\n]*\*\*正式回答：?\*\*)'
    match = re.search(pattern1, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        # 确保思考内容不包含"正式回答"标记
        if thinking and len(thinking) > 5 and '正式回答' not in thinking and '**正式回答' not in thinking:
            return thinking
    
    # 模式1.1: **思考过程** ... **正式回答**（无冒号变体）
    pattern1_1 = r'\*\*思考过程\*\*[\s\n]*(.*?)(?=[\s\n]*\*\*正式回答\*\*)'
    match = re.search(pattern1_1, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking and len(thinking) > 5 and '正式回答' not in thinking:
            return thinking
    
    # 模式2: 思考过程： ... 正式回答：（无星号格式）
    pattern2 = r'思考过程：?[\s\n]+(.*?)(?=[\s\n]+正式回答：?)'
    match = re.search(pattern2, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking and len(thinking) > 5 and '正式回答' not in thinking:
            return thinking
    
    # 模式2.1: 思考过程 ... 正式回答（无冒号）
    pattern2_1 = r'思考过程[\s\n]+(.*?)(?=[\s\n]+正式回答)'
    match = re.search(pattern2_1, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking and len(thinking) > 5 and '正式回答' not in thinking:
            return thinking
    
    # 模式3: 思考： ... 回答：（简化格式）
    pattern3 = r'思考：?[\s\n]+(.*?)[\s\n]+回答：?'
    match = re.search(pattern3, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking and len(thinking) > 5:
            return thinking
    
    # 模式3.1: 思考 ... 回答（无冒号）
    pattern3_1 = r'思考[\s\n]+(.*?)[\s\n]+回答'
    match = re.search(pattern3_1, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking and len(thinking) > 5:
            return thinking
    
    # 模式4: [思考] ... [回答] 或 [思考过程] ... [正式回答]
    pattern4 = r'\[思考过程?\]?[\s\n]*(.*?)[\s\n]*\[正式回答?\]?'
    match = re.search(pattern4, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking:
            return thinking
    
    # 模式5: 尝试查找包含"思考"和"回答"的段落
    # 如果回复中包含明显的思考段落（通常在前半部分）
    lines = response.split('\n')
    thinking_lines = []
    found_thinking_marker = False
    found_answer_marker = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # 查找思考标记
        if any(marker in line_lower for marker in ['思考过程', '思考：', '**思考', '[思考']):
            found_thinking_marker = True
            # 提取标记后的内容
            for marker in ['思考过程：', '思考：', '**思考过程：**', '**思考：**', '[思考过程]', '[思考]']:
                if marker in line:
                    thinking_content = line.split(marker, 1)[-1].strip()
                    if thinking_content:
                        thinking_lines.append(thinking_content)
                    break
            continue
        
        # 如果找到了思考标记，继续收集直到找到回答标记
        if found_thinking_marker and not found_answer_marker:
            if any(marker in line_lower for marker in ['正式回答', '回答：', '**正式回答', '[回答', '**回答']):
                found_answer_marker = True
                break
            if line.strip():
                thinking_lines.append(line)
    
    if thinking_lines:
        thinking = '\n'.join(thinking_lines).strip()
        if thinking and len(thinking) > 10:  # 确保不是太短
            return thinking
    
    return None


def extract_thinking_relaxed(response: str) -> Optional[str]:
    """
    更宽松的思考过程提取（当标准格式提取失败时使用）
    尝试识别任何包含"思考"、"分析"、"理解"等关键词的段落
    """
    if not response:
        return None
    
    lines = response.split('\n')
    thinking_lines = []
    found_thinking_keywords = False
    
    # 查找包含思考相关关键词的行
    thinking_keywords = ['思考', '分析', '理解', '感受', '考虑', '觉得', '认为', '意识到', '注意到']
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # 检查是否包含思考相关关键词
        if any(keyword in line_lower for keyword in thinking_keywords):
            # 如果这行看起来像是思考内容（不是回答）
            if not any(marker in line_lower for marker in ['回答', '回复', '建议', '可以', '应该']):
                found_thinking_keywords = True
                thinking_lines.append(line.strip())
        elif found_thinking_keywords:
            # 如果已经找到思考内容，继续收集直到遇到明显的回答标记
            if any(marker in line_lower for marker in ['回答', '回复', '所以', '因此', '建议', '可以这样']):
                break
            if line.strip() and len(line.strip()) > 5:  # 忽略太短的行
                thinking_lines.append(line.strip())
    
    if thinking_lines:
        thinking = '\n'.join(thinking_lines).strip()
        # 确保思考内容足够长且有意义
        if thinking and len(thinking) > 20:
            return thinking
    
    # 如果回复的前30%包含思考关键词，可能整个回复都是思考过程
    # 这种情况下，模型可能没有明确区分思考和回答
    first_part = response[:len(response)//3]
    if any(keyword in first_part.lower() for keyword in thinking_keywords):
        # 检查是否包含明显的回答标记
        if not any(marker in response.lower() for marker in ['正式回答', '回答：', '**回答', '[回答']):
            # 可能整个回复都是思考过程，返回前一部分
            return first_part.strip()
    
    return None


def remove_thinking_from_response(response: str, thinking: str) -> str:
    """
    从回复中移除思考部分，只保留正式回答
    改进版：更准确地识别边界，避免截断句子
    """
    if not thinking:
        return response
    
    # 方法1: 尝试通过标记找到正式回答的开始位置
    answer_markers = [
        r'\*\*正式回答：?\*\*',
        r'正式回答：?',
        r'\*\*回答：?\*\*',
        r'回答：?',
    ]
    
    answer_start_pos = -1
    for marker in answer_markers:
        match = re.search(marker, response, re.IGNORECASE)
        if match:
            answer_start_pos = match.end()
            break
    
    # 方法2: 如果找到了正式回答标记，从标记后开始提取
    if answer_start_pos > 0:
        answer = response[answer_start_pos:].strip()
        # 清理开头的空行和标记残留
        answer = re.sub(r'^[\s\n]+', '', answer)
        # 只清理开头的句号，其他标点保留（可能是回答的一部分）
        answer = re.sub(r'^[。\s]+', '', answer)
        # 如果清理后为空或太短，尝试查找第一个中文字符或英文字母
        if not answer or len(answer) < 3:
            first_char_match = re.search(r'[A-Za-z\u4e00-\u9fa5]', response[answer_start_pos:])
            if first_char_match:
                answer = response[answer_start_pos + first_char_match.start():].strip()
        return answer
    
    # 方法3: 如果思考内容在回复中，找到它的结束位置
    thinking_pos = response.find(thinking)
    if thinking_pos >= 0:
        # 从思考内容结束后开始查找正式回答
        after_thinking = response[thinking_pos + len(thinking):].strip()
        
        # 查找正式回答标记
        for marker in answer_markers:
            match = re.search(marker, after_thinking, re.IGNORECASE)
            if match:
                answer = after_thinking[match.end():].strip()
                # 清理开头的空行
                answer = re.sub(r'^[\s\n]+', '', answer)
                # 只清理开头的句号，其他标点保留（可能是回答的一部分）
                answer = re.sub(r'^[。\s]+', '', answer)
                # 如果清理后为空或太短，尝试查找第一个中文字符或英文字母
                if not answer or len(answer) < 3:
                    first_char_match = re.search(r'[A-Za-z\u4e00-\u9fa5]', after_thinking[match.end():])
                    if first_char_match:
                        answer = after_thinking[match.end() + first_char_match.start():].strip()
                return answer
        
        # 如果没有找到标记，尝试找到第一个中文字符或英文字母开始的位置
        first_char_match = re.search(r'[A-Za-z\u4e00-\u9fa5]', after_thinking)
        if first_char_match:
            answer = after_thinking[first_char_match.start():].strip()
            # 如果答案太短，可能不是完整的回答，返回原回复
            if len(answer) < 10:
                return response
            return answer
    
    # 方法4: 如果以上都失败，尝试通过正则表达式移除思考部分
    patterns = [
        r'\*\*思考过程：?\*\*[\s\n]*.*?[\s\n]*\*\*正式回答：?\*\*',
        r'思考过程：?[\s\n]*.*?[\s\n]+正式回答：?',
        r'思考：?[\s\n]*.*?[\s\n]+回答：?',
        r'\[思考过程?\]?[\s\n]*.*?[\s\n]*\[正式回答?\]?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            # 找到匹配后，提取正式回答部分
            full_match = match.group(0)
            # 查找正式回答标记在匹配中的位置
            for marker in answer_markers:
                marker_match = re.search(marker, full_match, re.IGNORECASE)
                if marker_match:
                    answer = full_match[marker_match.end():].strip()
                    # 只清理开头的句号，其他标点保留（可能是回答的一部分）
                    answer = re.sub(r'^[。\s]+', '', answer)
                    # 如果清理后为空或太短，尝试查找第一个中文字符或英文字母
                    if not answer or len(answer) < 3:
                        first_char_match = re.search(r'[A-Za-z\u4e00-\u9fa5]', full_match[marker_match.end():])
                        if first_char_match:
                            answer = full_match[marker_match.end() + first_char_match.start():].strip()
                    # 替换原回复中的匹配部分为答案部分
                    response = response.replace(full_match, answer, 1)
                    break
    
    # 清理多余的空行和标记
    response = re.sub(r'\*\*正式回答：?\*\*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'正式回答：?', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # 清理多余空行
    
    # 最后只清理开头的句号，其他标点保留
    response = re.sub(r'^[。\s]+', '', response.strip())
    
    return response.strip()


# 初始化组件（需要在 lifespan 之前初始化）
model_manager = ModelManager()
conversation_manager = ConversationManager()

# 全局变量：模型加载状态
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理（启动和关闭）"""
    import traceback
    global model_loaded
    # 启动时
    try:
        logger.info("=" * 60)
        logger.info("正在启动应用...")
        logger.info(f"Python 版本: {sys.version}")
        logger.info(f"工作目录: {os.getcwd()}")
        model_manager.load_model()
        model_loaded = True
        logger.info("✅ 应用启动完成，模型已加载")
        logger.info("=" * 60)
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ 启动失败: {str(e)}")
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        logger.error("=" * 60)
        model_loaded = False
    
    yield
    
    # 关闭时（如果需要清理资源）
    logger.info("应用正在关闭...")


# 初始化 FastAPI 应用
app = FastAPI(title="WarmTalk", version="1.0.0", lifespan=lifespan)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求/响应模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_chain_of_thought: bool = True


class ChatResponse(BaseModel):
    response: str
    session_id: str
    thinking: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    
    class Config:
        protected_namespaces = ()  # 解决 Pydantic 命名空间冲突警告


class ConfigResponse(BaseModel):
    health_check_interval: int  # 健康检查间隔（毫秒）


class ProgressResponse(BaseModel):
    is_generating: bool
    progress_percent: float
    current_tokens: int
    max_tokens: int
    tokens_per_second: float
    elapsed_time: float
    estimated_remaining: float




@app.get("/")
async def root():
    """返回前端页面"""
    import os
    static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")
    return FileResponse(static_path)


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    import time
    check_time = time.strftime('%Y-%m-%d %H:%M:%S')
    status = "ok" if model_loaded else "error"
    model_name = model_manager.model_name if model_loaded else None
    
    logger.info(f"[健康检查] 时间: {check_time} | 状态: {status} | 模型已加载: {model_loaded} | 模型: {model_name}")
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_name=model_name
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    处理聊天请求
    
    Args:
        request: 聊天请求，包含用户消息和会话ID
        
    Returns:
        聊天响应，包含助手回复和会话ID
    """
    import time
    global model_loaded
    
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"收到聊天请求 - 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not model_loaded:
        logger.error("模型未加载，拒绝请求")
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    
    # 生成或使用会话ID
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"会话ID: {session_id}")
    logger.info(f"用户消息: {request.message[:100]}..." if len(request.message) > 100 else f"用户消息: {request.message}")
    logger.info(f"使用思考链路: {request.use_chain_of_thought}")
    
    # 获取对话历史
    history = conversation_manager.get_history(session_id)
    logger.info(f"对话历史轮数: {len(history)}")
    if history:
        logger.debug("=" * 60)
        logger.debug("【对话历史】")
        for i, (user_msg, assistant_msg) in enumerate(history, 1):
            logger.debug(f"第 {i} 轮对话：")
            logger.debug(f"  用户: {user_msg[:200]}..." if len(user_msg) > 200 else f"  用户: {user_msg}")
            if assistant_msg:
                logger.debug(f"  助手: {assistant_msg[:200]}..." if len(assistant_msg) > 200 else f"  助手: {assistant_msg}")
            else:
                logger.debug(f"  助手: (未保存)")
        logger.debug("=" * 60)
    
    try:
        # 构建提示词
        logger.debug("开始构建提示词...")
        system_prompt = PromptBuilder.get_system_prompt()
        
        # 如果启用思考链路，添加思考提示
        if request.use_chain_of_thought:
            # 根据是否有历史对话选择不同的思考模板
            has_history = len(history) > 0
            thinking_prompt = PromptBuilder.get_chain_of_thought_template(has_history=has_history)
            # 将思考提示添加到用户消息中
            enhanced_prompt = f"{request.message}\n\n{thinking_prompt}"
            # 对于非 ChatGLM 模型，将思考提示也整合到系统提示中
            if "chatglm" not in model_manager.model_name.lower():
                system_prompt = f"{system_prompt}\n\n{thinking_prompt}"
            logger.debug(f"已添加思考链路提示（{'包含话题转换分析' if has_history else '基础版本'}）")
        else:
            enhanced_prompt = request.message
        
        logger.debug(f"提示词构建完成，长度: {len(enhanced_prompt)} 字符")
        logger.debug("=" * 60)
        logger.debug("【系统提示词】")
        logger.debug(system_prompt)
        logger.debug("=" * 60)
        logger.debug("【用户提示词】")
        logger.debug(enhanced_prompt)
        logger.debug("=" * 60)
        
        # 生成回复
        logger.debug("开始生成回复...")
        logger.debug(f"模型名称: {model_manager.model_name}")
        logger.debug(f"使用设备: {model_manager.device}")
        
        gen_start_time = time.time()
        response = model_manager.generate_response(
            prompt=enhanced_prompt,
            history=history if "chatglm" in model_manager.model_name.lower() else None,
            use_chain_of_thought=request.use_chain_of_thought,
            system_prompt=system_prompt if "chatglm" not in model_manager.model_name.lower() else None
        )
        gen_time = time.time() - gen_start_time
        logger.info(f"✅ 回复生成完成，耗时: {gen_time:.2f} 秒，长度: {len(response)} 字符")
        
        # 如果启用了思考链路，尝试提取思考部分
        thinking = None
        if request.use_chain_of_thought:
            logger.debug("尝试提取思考过程...")
            logger.debug(f"原始回复长度: {len(response)} 字符")
            logger.debug(f"原始回复预览: {response[:500]}..." if len(response) > 500 else f"原始回复: {response}")
            thinking = extract_thinking(response)
            if thinking:
                # 从原始回复中移除思考部分
                original_response = response
                response = remove_thinking_from_response(response, thinking)
                logger.debug(f"✅ 成功提取思考过程，长度: {len(thinking)} 字符")
                logger.debug(f"思考过程预览: {thinking[:200]}..." if len(thinking) > 200 else f"思考过程: {thinking}")
                logger.debug(f"提取后的回答长度: {len(response)} 字符")
                logger.debug(f"提取后的回答预览: {response[:200]}..." if len(response) > 200 else f"提取后的回答: {response}")
                # 如果提取后的回答太短或为空，可能是提取错误，保留原回复
                if len(response) < 10 or not response.strip():
                    logger.warning("⚠️ 提取后的回答太短，可能提取错误，保留原回复")
                    response = original_response
                    thinking = None
            else:
                logger.warning("⚠️ 未找到思考过程，返回完整回复")
                logger.debug(f"完整回复内容: {response}")
                # 尝试更宽松的提取方式
                thinking = extract_thinking_relaxed(response)
                if thinking:
                    logger.info(f"✅ 使用宽松模式成功提取思考过程，长度: {len(thinking)} 字符")
                    original_response = response
                    response = remove_thinking_from_response(response, thinking)
                    # 如果提取后的回答太短或为空，可能是提取错误，保留原回复
                    if len(response) < 10 or not response.strip():
                        logger.warning("⚠️ 宽松模式提取后的回答太短，可能提取错误，保留原回复")
                        response = original_response
                        thinking = None
        
        # 保存对话历史
        conversation_manager.add_message(session_id, request.message, response)
        logger.debug("对话历史已保存")
        
        total_time = time.time() - start_time
        logger.info(f"请求处理完成，总耗时: {total_time:.2f} 秒")
        logger.info("=" * 60)
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            thinking=thinking
        )
        
    except Exception as e:
        import traceback
        error_msg = f"处理聊天请求时出错: {str(e)}"
        logger.error(error_msg)
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        logger.info("=" * 60)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天端点（SSE），支持两阶段生成：先生成思考过程，再生成正式回答
    
    Args:
        request: 聊天请求
        
    Yields:
        SSE 格式的流式响应
    """
    import time
    global model_loaded
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    
    session_id = request.session_id or str(uuid.uuid4())
    history = conversation_manager.get_history(session_id)
    
    def generate():
        try:
            system_prompt = PromptBuilder.get_system_prompt()
            
            # 第一阶段：生成思考过程
            if request.use_chain_of_thought:
                thinking_text = ""
                try:
                    # 发送开始思考的信号
                    yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                    
                    # 生成思考过程（流式，不使用 system_prompt，但需要传入会话历史）
                    for chunk in model_manager.generate_thinking_stream(
                        prompt=request.message,
                        history=history,  # 所有模型都需要传入会话历史，以便思考过程能分析话题转换等
                        system_prompt=None  # 思考过程不使用 system_prompt
                    ):
                        thinking_text += chunk
                        # 发送思考过程片段
                        yield f"data: {json.dumps({'type': 'thinking', 'content': chunk})}\n\n"
                    
                    # 清理思考过程（移除可能的标记等）
                    thinking_text = thinking_text.strip()
                    # 移除可能的"思考过程："标记（如果模型输出了）
                    if thinking_text.startswith("**思考过程：**"):
                        thinking_text = thinking_text[len("**思考过程：**"):].strip()
                    elif thinking_text.startswith("思考过程："):
                        thinking_text = thinking_text[len("思考过程："):].strip()
                    elif thinking_text.startswith("思考过程如下"):
                        thinking_text = thinking_text[len("思考过程如下"):].strip()
                    # 移除可能的"正式回答："标记（如果模型输出了）
                    if "**正式回答：**" in thinking_text:
                        thinking_text = thinking_text.split("**正式回答：**")[0].strip()
                    elif "正式回答：" in thinking_text:
                        thinking_text = thinking_text.split("正式回答：")[0].strip()
                    
                    # 发送思考完成信号
                    yield f"data: {json.dumps({'type': 'thinking_complete', 'content': thinking_text})}\n\n"
                    
                    # 第二阶段：生成正式回答
                    yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"
                    
                    answer_text = ""
                    for chunk in model_manager.generate_answer_stream(
                        prompt=request.message,
                        thinking=thinking_text,
                        history=history if "chatglm" in model_manager.model_name.lower() else None,
                        system_prompt=system_prompt if "chatglm" not in model_manager.model_name.lower() else None
                    ):
                        answer_text += chunk
                        # 发送回答片段
                        yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
                    
                    # 清理回答（移除标记等）
                    answer_text = answer_text.strip()
                    if answer_text.startswith("**正式回答：**"):
                        answer_text = answer_text[len("**正式回答：**"):].strip()
                    
                    # 保存对话历史
                    conversation_manager.add_message(session_id, request.message, answer_text)
                    
                    # 发送完成信号
                    yield f"data: {json.dumps({'type': 'complete', 'thinking': thinking_text, 'answer': answer_text, 'session_id': session_id})}\n\n"
                    
                except Exception as e:
                    import traceback
                    logger.error(f"流式生成错误: {str(e)}")
                    logger.error(f"错误堆栈:\n{traceback.format_exc()}")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            else:
                # 不使用思考链路，直接生成回答
                yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"
                
                answer_text = ""
                # 使用原有的生成方法，但需要改为流式
                # 这里简化处理，直接调用 generate_response 然后流式返回
                response = model_manager.generate_response(
                    prompt=request.message,
                    history=history if "chatglm" in model_manager.model_name.lower() else None,
                    use_chain_of_thought=False,
                    system_prompt=system_prompt if "chatglm" not in model_manager.model_name.lower() else None
                )
                
                # 模拟流式输出
                for char in response:
                    answer_text += char
                    yield f"data: {json.dumps({'type': 'answer', 'content': char})}\n\n"
                
                conversation_manager.add_message(session_id, request.message, answer_text)
                yield f"data: {json.dumps({'type': 'complete', 'answer': answer_text, 'session_id': session_id})}\n\n"
                
        except Exception as e:
            import traceback
            logger.error(f"流式生成错误: {str(e)}")
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """获取前端配置"""
    return ConfigResponse(
        health_check_interval=HEALTH_CHECK_INTERVAL
    )


@app.get("/api/progress", response_model=ProgressResponse)
async def get_progress():
    """获取当前生成进度"""
    if not model_manager.current_progress:
        return ProgressResponse(
            is_generating=False,
            progress_percent=0.0,
            current_tokens=0,
            max_tokens=0,
            tokens_per_second=0.0,
            elapsed_time=0.0,
            estimated_remaining=0.0
        )
    
    progress = model_manager.current_progress
    return ProgressResponse(
        is_generating=progress.get("is_generating", False),
        progress_percent=progress.get("progress_percent", 0.0),
        current_tokens=progress.get("current_token_count", 0),
        max_tokens=progress.get("max_new_tokens", 0),
        tokens_per_second=progress.get("tokens_per_second", 0.0),
        elapsed_time=progress.get("elapsed_time", 0.0),
        estimated_remaining=progress.get("estimated_remaining", 0.0)
    )


@app.post("/api/clear")
async def clear_history(request: dict):
    """清空指定会话的历史"""
    session_id = request.get("session_id")
    if session_id:
        conversation_manager.clear_history(session_id)
        return {"status": "ok", "message": "历史已清空"}
    return {"status": "error", "message": "缺少 session_id"}


@app.get("/api/sessions")
async def get_sessions():
    """获取所有会话ID"""
    return {"sessions": conversation_manager.get_all_sessions()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

