"""
主应用 - FastAPI 后端服务
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import uuid
import os
import sys

from app.model_manager import ModelManager
from app.conversation_manager import ConversationManager
from app.prompt_builder import PromptBuilder
from config import HOST, PORT
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    # 模式1: **思考过程：** ... **正式回答：**
    pattern1 = r'\*\*思考过程：?\*\*[\s\n]*(.*?)[\s\n]*\*\*正式回答：?\*\*'
    match = re.search(pattern1, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking:
            return thinking
    
    # 模式2: 思考过程： ... 正式回答：
    pattern2 = r'思考过程：?[\s\n]*(.*?)[\s\n]*正式回答：?'
    match = re.search(pattern2, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking:
            return thinking
    
    # 模式3: 思考： ... 回答：
    pattern3 = r'思考：?[\s\n]*(.*?)[\s\n]*回答：?'
    match = re.search(pattern3, response, re.DOTALL | re.IGNORECASE)
    if match:
        thinking = match.group(1).strip()
        if thinking:
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


def remove_thinking_from_response(response: str, thinking: str) -> str:
    """
    从回复中移除思考部分，只保留正式回答
    """
    if not thinking:
        return response
    
    # 尝试多种方式移除思考部分
    # 方法1: 移除思考过程标记和内容
    patterns = [
        r'\*\*思考过程：?\*\*[\s\n]*.*?[\s\n]*\*\*正式回答：?\*\*',
        r'思考过程：?[\s\n]*.*?[\s\n]*正式回答：?',
        r'思考：?[\s\n]*.*?[\s\n]*回答：?',
        r'\[思考过程?\]?[\s\n]*.*?[\s\n]*\[正式回答?\]?',
    ]
    
    for pattern in patterns:
        response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # 方法2: 如果思考内容在回复中，直接移除
    if thinking in response:
        response = response.replace(thinking, '').strip()
    
    # 清理多余的空行和标记
    response = re.sub(r'\*\*正式回答：?\*\*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'正式回答：?', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # 清理多余空行
    
    return response.strip()

# 初始化 FastAPI 应用
app = FastAPI(title="WarmTalk", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
model_manager = ModelManager()
conversation_manager = ConversationManager()

# 全局变量：模型加载状态
model_loaded = False


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


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    import traceback
    global model_loaded
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


@app.get("/")
async def root():
    """返回前端页面"""
    import os
    static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")
    return FileResponse(static_path)


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(
        status="ok" if model_loaded else "error",
        model_loaded=model_loaded,
        model_name=model_manager.model_name if model_loaded else None
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
    
    try:
        # 构建提示词
        logger.info("开始构建提示词...")
        system_prompt = PromptBuilder.SYSTEM_PROMPT
        
        # 如果启用思考链路，添加思考提示
        if request.use_chain_of_thought:
            thinking_prompt = f"""{PromptBuilder.CHAIN_OF_THOUGHT_TEMPLATE}"""
            # 将思考提示添加到用户消息中
            enhanced_prompt = f"{request.message}\n\n{thinking_prompt}"
            # 对于非 ChatGLM 模型，将思考提示也整合到系统提示中
            if "chatglm" not in model_manager.model_name.lower():
                system_prompt = f"{system_prompt}\n\n{thinking_prompt}"
            logger.info("已添加思考链路提示")
        else:
            enhanced_prompt = request.message
        
        logger.info(f"提示词构建完成，长度: {len(enhanced_prompt)} 字符")
        
        # 生成回复
        logger.info("开始生成回复...")
        logger.info(f"模型名称: {model_manager.model_name}")
        logger.info(f"使用设备: {model_manager.device}")
        
        gen_start_time = time.time()
        response = model_manager.generate_response(
            prompt=enhanced_prompt,
            history=history if "chatglm" in model_manager.model_name.lower() else None,
            use_chain_of_thought=request.use_chain_of_thought,
            system_prompt=system_prompt if "chatglm" not in model_manager.model_name.lower() else None
        )
        gen_time = time.time() - gen_start_time
        logger.info(f"回复生成完成，耗时: {gen_time:.2f} 秒")
        logger.info(f"生成回复长度: {len(response)} 字符")
        logger.info(f"回复预览: {response[:200]}..." if len(response) > 200 else f"回复: {response}")
        
        # 如果启用了思考链路，尝试提取思考部分
        thinking = None
        if request.use_chain_of_thought:
            logger.info("尝试提取思考过程...")
            thinking = extract_thinking(response)
            if thinking:
                # 从原始回复中移除思考部分
                response = remove_thinking_from_response(response, thinking)
                logger.info(f"成功提取思考过程，长度: {len(thinking)} 字符")
            else:
                logger.info("未找到思考过程，返回完整回复")
        
        # 保存对话历史
        conversation_manager.add_message(session_id, request.message, response)
        logger.info("对话历史已保存")
        
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

