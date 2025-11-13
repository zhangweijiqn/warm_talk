# WarmTalk

一个参照 DeepSeek 设计的智能对话系统，支持思考链路（Chain of Thought）和高情商对话风格。系统能够结合上下文进行连续对话，并像一位高情商的倾听者和安慰者一样与用户交流。

## 功能特点

- 🤔 **思考链路**：模型会先进行思考分析，再给出正式回答（类似 DeepSeek）
- 💬 **上下文对话**：支持多轮连续对话，能够理解上下文语境
- ❤️ **高情商对话**：专门设计的高情商提示词，让 AI 成为温暖的倾听者和安慰者
- 🚀 **轻量级模型**：支持使用中文开源小模型进行测试
- 🎨 **现代化界面**：美观的 Web 界面，流畅的交互体验

## 技术栈

- **后端**：FastAPI
- **模型**：Transformers（支持 ChatGLM、Qwen、Baichuan 等中文模型）
- **前端**：原生 HTML + CSS + JavaScript
- **深度学习框架**：PyTorch

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

编辑 `config/model_config.py` 文件，选择你想要使用的模型和调整参数：

```python
# 切换模型
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # 推荐用于 CPU

# 调整生成参数
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    # ... 更多参数
}

# 配置量化（CPU 模式推荐启用）
USE_QUANTIZATION = True
QUANTIZATION_BITS = 8
```

**注意**：所有模型相关的配置都在 `config/model_config.py` 中，便于切换模型和调参。

**推荐的中文小模型：**

- `THUDM/chatglm3-6b` - ChatGLM3-6B（推荐，6B 参数）
- `THUDM/chatglm2-6b` - ChatGLM2-6B（更小版本）
- `Qwen/Qwen-7B-Chat` - 通义千问 7B
- `baichuan-inc/Baichuan2-7B-Chat` - 百川 2-7B

**注意**：首次运行会自动从 Hugging Face 下载模型，需要较长时间和足够的磁盘空间。

### 3. 运行服务

```bash
# 使用启动脚本（推荐）
./scripts/start.sh

# 或直接运行
python -m app.main

# 或使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. 访问界面

打开浏览器访问：`http://localhost:8000`

## 使用说明

### 基本对话

1. 在输入框中输入你的消息
2. 按 Enter 或点击发送按钮
3. AI 会先显示思考过程，然后给出正式回答

### 清空对话

点击"清空对话"按钮可以清除当前会话的历史记录。

### API 接口

#### 健康检查

```bash
GET /health
```

返回服务状态和模型加载情况。

#### 发送消息

```bash
POST /api/chat
Content-Type: application/json

{
    "message": "你好",
    "session_id": "可选，会话ID",
    "use_chain_of_thought": true
}
```

响应：

```json
{
    "response": "AI 的回复",
    "session_id": "会话ID",
    "thinking": "思考过程（如果启用）"
}
```

#### 清空历史

```bash
POST /api/clear
Content-Type: application/json

{
    "session_id": "会话ID"
}
```

## 项目结构

```
.
├── app/                      # 应用代码
│   ├── __init__.py
│   ├── main.py              # FastAPI 主应用
│   ├── model_manager.py     # 模型加载和推理
│   ├── conversation_manager.py  # 对话历史管理
│   └── prompt_builder.py    # 提示词构建（思考链路+高情商）
├── config/                   # 配置文件
│   ├── __init__.py
│   ├── model_config.py      # 模型配置（模型选择、生成参数、量化等）
│   └── server_config.py     # 服务器配置（端口、主机等）
├── scripts/                  # 脚本文件
│   └── start.sh             # 启动脚本
├── docs/                     # 文档
│   ├── README.md            # 项目说明（本文件）
│   ├── DEBUG.md             # 调试说明
│   └── EXAMPLE.md           # 使用示例
├── static/                   # 静态文件
│   └── index.html           # 前端界面
├── requirements.txt          # 依赖列表
└── .gitignore               # Git 忽略文件
```

## 核心特性详解

### 思考链路（Chain of Thought）

系统会在回答前进行思考分析：

1. 理解用户的核心诉求和情感状态
2. 分析当前对话的上下文背景
3. 考虑最合适的回应方式和语气
4. 组织语言，确保既温暖又有效

### 高情商对话风格

AI 助手被设计为：

- **深度倾听**：认真理解用户的每一个字句
- **共情能力**：能够感同身受，理解用户的情绪
- **温暖安慰**：用温和、理解的方式提供支持
- **智慧引导**：在适当的时候提供建设性建议
- **耐心陪伴**：无论用户说什么，都会耐心倾听

## 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）
- 至少 8GB 内存（推荐 16GB+）
- 足够的磁盘空间（模型文件通常 10GB+）

## 常见问题

### Q: 模型加载很慢？

A: 首次运行需要从 Hugging Face 下载模型，可能需要较长时间。建议使用国内镜像或提前下载模型。

### Q: 内存不足？

A: 可以尝试使用更小的模型（如 Qwen2-0.5B），或者在 `config/model_config.py` 中调整生成参数。

### Q: 如何切换模型？

A: 修改 `config/model_config.py` 中的 `MODEL_NAME` 变量，然后重启服务。

### Q: 支持 GPU 加速吗？

A: 支持。如果检测到 CUDA，会自动使用 GPU 加速。

## 开发计划

- [ ] 支持流式输出（Streaming）
- [ ] 支持更多模型格式（GGML、量化模型等）
- [ ] 添加对话导出功能
- [ ] 优化思考链路的展示方式
- [ ] 支持自定义提示词模板

## 许可证

MIT License

## 致谢

- 感谢 ChatGLM、Qwen、Baichuan 等开源模型团队
- 参考了 DeepSeek 的产品设计理念

