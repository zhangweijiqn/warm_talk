# WarmTalk

一个参照 DeepSeek 设计的智能对话系统，支持思考链路（Chain of Thought）和高情商对话风格。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
./scripts/start.sh
```

## 详细文档

请查看 [docs/README.md](docs/README.md) 获取完整的使用说明。

## 项目结构

```
.
├── app/              # 应用代码
├── config/           # 配置文件
├── scripts/          # 脚本文件
├── docs/             # 文档
├── static/           # 静态文件
└── requirements.txt  # 依赖列表
```

## 配置模型

编辑 `config/model_config.py` 来切换模型和调整参数。

## 许可证

MIT License

