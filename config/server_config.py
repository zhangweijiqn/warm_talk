"""
服务器配置文件
"""
import os

# ==================== 服务器配置 ====================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# ==================== 前端配置 ====================

# 健康检查间隔（毫秒）
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 60000))  # 默认60秒（60000毫秒）

# ==================== 日志配置 ====================

# 日志级别：INFO（仅显示重要信息）或 DEBUG（显示详细信息）
# 可以通过环境变量 LOG_LEVEL 设置，例如：export LOG_LEVEL=DEBUG
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # 默认 INFO

