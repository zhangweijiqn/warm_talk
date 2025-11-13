"""
服务器配置文件
"""
import os

# ==================== 服务器配置 ====================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

