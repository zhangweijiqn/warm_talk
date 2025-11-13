"""
对话管理器 - 管理对话历史和上下文
"""
from typing import List, Tuple, Optional
from config.model_config import MAX_HISTORY_LENGTH, SAVE_ONLY_USER_MESSAGES


class ConversationManager:
    """管理对话历史和上下文"""
    
    def __init__(self, max_history: int = MAX_HISTORY_LENGTH):
        self.max_history = max_history
        self.conversations: dict = {}  # {session_id: [(user_msg, assistant_msg), ...]}
    
    def add_message(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_message: str
    ):
        """
        添加一轮对话
        
        Args:
            session_id: 会话ID
            user_message: 用户消息
            assistant_message: 助手回复
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # 根据配置决定是否保存助手回复
        if SAVE_ONLY_USER_MESSAGES:
            # 仅保存用户问题，助手回复设为空字符串
            self.conversations[session_id].append((user_message, ""))
        else:
            # 保存完整的对话（用户消息 + 助手回复）
            self.conversations[session_id].append((user_message, assistant_message))
        
        # 限制历史长度
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """
        获取对话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            对话历史列表
        """
        return self.conversations.get(session_id, [])
    
    def clear_history(self, session_id: str):
        """
        清空指定会话的历史
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.conversations:
            self.conversations[session_id] = []
    
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return list(self.conversations.keys())

