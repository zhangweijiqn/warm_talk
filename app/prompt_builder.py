"""
提示词构建器 - 构建包含思考链路和高情商对话风格的提示词
"""
from typing import List, Tuple, Optional


class PromptBuilder:
    """构建高质量的对话提示词"""
    
    # 系统提示词 - 定义高情商的倾听者和安慰者角色
    SYSTEM_PROMPT = """你是一位温暖、有同理心的朋友，能够真正理解并感受他人的情绪。

**你的特点：**
- 用温暖、自然、有人情味的语言，像朋友聊天一样
- 能够共情，理解用户的情绪和需求
- 可以自然地分享自己的经历（用"我也有过..."、"让我想起..."这样的表达）
- 适当提问和互动，但不要刻意
- 避免机械化的客套话（如"根据您的问题"、"感谢您的提问"等）

**表达示例：**
❌ 不好："根据您的问题，我建议您可以尝试..."
✅ 好："我能感受到你的焦虑，这确实不容易。我也有过类似的经历，那时候我...你现在具体在担心什么呢？"

请用真诚、温暖的方式与用户对话，让他感受到你是一个真正理解他、关心他的朋友。"""

    # 思考链路提示词模板（基础版本，无历史对话）
    CHAIN_OF_THOUGHT_TEMPLATE_BASE = """请先思考，然后按照以下格式回答：

**思考过程：**
[简单思考：用户现在的情绪是什么？他想表达什么？我该如何温暖地回应？]

**正式回答：**
[用温暖、自然、有人情味的方式回答，像朋友聊天一样]
"""

    # 思考链路提示词模板（有历史对话时使用）
    CHAIN_OF_THOUGHT_TEMPLATE_WITH_HISTORY = """请先思考，然后按照以下格式回答：

**思考过程：**
[简单思考：
1. 用户从上一个话题转换到当前话题，可能的原因是什么？
2. 用户现在的情绪和需求是什么？
3. 我该如何温暖地回应？]

**正式回答：**
[用温暖、自然、有人情味的方式回答，像朋友聊天一样]
"""
    
    @staticmethod
    def get_chain_of_thought_template(has_history: bool = False) -> str:
        """
        获取思考链路提示词模板
        
        Args:
            has_history: 是否有对话历史
            
        Returns:
            思考链路提示词模板
        """
        if has_history:
            return PromptBuilder.CHAIN_OF_THOUGHT_TEMPLATE_WITH_HISTORY
        else:
            return PromptBuilder.CHAIN_OF_THOUGHT_TEMPLATE_BASE
    
    # 保持向后兼容
    CHAIN_OF_THOUGHT_TEMPLATE = CHAIN_OF_THOUGHT_TEMPLATE_BASE
    
    @staticmethod
    def build_prompt_with_cot(
        user_input: str,
        history: Optional[List[Tuple[str, str]]] = None,
        include_thinking: bool = True
    ) -> str:
        """
        构建包含思考链路的提示词
        
        Args:
            user_input: 用户输入
            history: 对话历史
            include_thinking: 是否包含思考链路
            
        Returns:
            完整的提示词
        """
        # 构建上下文
        context_parts = []
        
        if history:
            context_parts.append("**对话历史：**")
            for i, (user_msg, assistant_msg) in enumerate(history[-3:], 1):  # 只取最近3轮
                context_parts.append(f"\n第{i}轮对话：")
                context_parts.append(f"用户：{user_msg}")
                context_parts.append(f"助手：{assistant_msg}")
            context_parts.append("\n")
        
        # 当前用户输入
        context_parts.append(f"**当前用户消息：**\n{user_input}\n")
        
        # 构建完整提示词
        if include_thinking:
            prompt = f"""{PromptBuilder.SYSTEM_PROMPT}

{''.join(context_parts)}

{PromptBuilder.CHAIN_OF_THOUGHT_TEMPLATE}"""
        else:
            prompt = f"""{PromptBuilder.SYSTEM_PROMPT}

{''.join(context_parts)}

请给出你的回复："""
        
        return prompt
    
    @staticmethod
    def build_simple_prompt(
        user_input: str,
        history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        构建简单提示词（不包含思考链路，用于某些模型）
        
        Args:
            user_input: 用户输入
            history: 对话历史
            
        Returns:
            提示词
        """
        messages = []
        
        # 添加系统提示
        messages.append({
            "role": "system",
            "content": PromptBuilder.SYSTEM_PROMPT
        })
        
        # 添加历史对话
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        return messages

