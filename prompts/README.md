# Prompt 热重载功能

## 功能说明

WarmTalk 支持 prompt 热重载功能，修改 prompt 文件后**无需重启服务**即可生效。

## 文件说明

prompt 文件位于 `prompts/` 目录下：

- **系统提示词**：
  - `system_prompt_simple.txt` - 简洁版系统提示词（默认）
  - `system_prompt_detailed.txt` - 详细版系统提示词

- **思考链路模板**：
  - `cot_template_base_simple.txt` - 简洁版思考模板（无历史对话）
  - `cot_template_with_history_simple.txt` - 简洁版思考模板（有历史对话）
  - `cot_template_base_detailed.txt` - 详细版思考模板（无历史对话）
  - `cot_template_with_history_detailed.txt` - 详细版思考模板（有历史对话）

## 使用方法

1. **修改 prompt 文件**：
   - 直接编辑 `prompts/` 目录下的 `.txt` 文件
   - 保存文件后，下次请求时会自动使用新的 prompt

2. **生效时机**：
   - 修改文件后，**下次对话请求时**会自动加载新的 prompt
   - 无需重启服务

3. **回退机制**：
   - 如果文件不存在或读取失败，会自动使用代码中的默认 prompt
   - 不会影响服务正常运行

## 注意事项

- 文件编码必须是 UTF-8
- 修改文件后，确保文件格式正确（特别是 Markdown 格式）
- 建议在修改前备份原文件
- 修改后可以通过发送一条消息来测试新 prompt 是否生效

## 示例

修改 `system_prompt_simple.txt`：

```bash
# 编辑文件
vim prompts/system_prompt_simple.txt

# 或者使用其他编辑器
code prompts/system_prompt_simple.txt
```

保存后，下次对话请求时会自动使用新的 prompt，无需重启服务。

