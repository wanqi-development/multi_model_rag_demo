## 核心模块说明

### 状态管理 (`state.py`)

定义工作流的全局状态：

- `input_type`: 输入类型（`only_text` / `only_image` / `text_and_image`）
- `input_text`: 文本输入内容
- `input_image`: 图片输入（URL 或 Base64）
- `retrieved_docs`: 检索到的文档列表
- `retrieved_images`: 检索到的图片列表
- `answer`: 生成的答案
- `precision`: 答案准确度评分
- `approve_result`: 人工审批结果

### 工作流节点 (`node/nodes.py`)

| 节点 | 功能 |
|------|------|
| `process_user_input_node` | 解析用户输入，提取文本和图片 |
| `chat_history_ai_node` | 处理历史对话检索 |
| `context_ai_node` | 处理向量知识库检索 |
| `evaluate_node` | 评估答案准确性（RAGAS） |
| `human_interrupt_node` | 人工审批节点 |
| `web_search_ai_node` | 网络搜索节点 |

### 路由逻辑 (`router/router.py`)

| 路由 | 作用 |
|------|------|
| `process_user_input_router` | 根据输入类型选择后续节点 |
| `chat_history_router` | 历史检索结果路由 |
| `context_ai_router` | 向量库检索结果路由 |
| `need_human_approval_router` | 根据评估结果决定是否人工审批 |

### 工具 (`tools/tools.py`)

| 工具 | 功能 |
|------|------|
| `web_search_tool` | 调用 Zhipu AI 进行网络搜索 |
| `search_from_long_term_context` | 从历史对话记录中检索 |
| `retrieve_from_milvus` | 从向量知识库中检索 |

## 使用到的模型

项目当前使用以下模型和服务：

- **主LLM**: Kimi (kimi-k2.6) - 用于生成回答和工具调用决策
- **多模态嵌入**: Qwen3-VL-Embedding（DashScope）- 用于图片和文本的向量编码
- **评估LLM**: GPT-4o-mini - 用于 RAGAS 答案准确性评估
- **嵌入模型**: OpenAI Embeddings - 用于文本向量编码
- **向量数据库**: Milvus - 用于存储和检索向量数据
- **网络搜索**: Zhipu AI Web Search - 用于实时信息检索

## 注意事项

1. 确保 Milvus 服务已启动并可连接
2. 配置的 API Key 需要有足够的权限
3. 图片输入支持本地文件路径、远程 URL 和 Base64 编码
4. 工作流使用 InMemorySaver，重启后会话历史会丢失
5. **必须在项目根目录创建 `.env` 文件并配置正确的 API Key 才能运行**

## License

MIT License
