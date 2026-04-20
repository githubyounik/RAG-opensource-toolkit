[English](README.md) | [中文](README_CN.md)

# RAG 开源工具包

一个用于构建检索增强生成（RAG）流水线的模块化 Python 工具包。

本仓库已不仅是框架骨架，现已包含一个完整的端到端 RAG 流程，主要功能包括：

- PDF 文件加载
- CSV 文件加载
- 文本清洗与分块
- 检索前查询转换
- 基于嵌入的索引构建
- 密集嵌入检索
- 稀疏 BM25 检索
- 基于 RRF 融合的混合检索
- 大语言模型生成
- 流水线编排

## 当前架构

当前代码库支持 PDF 和 CSV 两种输入格式，均遵循相同的高层流水线：

```text
FileLoader
  -> 配置的 TextProcessor
  -> list[Document]
  -> 可选的 PreRetriever
  -> 配置的 Retriever
  -> 配置的 Generator
  -> RAGPipeline
```

检索层支持三条路径：

```text
密集检索路径：
FileLoader
  -> 配置的 TextProcessor
  -> list[Document]
  -> 配置的 Embedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> EmbeddingRetriever
  -> 配置的 Generator
  -> RAGPipeline
```

```text
稀疏检索路径：
FileLoader
  -> 配置的 TextProcessor
  -> list[Document]
  -> BM25Retriever
  -> 配置的 Generator
  -> RAGPipeline
```

```text
混合检索路径：
FileLoader
  -> 配置的 TextProcessor
  -> list[Document]
  -> 配置的 Embedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> EmbeddingRetriever + BM25Retriever
  -> HybridRetriever（RRF 融合）
  -> 配置的 Generator
  -> RAGPipeline
```

目前已实现的具体加载器：

```text
PDFLoader
CSVLoader
```

流水线执行逻辑简述：

1. 加载源文件并提取结构化文本。
2. 清洗文本并切分为文本块。
3. 可选：在检索前对用户查询进行改写或扩展。
4. 选择检索策略。
5. 对于密集检索，将文本块转换为嵌入向量并存储到 FAISS 向量索引中。启用持久化时，索引（含原始嵌入向量）会保存到磁盘，在后续相同文档的运行中自动复用。
6. 通过嵌入相似度、BM25 关键词匹配或两者的混合方式检索相关文本块。
7. 将检索到的上下文发送给生成模型。
8. 通过统一的流水线接口返回最终答案。

## 模块说明

### `core`

整个项目共用的基础类和数据模型。

- `ParsedFile`：文件加载的统一输出
- `Query`：输入查询对象
- `Document`：块级文档对象
- `RetrievalResult`：检索阶段输出
- `GenerationResult`：生成阶段输出
- `EvaluationResult`：评估阶段输出

### `indexing`

负责文件加载并将原始文件转换为文档块。

- `PDFLoader`：读取 PDF 文件并返回 `ParsedFile`
- `CSVLoader`：读取 CSV 文件并将每行转换为可读文本
- `DocumentProcessor`：常规固定大小分块
- `PropositionProcessor`：由 LLM 驱动的命题级分块
- `create_text_processor_from_config`：根据配置文件选择分块策略
- `FileLoader`、`TextProcessor`：索引相关组件的基础接口
- `prompts.py`：索引组件使用的 LLM 提示词（`PROPOSITION_SYSTEM_PROMPT`）

### `embeddings`

负责密集向量的创建和向量索引的构建。

- `OpenRouterEmbedder`：通过 OpenRouter 调用嵌入模型
- `LocalEmbedder`：使用本地 HuggingFace 编码器模型生成嵌入向量（无需 API 密钥）
- `create_embedder_from_config`：根据配置文件选择嵌入后端
- `EmbeddingIndexer`：对文档块进行嵌入并构建 `VectorIndex`；通过 `build_or_load` 支持 FAISS 持久化
- `VectorIndex`：基于 FAISS 的内存文档与嵌入向量存储，支持 `save`、`load`、`load_all` 磁盘持久化

### `retrieval`

负责为查询找到相关文本块。

- `EmbeddingRetriever`：对查询进行嵌入并按余弦相似度排序
- `BM25Retriever`：对已索引文本块进行分词并通过 BM25 稀疏评分排序
- `HybridRetriever`：同时运行密集和稀疏检索，并使用倒数排名融合（RRF）合并结果
- `create_retriever_from_config`：根据配置文件选择检索策略

### `generation`

负责根据检索到的上下文生成最终答案。

- `ZhipuGenerator`：将检索上下文和查询发送给智谱 GLM 模型
- `OpenRouterGenerator`：将检索上下文和查询发送给 OpenRouter 聊天模型
- `create_generator_from_config`：根据配置文件选择生成后端
- `prompts.py`：生成组件使用的 LLM 提示词（`SYSTEM_PROMPT`）

### `pre_retrieval`

负责在检索前进行查询转换。

- `QueryRewritePreRetriever`：将原始问题改写为更适合检索的查询
- `StepBackPreRetriever`：将问题改写为更宏观的背景查询
- `HyDEPreRetriever`：先生成假设性答案文档，再将其用作检索文本
- `QueryTransformer`：基于共享 LLM 层构建的可复用查询转换助手
- `create_pre_retriever_from_config`：根据配置文件选择前检索策略
- `prompts.py`：前检索组件使用的 LLM 提示词（`REWRITE_SYSTEM_PROMPT`、`STEP_BACK_SYSTEM_PROMPT`、`HYDE_SYSTEM_PROMPT_TEMPLATE`）

### `llm`

被 `generation`、`pre_retrieval`、`post_retrieval` 和 `evaluation` 复用的共享提供商层。

- `OpenRouterChatClient`：带重试和延迟处理的共享 OpenRouter 聊天客户端
- `ZhipuChatClient`：共享智谱聊天客户端
- `LocalChatClient`：使用本地 HuggingFace 因果语言模型运行聊天补全（无需 API 密钥）
- `create_chat_llm_client`：根据配置选择提供商客户端

### `pipelines`

负责将各模块串联起来。

- `RAGPipeline`：前检索、检索、后检索、生成和评估的标准编排入口

### `post_retrieval` 和 `evaluation`

`post_retrieval` 目前包含：

- `RelevantSegmentExtractor`：从邻近检索块中重建连续文档片段
- `ContextualCompressor`：将每个检索块压缩为仅与查询相关的内容
- `CohereReranker`：调用 OpenRouter 的 rerank API，使用专用重排序模型（如 `cohere/rerank-v3.5`）
- `CrossReranker`：使用本地 HuggingFace 交叉编码器模型重排序；对每个（查询, 文档）对进行联合编码，精度较高
- `BiReranker`：使用本地 HuggingFace 双编码器模型重排序；对查询和文档分别编码后按点积相似度排序
- `create_post_retriever_from_config`：根据配置文件选择后检索策略
- `prompts.py`：后检索组件使用的 LLM 提示词（`CONTEXTUAL_COMPRESSION_SYSTEM_PROMPT`）

`evaluation` 目前包含：

- `DeepEvalEvaluator`：DeepEval 风格的正确性、忠实度和上下文相关性评估
- `create_evaluator_from_config`：根据配置文件选择评估策略
- `prompts.py`：评估组件使用的 LLM 提示词（`CORRECTNESS_SYSTEM_PROMPT`、`FAITHFULNESS_SYSTEM_PROMPT`、`CONTEXTUAL_RELEVANCY_SYSTEM_PROMPT`）

## 项目结构

```text
RAG-opensource-toolkit/
├── .env.example
├── examples/
├── pyproject.toml
├── README.md
├── README_CN.md
├── configs/
├── docs/
├── src/
│   └── rag_toolkit/
│       ├── core/
│       ├── embeddings/
│       ├── evaluation/
│       ├── generation/
│       ├── indexing/
│       ├── pipelines/
│       ├── post_retrieval/
│       ├── pre_retrieval/
│       └── retrieval/
└── tests/
```

## 安装

需要 Python 3.10 及以上版本。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

如需运行完整的端到端示例，还需安装可选的 LLM 依赖：

```bash
pip install -e ".[dev,llm]"
```

## 环境变量

根据 `./.env.example` 创建 `.env` 文件：

```env
OPENROUTER_API_KEY=your-openrouter-key-here
ZHIPU_API_KEY=your-zhipu-key-here
```

## 分块配置

分块参数可通过 [configs/pipeline.example.yaml](configs/pipeline.example.yaml) 进行配置。

当前设置：

```yaml
indexing:
  document_processing:
    strategy: proposition
    chunk_size: 1000
    chunk_overlap: 200
    proposition:
      model: nvidia/nemotron-3-super-120b-a12b:free
      temperature: 0.0
      max_tokens: 512
      max_retries: 2
      retry_delay_seconds: 2.0
```

可通过 `indexing.document_processing.strategy` 切换分块策略：

```yaml
indexing:
  document_processing:
    strategy: default
```

或

```yaml
indexing:
  document_processing:
    strategy: proposition
```

`default` 使用 `DocumentProcessor` 进行常规固定大小分块。

`proposition` 先生成常规基础块，再通过 `PropositionProcessor` 将每个块改写为命题级文档。

## 嵌入配置

嵌入提供商和模型通过 [configs/pipeline.example.yaml](configs/pipeline.example.yaml) 配置。

当前默认值：

```yaml
embeddings:
  provider: openrouter
  model: nvidia/llama-nemotron-embed-vl-1b-v2:free
  storage:
    enabled: true
    cache_dir: .rag_cache/faiss
    reuse_existing: true
```

支持的 `provider` 值：

- `openrouter`：调用 OpenRouter 嵌入 API；需要 `OPENROUTER_API_KEY`
- `local`：在本地加载 HuggingFace 编码器模型；无需 API 密钥

使用 `provider: local` 时，可配置额外参数：

```yaml
embeddings:
  provider: local
  model: BAAI/bge-small-en-v1.5   # HuggingFace 模型 ID 或本地路径
  max_length: 512
  batch_size: 32
  device: auto                     # auto / cpu / cuda
  pooling_method: mean             # mean / cls
```

当 `retrieval.strategy` 为 `embedding` 或 `hybrid` 时需要嵌入配置。

## FAISS 索引持久化

当 `embeddings.storage.enabled: true` 时，FAISS 索引在首次运行后保存到磁盘，在后续相同文档的运行中自动复用，避免重复调用嵌入 API。

每个持久化索引存储在 `{cache_dir}/{namespace}/{fingerprint}/` 目录下，包含四个文件：

```text
index.faiss       — FAISS 平面内积索引
documents.json    — 对齐的文档块
embeddings.npy    — 原始嵌入向量（numpy 数组）
metadata.json     — 文档数量、嵌入数量和归一化标志
```

指纹是由文档内容和嵌入模型名称派生的 16 位 SHA-256 哈希值。任一项变更时，会自动创建新的索引目录。

关键配置参数：

- `storage.enabled`：设为 `true` 启用持久化；`false` 则仅保存在内存中
- `storage.cache_dir`：所有持久化索引的根目录（默认：`.rag_cache/faiss`）
- `storage.reuse_existing`：为 `true` 时，若存在相同文档和模型的索引则直接加载，不重新嵌入

当 `storage.enabled: true` 时，也可以在不提供 PDF 路径的情况下进行查询。示例程序会自动加载并合并 `cache_dir` 下所有持久化索引：

```bash
PYTHONPATH=src python examples/simple_rag.py "<你的问题>"
```

这对于查询已建索引的内容非常便捷，无需重新加载源文件。

## 检索配置

检索通过 [configs/pipeline.example.yaml](configs/pipeline.example.yaml) 配置。

示例：

```yaml
retrieval:
  strategy: embedding
  embedding:
    top_k: 4
  bm25:
    top_k: 4
    k1: 1.5
    b: 0.75
    lowercase: true
  hybrid:
    top_k: 4
    rrf_k: 60
```

支持的 `strategy` 值：

- `embedding`：构建内存 `VectorIndex`，对查询进行嵌入，按余弦相似度检索
- `bm25`：完全跳过嵌入，直接使用 BM25 对文本块进行检索
- `hybrid`：同时运行密集和稀疏检索，再通过 RRF 融合排序结果

关键参数：

- `embedding.top_k`：密集检索返回的文本块数量
- `bm25.top_k`：BM25 返回的文本块数量
- `bm25.k1`：BM25 词频饱和参数
- `bm25.b`：BM25 文档长度归一化参数
- `bm25.lowercase`：分词前是否将文档和查询文本转为小写
- `hybrid.top_k`：融合后保留的文本块数量
- `hybrid.rrf_k`：合并嵌入和 BM25 排序列表时使用的倒数排名融合常数

## 前检索配置

前检索为可选模块，通过 [configs/pipeline.example.yaml](configs/pipeline.example.yaml) 配置。

当前默认值：

```yaml
pre_retrieval:
  enabled: true
  strategy: rewrite
  provider: openrouter
  model: z-ai/glm-5.1
  temperature: 0.0
  max_tokens: 256
  max_retries: 2
  retry_delay_seconds: 2.0
  hyde_target_char_length: 800
```

支持的 `strategy` 值：

- `rewrite`：将原始问题改写为更适合检索的查询
- `step_back`：将原始问题改写为更宏观的背景查询
- `hyde`：先生成假设性答案文档，再将该文档用作检索查询

支持的 `provider` 值：

- `openrouter`
- `zhipu`
- `local`：运行本地 HuggingFace 因果语言模型；在同一配置块中添加 `device` 和 `max_length`

`hyde_target_char_length` 仅在 `strategy: hyde` 时生效，为假设性文档提供软性长度目标，使生成文本更接近已索引块的大小。

## 后检索配置

后检索为可选模块，通过 [configs/pipeline.example.yaml](configs/pipeline.example.yaml) 配置。

示例：

```yaml
post_retrieval:
  enabled: false
  strategy: relevant_segment_extraction
  relevant_segment_extraction:
    irrelevant_chunk_penalty: 0.2
    rank_decay: 0.08
    max_segment_length: 6
    overall_max_length: 12
    minimum_segment_value: 0.15
  contextual_compression:
    provider: openrouter
    model: z-ai/glm-5.1
    temperature: 0.0
    max_tokens: 1200
    max_retries: 2
    retry_delay_seconds: 2.0
  rerank:
    provider: openrouter
    model: cohere/rerank-v3.5
    top_k: 3
    max_tokens_per_doc:
    max_retries: 2
    retry_delay_seconds: 2.0
```

支持的 `strategy` 值：

- `relevant_segment_extraction`：将邻近相关块合并为连续上下文片段
- `contextual_compression`：使用 LLM 将每个检索块压缩为仅与查询相关的内容
- `rerank`：使用 OpenRouter 专用重排序端点，通过云端重排序模型对检索块重新排序
- `cross_rerank`：使用本地交叉编码器模型对检索块重排序（无需 API 密钥）
- `bi_rerank`：使用本地双编码器模型对检索块重排序（无需 API 密钥）

`rerank` 的关键参数：

- `rerank.top_k`：重排序后保留的最大文档数量
- `rerank.max_tokens_per_doc`：传递给 rerank API 的可选单文档截断预算
- `rerank.model`：发送给 OpenRouter 的重排序模型名称，例如 `cohere/rerank-v3.5`

`cross_rerank` 和 `bi_rerank` 的关键参数：

```yaml
cross_rerank:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2  # HuggingFace 模型 ID 或本地路径
  top_k: 3
  max_length: 512
  batch_size: 32
  device: auto                                   # auto / cpu / cuda

bi_rerank:
  model: BAAI/bge-small-en-v1.5
  top_k: 3
  max_length: 512
  batch_size: 32
  device: auto
  pooling_method: mean                           # mean / cls
```

重排序器对比：

| 策略 | 模型位置 | 评分方式 | 精度 | 速度 |
|---|---|---|---|---|
| `rerank` | 云端 API（OpenRouter） | 专用重排序服务 | 高 | 依赖网络 |
| `cross_rerank` | 本地 HuggingFace | （查询 + 文档）联合编码，输出单一相关性 logit | 高 | 较慢（每对需一次前向传播） |
| `bi_rerank` | 本地 HuggingFace | 查询和文档分别编码，按点积排序 | 中 | 较快（嵌入相互独立） |

重要约束：

- 当 `post_retrieval.enabled: true` 且 `strategy: relevant_segment_extraction` 时，工具包会自动强制索引使用 `DocumentProcessor` 且 `chunk_overlap = 0`
- 即使 YAML 文件中配置了 `strategy: proposition` 或非零重叠，代码中也会覆盖此设置

此约束是因为相关片段提取依赖于干净、无重叠的块边界，才能可靠地重建连续片段。

其他所有策略均无特殊分块限制。

## 评估配置

评估为可选模块，通过 [configs/pipeline.example.yaml](configs/pipeline.example.yaml) 配置。

示例：

```yaml
evaluation:
  enabled: false
  strategy: deep_eval_style
  provider: openrouter
  model: z-ai/glm-5.1
  temperature: 0.0
  max_tokens: 256
  correctness_threshold: 0.7
  faithfulness_threshold: 0.7
  contextual_relevancy_threshold: 0.7
  max_retries: 2
  retry_delay_seconds: 2.0
```

支持的 `strategy` 值：

- `deep_eval_style`：通过共享 LLM 层评估正确性、忠实度和上下文相关性

注意事项：

- `correctness` 需要 `query.metadata["expected_output"]`
- 若未提供参考答案，正确性评估会自动跳过
- `faithfulness` 使用附加在 `GenerationResult.contexts` 中的检索上下文
- `contextual_relevancy` 检查检索到的上下文是否对查询有用

## 生成模型配置

生成提供商和模型同样通过 [configs/pipeline.example.yaml](configs/pipeline.example.yaml) 控制。

当前默认值：

```yaml
generation:
  provider: openrouter
  model: z-ai/glm-5.1
  temperature: 0.6
  max_tokens:
  max_retries: 2
  retry_delay_seconds: 2.0
```

切换为智谱生成：

```yaml
generation:
  provider: zhipu
  model: glm-4.7
  temperature: 0.6
  max_tokens:
```

使用不同的 OpenRouter 模型，直接在配置中填入完整模型 ID：

```yaml
generation:
  provider: openrouter
  model: z-ai/glm-5.1
  temperature: 0.6
  max_tokens:
  max_retries: 2
  retry_delay_seconds: 2.0
```

使用本地模型进行生成：

```yaml
generation:
  provider: local
  model: Qwen/Qwen2.5-0.5B-Instruct   # HuggingFace 模型 ID 或本地路径
  device: auto
  max_length: 2048
  temperature: 0.6
  max_tokens: 512
```

`local` 提供商同样适用于 `pre_retrieval`、`post_retrieval.contextual_compression` 和 `evaluation`，只需在对应配置块中设置 `provider: local` 和 `model` 即可。

OpenRouter 生成器还内置了针对瞬时错误（如 `429 Too Many Requests`）的延迟重试逻辑。

## 运行示例

PDF 示例：

```bash
PYTHONPATH=src python examples/simple_rag.py <pdf路径> "<你的问题>"
```

示例：

```bash
PYTHONPATH=src python examples/simple_rag.py docs/Understanding_Climate_Change.pdf "What marked the beginning of the modern climate era and human civilization?"
```

CSV 示例：

```bash
PYTHONPATH=src python examples/simple_csv_rag.py <csv路径> "<你的问题>"
```

示例：

```bash
PYTHONPATH=src python examples/simple_csv_rag.py docs/customers-100.csv "how can i contact Sheryl"
```

## 设计目标

- 保持清晰的模块边界
- 使每个阶段可独立替换
- 保持流水线易于理解
- 支持从最小可用 RAG 系统逐步扩展

## 当前状态

已实现的功能：

- PDF 加载已实现
- CSV 加载已实现
- 基于分块的预处理已实现
- 基于命题的预处理已实现
- OpenRouter 嵌入已实现
- 内存向量索引已实现
- 余弦相似度检索已实现
- BM25 稀疏检索已实现
- 基于 RRF 融合的混合检索已实现
- 基于智谱的生成已实现
- 基于 OpenRouter 的生成已实现
- OpenRouter 模型选择可通过 YAML 配置
- DeepEval 风格评估已实现
- 带原始嵌入向量存储的 FAISS 索引持久化已实现
- 基于内容指纹的缓存复用已实现
- 仅查询模式（缓存存在时无需输入文件）已实现
- 通过 OpenRouter 的 API 重排序（CohereReranker）已实现
- 本地交叉编码器重排序（CrossReranker）已实现
- 本地双编码器重排序（BiReranker）已实现
- 嵌入阶段的本地 HuggingFace 模型支持（LocalEmbedder）已实现
- 所有 LLM 阶段的本地 HuggingFace 模型支持（LocalChatClient）已实现

尚未扩展：

- 高级前检索逻辑
- 多文件摄取工作流
