# Enterprise RAG 系统 · Code Wiki

> 本文档基于项目仓库 (`ai_0119_rag`) 的目录结构、`pyproject.toml` 依赖声明与 `.env.example` 配置项进行系统性梳理，覆盖项目整体架构、模块职责、关键组件、依赖关系与运行方式。

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 项目目录结构](#2-项目目录结构)
- [3. 技术栈](#3-技术栈)
- [4. 整体架构](#4-整体架构)
- [5. 模块职责说明](#5-模块职责说明)
- [6. 关键组件与设计说明](#6-关键组件与设计说明)
- [7. 依赖关系](#7-依赖关系)
- [8. 配置项说明](#8-配置项说明)
- [9. 核心业务流程](#9-核心业务流程)
- [10. 项目运行方式](#10-项目运行方式)

---

## 1. 项目概述

**项目名称**：`ai_0119_rag`（Enterprise RAG）

**项目定位**：一套面向企业级知识库场景的 **检索增强生成（Retrieval-Augmented Generation, RAG）** 系统。系统被拆分为两个独立部署的服务：

| 服务 | 名称 | 默认端口 | 职责 |
|------|------|----------|------|
| Import Service | Enterprise RAG Import Service | `8000` | 文档导入：解析、切分、向量化、入库 |
| Query Service  | Enterprise RAG Query Service  | `8001` | 知识问答：检索、重排、Agent 编排、生成 |

**核心能力**：
- 多格式文档解析（基于 MinerU / magic-pdf）
- 文本切分与向量化（BGE-M3 Embedding）
- 向量检索 + 重排序（BGE Reranker v2-m3）
- 多路召回与 Agent 编排（LangGraph / OpenAI Agents）
- 联网搜索增强（DashScope WebSearch MCP）
- 视觉语言模型支持（qwen-vl-max）

**运行环境**：Python ≥ 3.11，使用 `uv` 作为包管理工具。

---

## 2. 项目目录结构

项目采用 **分层 + 模块化** 的目录组织方式，整体遵循领域驱动 / 清晰架构（Clean Architecture）思想：

```
ai_0119_rag/
├── app/                            # 应用主包
│   ├── api/                        # API 接入层
│   │   ├── http/                   #   HTTP 路由（FastAPI）
│   │   └── schemas/                #   请求 / 响应数据模型（Pydantic）
│   ├── infra/                      # 基础设施层（外部资源接入）
│   │   ├── config/                 #   配置加载
│   │   ├── llm/                    #   LLM / VL 模型接入
│   │   ├── document_parse/         #   文档解析（MinerU）
│   │   ├── object_storage/         #   对象存储（MinIO）
│   │   ├── persistence/            #   持久化（MongoDB）
│   │   └── vectorstore/            #   向量库（Milvus）
│   ├── process/                    # 流程编排层（应用服务）
│   │   ├── import_/                #   导入流程
│   │   │   ├── agent/              #     导入侧 Agent 编排
│   │   │   └── page/               #     导入侧页面/分页处理
│   │   └── query/                  #   查询流程
│   │       ├── agent/              #     查询侧 Agent 编排
│   │       └── page/               #     查询侧页面/分页处理
│   ├── rag/                        # RAG 领域层（核心业务逻辑）
│   │   ├── import_/                #   导入领域逻辑
│   │   └── query/                  #   查询领域逻辑
│   ├── resources/                  # 静态资源
│   │   └── prompts/                #   Prompt 模板
│   └── shared/                     # 共享内核（跨层通用能力）
│       ├── clients/                #   外部客户端封装
│       ├── config/                 #   通用配置
│       ├── model/                  #   通用数据模型
│       ├── runtime/                #   运行时上下文
│       ├── tool/                   #   工具集（供 Agent 调用）
│       └── utils/                  #   通用工具函数
├── test/                           # 测试包
├── .env.example                    # 环境变量示例
├── pyproject.toml                  # 项目元数据与依赖声明
└── uv.lock                         # uv 锁定文件
```

> 说明：当前仓库各模块以 `__init__.py` 形式占位，整体处于 **架构骨架已搭建、业务实现待填充** 的阶段。本文档依据目录结构、依赖声明与配置项对架构意图与模块职责进行完整描述。

---

## 3. 技术栈

依赖来源：`pyproject.toml`，使用清华 PyPI 镜像源。

### 3.1 Web 与服务
| 依赖 | 用途 |
|------|------|
| `fastapi` | Web 框架，构建 HTTP API |
| `uvicorn` | ASGI 服务器，运行 FastAPI 应用 |
| `python-multipart` | 表单 / 文件上传支持 |

### 3.2 LLM 编排与 Agent
| 依赖 | 用途 |
|------|------|
| `langchain` | LLM 应用编排框架 |
| `langchain-community` | LangChain 社区集成 |
| `langchain-openai` | OpenAI 兼容 LLM 接入（qwen-plus 等） |
| `langgraph` | 基于图的状态机式 Agent 编排 |
| `openai-agents` | OpenAI Agents SDK |
| `langchain-mcp-adapters` | MCP（Model Context Protocol）工具适配 |
| `dashscope` | 阿里云 DashScope（qwen 模型 + WebSearch MCP） |

### 3.3 Embedding 与 Reranker
| 依赖 | 用途 |
|------|------|
| `flagembedding` | BGE 系列模型（BGE-M3、BGE Reranker） |
| `transformers` | 模型加载与推理 |
| `torch` / `torchaudio` / `torchvision` | 深度学习后端 |
| `modelscope` | 模型下载（魔搭社区） |
| `datasets` | 数据集处理 |

### 3.4 向量库与存储
| 依赖 | 用途 |
|------|------|
| `pymilvus` / `pymilvus-model` | Milvus 向量数据库客户端 |
| `pymongo` | MongoDB 文档数据库客户端 |
| `minio` | MinIO 对象存储客户端 |

### 3.5 文档解析
| 依赖 | 用途 |
|------|------|
| `magic-pdf` | PDF 解析 |
| `mineru-kie-sdk` | MinerU KIE（关键信息抽取）SDK |

### 3.6 基础工具
| 依赖 | 用途 |
|------|------|
| `loguru` | 日志 |
| `python-dotenv` | `.env` 加载 |
| `numpy` / `pandas` | 数值与数据处理 |
| `regex` | 正则处理 |
| `requests` | HTTP 请求 |
| `grandalf` | 图布局（可用于流程可视化） |

---

## 4. 整体架构

系统采用 **分层架构 + CQRS（导入 / 查询分离）** 设计。

### 4.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                      API 接入层 (api)                     │
│        HTTP 路由 · 请求/响应 Schema · 参数校验            │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                  流程编排层 (process)                     │
│       import_ / query 的 Agent 与 Page 编排              │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                  RAG 领域层 (rag)                         │
│        导入领域逻辑 · 查询领域逻辑（核心业务）             │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                基础设施层 (infra)                          │
│  LLM · 向量库 · 持久化 · 对象存储 · 文档解析 · 配置        │
└─────────────────────────────────────────────────────────┘

        共享内核 (shared) 横跨所有层：clients · config · model ·
                              runtime · tool · utils
        静态资源 (resources/prompts) 提供统一 Prompt 管理
```

**分层职责约束**：
- `api` 仅负责协议接入与数据校验，不含业务逻辑；
- `process` 负责流程编排（调用 Agent / 分页处理），协调领域层与基础设施；
- `rag` 封装核心 RAG 领域逻辑（切分策略、召回策略、重排策略等）；
- `infra` 屏蔽外部资源差异，向上提供统一接口；
- `shared` 提供跨层复用的通用能力，不依赖具体业务层。

### 4.2 导入 / 查询分离（CQRS）

系统将「写」（文档导入）与「读」（知识问答）拆分为两个独立服务，各自拥有完整的 `process → rag` 链路，互不干扰、可独立扩缩容：

```
   ┌───────────── Import Service (8000) ─────────────┐
   │  api → process/import_ → rag/import_ → infra    │
   │  职责: 文档解析 → 切分 → 向量化 → 入库           │
   └─────────────────────────────────────────────────┘

   ┌───────────── Query Service (8001) ──────────────┐
   │  api → process/query → rag/query → infra        │
   │  职责: 查询理解 → 多路召回 → 重排 → Agent 生成   │
   └─────────────────────────────────────────────────┘
```

---

## 5. 模块职责说明

### 5.1 `app/api` — API 接入层

| 子模块 | 职责 |
|--------|------|
| `api/http` | 基于 FastAPI 的 HTTP 路由定义，承接 Import / Query 两个服务的对外接口；处理文件上传、CORS、请求分发 |
| `api/schemas` | 基于 Pydantic 的请求体 / 响应体模型定义，负责参数校验与序列化 |

### 5.2 `app/process` — 流程编排层

按导入 / 查询两条链路分别组织，每条链路下含 `agent` 与 `page` 两个子模块：

| 子模块 | 职责 |
|--------|------|
| `process/import_/agent` | 导入侧的 Agent 编排：驱动文档解析、切分、入库等步骤的状态流转 |
| `process/import_/page` | 导入侧的分页 / 批处理逻辑：处理大文档分块入库、进度推进 |
| `process/query/agent` | 查询侧的 Agent 编排：基于 LangGraph / OpenAI Agents 编排「检索 → 重排 → 生成 → 联网搜索」流程 |
| `process/query/page` | 查询侧的分页 / 上下文管理：管理对话上下文、分页结果聚合 |

### 5.3 `app/rag` — RAG 领域层

| 子模块 | 职责 |
|--------|------|
| `rag/import_` | 导入领域逻辑：文档切分策略、元数据抽取、向量化与入库规则 |
| `rag/query` | 查询领域逻辑：查询改写、多路召回策略、重排策略、上下文组装 |

### 5.4 `app/infra` — 基础设施层

| 子模块 | 对应外部资源 | 职责 |
|--------|-------------|------|
| `infra/config` | `.env` | 集中加载与暴露配置项（基于 python-dotenv） |
| `infra/llm` | OpenAI 兼容 API / DashScope | LLM（qwen-plus）与 VL（qwen-vl-max）模型客户端封装 |
| `infra/document_parse` | MinerU API | 文档解析服务接入，输出结构化文本与版面信息 |
| `infra/object_storage` | MinIO | 图片 / 文件对象存储，管理抽取出的图片资源 |
| `infra/persistence` | MongoDB | 业务元数据持久化（文档记录、切片元信息等） |
| `infra/vectorstore` | Milvus | 向量存储与检索，管理 3 个 Collection |

### 5.5 `app/shared` — 共享内核

| 子模块 | 职责 |
|--------|------|
| `shared/clients` | 外部服务客户端的统一封装与复用 |
| `shared/config` | 通用配置常量与运行参数 |
| `shared/model` | 跨层共享的数据模型 / DTO |
| `shared/runtime` | 运行时上下文（请求上下文、生命周期资源） |
| `shared/tool` | Agent 可调用的工具集（如联网搜索工具） |
| `shared/utils` | 通用工具函数（字符串、IO、时间等） |

### 5.6 `app/resources` — 静态资源

| 子模块 | 职责 |
|--------|------|
| `resources/prompts` | 统一管理各场景的 Prompt 模板（导入抽取、查询问答、查询改写等） |

---

## 6. 关键组件与设计说明

### 6.1 模型组件

| 组件 | 模型 / 资源 | 设备 | 用途 |
|------|------------|------|------|
| LLM | `qwen-plus`（OpenAI 兼容） | 远程 | 文本理解与生成 |
| VL 模型 | `qwen-vl-max` | 远程 | 图像理解（文档图片、版面） |
| Embedding | `BAAI/bge-m3`（本地 `./models/bge-m3`） | `cpu`（可配） | 文本向量化 |
| Reranker | `bge-reranker-v2-m3`（本地 `./models/bge-reranker-v2-m3`） | `cpu`（可配） | 召回结果重排序 |

> Embedding / Reranker 均支持 `BGE_DEVICE`（cpu/cuda）与 `BGE_FP16`（半精度）配置，便于在 CPU 与 GPU 环境间切换。

### 6.2 Milvus 向量库设计

系统在 Milvus 中维护 **3 个 Collection**，体现了「切片 + 实体 + 条目名」的多粒度检索设计：

| Collection 名 | 配置项 | 推测职责 |
|---------------|--------|----------|
| `kb_chunks` | `CHUNKS_COLLECTION` | 文档切片向量，粒度最细，支撑细粒度语义召回 |
| `kb_entities` | `ENTITY_NAME_COLLECTION` | 实体名向量，支撑基于实体/关键词的精确召回 |
| `kb_item_names` | `ITEM_NAME_COLLECTION` | 条目/文档名向量，支撑文档级别的召回与定位 |

该多粒度结构支持 **多路召回（multi-route retrieval）**，由 `rag/query` 层统一融合后交由 Reranker 重排。

### 6.3 MongoDB 持久化

| 配置项 | 值 | 说明 |
|--------|----|----|
| `MONGO_URL` | `mongodb://127.0.0.1:27017` | MongoDB 连接地址 |
| `MONGO_DB_NAME` | `enterprise_rag` | 业务数据库名 |

存储文档元数据、切片元信息、导入任务状态等结构化数据，与 Milvus（向量）形成互补。

### 6.4 MinIO 对象存储

| 配置项 | 说明 |
|--------|------|
| `MINIO_ENDPOINT` | `127.0.0.1:9000`（不带协议） |
| `MINIO_BUCKET_NAME` | `enterprise-rag` |
| `MINIO_IMG_DIR` | `/kb-images`（图片存储目录） |
| `MINIO_SECURE` | `False`（是否启用 HTTPS） |

用于存储文档解析过程中抽取出的图片资源，供 VL 模型与前端展示使用。

### 6.5 Agent 编排

查询侧基于 **LangGraph + OpenAI Agents** 进行多步编排，典型节点包括：
1. **查询理解 / 改写**：对用户问题进行归一化与扩展；
2. **多路召回**：并行检索 `kb_chunks` / `kb_entities` / `kb_item_names`；
3. **重排**：使用 BGE Reranker 对召回结果重排序；
4. **工具调用**：必要时通过 MCP 调用 DashScope WebSearch 进行联网搜索（`node_web_search_mcp`）；
5. **生成**：结合上下文与检索结果生成最终回答。

`shared/tool` 模块负责封装 Agent 可调用的工具（如联网搜索工具），`langchain-mcp-adapters` 负责将 MCP 服务适配为 LangChain 工具。

---

## 7. 依赖关系

### 7.1 模块间依赖（自上而下）

```
api  ──►  process  ──►  rag  ──►  infra
  │           │          │         │
  └───────────┴──────────┴─────────┴──►  shared
                 resources/prompts  ◄── (被 rag/process 引用)
```

- **纵向**：上层依赖下层，禁止反向依赖（如 `infra` 不得依赖 `process`）。
- **横向**：各业务模块（`import_` / `query`）相互独立，不直接耦合。
- **共享**：所有层均可依赖 `shared`；`shared` 不依赖任何业务层。

### 7.2 基础设施依赖图

```
                    ┌───────────────┐
                    │   infra/llm   │ ──► OpenAI 兼容 API (qwen-plus / qwen-vl-max)
                    └───────────────┘
                    ┌───────────────┐
                    │ infra/vector  │ ──► Milvus (kb_chunks / kb_entities / kb_item_names)
                    │   store       │
                    └───────────────┘
   rag/process ────►┌───────────────┐
                    │ infra/persist │ ──► MongoDB (enterprise_rag)
                    └───────────────┘
                    ┌───────────────┐
                    │ infra/object  │ ──► MinIO (enterprise-rag / /kb-images)
                    │   _storage    │
                    └───────────────┘
                    ┌───────────────┐
                    │ infra/doc_parse│──► MinerU API
                    └───────────────┘
                    ┌───────────────┐
                    │  shared/tool  │ ──► DashScope WebSearch MCP
                    └───────────────┘
```

### 7.3 本地模型依赖

- Embedding / Reranker 模型需预先下载至本地：
  - `./models/bge-m3`
  - `./models/bge-reranker-v2-m3`
- 可通过 `modelscope` / `flagembedding` 自动拉取（`BGE_M3=BAAI/bge-m3` 提供 HuggingFace / ModelScope 标识）。

---

## 8. 配置项说明

所有配置通过 `.env` 文件管理，示例见 `.env.example`。配置分为以下几组：

### 8.1 应用基础配置
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `IMPORT_APP_NAME` | Enterprise RAG Import Service | 导入服务名 |
| `QUERY_APP_NAME` | Enterprise RAG Query Service | 查询服务名 |
| `APP_ENV` | `dev` | 运行环境 |
| `APP_HOST` | `0.0.0.0` | 监听地址 |
| `IMPORT_APP_PORT` | `8000` | 导入服务端口 |
| `QUERY_APP_PORT` | `8001` | 查询服务端口 |
| `CORS_ORIGINS` | `*` | CORS 允许来源 |

### 8.2 LLM / VL 模型配置
| 配置项 | 说明 |
|--------|------|
| `OPENAI_BASE_URL` | OpenAI 兼容 LLM 端点 |
| `OPENAI_API_KEY` | API Key（同时被 DashScope MCP 复用） |
| `LLM_DEFAULT_MODEL` | 默认 LLM（`qwen-plus`） |
| `LLM_DEFAULT_TEMPERATURE` | 采样温度（须可转 float，默认 `0.1`） |
| `VL_MODEL` | 视觉语言模型（`qwen-vl-max`） |

### 8.3 Embedding 配置
| 配置项 | 说明 |
|--------|------|
| `BGE_M3_PATH` | 本地模型路径（`./models/bge-m3`） |
| `BGE_M3` | 模型标识（`BAAI/bge-m3`） |
| `BGE_DEVICE` | 推理设备（`cpu`） |
| `BGE_FP16` | 是否半精度（`False`） |

### 8.4 Reranker 配置
| 配置项 | 说明 |
|--------|------|
| `BGE_RERANKER_LARGE` | 本地模型路径（`./models/bge-reranker-v2-m3`） |
| `BGE_RERANKER_DEVICE` | 推理设备（`cpu`） |
| `BGE_RERANKER_FP16` | 是否半精度（`False`） |

### 8.5 Milvus 配置
| 配置项 | 说明 |
|--------|------|
| `MILVUS_URL` | Milvus 地址（`http://127.0.0.1:19530`） |
| `CHUNKS_COLLECTION` | 切片集合（`kb_chunks`） |
| `ENTITY_NAME_COLLECTION` | 实体集合（`kb_entities`） |
| `ITEM_NAME_COLLECTION` | 条目名集合（`kb_item_names`） |

### 8.6 Mongo 配置
| 配置项 | 说明 |
|--------|------|
| `MONGO_URL` | MongoDB 地址 |
| `MONGO_DB_NAME` | 数据库名（`enterprise_rag`） |

### 8.7 MinIO 配置
| 配置项 | 说明 |
|--------|------|
| `MINIO_ENDPOINT` | 端点（不带协议，由 `MINIO_SECURE` 决定） |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 访问凭证 |
| `MINIO_BUCKET_NAME` | 桶名（`enterprise-rag`） |
| `MINIO_IMG_DIR` | 图片目录（建议以 `/` 开头） |
| `MINIO_SECURE` | 是否启用 HTTPS |

### 8.8 MinerU 配置
| 配置项 | 说明 |
|--------|------|
| `MINERU_BASE_URL` | MinerU 服务端点 |
| `MINERU_API_TOKEN` | MinerU API Token |

### 8.9 MCP / WebSearch 配置
| 配置项 | 说明 |
|--------|------|
| `MCP_DASHSCOPE_BASE_URL` | DashScope WebSearch MCP 端点（当前代码未实际使用，建议保留） |

### 8.10 项目根目录
| 配置项 | 说明 |
|--------|------|
| `PROJECT_ROOT` | 项目根目录；未设置时代码自动递归查找 `.env` |

---

## 9. 核心业务流程

### 9.1 导入流程（Import Service）

```
用户上传文档
    │
    ▼
api/http 接收请求 ──► schemas 校验
    │
    ▼
process/import_/agent 编排
    │
    ├──► infra/document_parse (MinerU) 解析为结构化文本 + 图片
    │
    ├──► infra/object_storage (MinIO) 存储图片
    │
    ▼
rag/import_ 领域逻辑
    │
    ├──► 文本切分 (chunking)
    ├──► 实体 / 条目名抽取
    ▼
infra/llm (BGE-M3) 向量化
    │
    ├──► infra/vectorstore (Milvus) 写入 kb_chunks / kb_entities / kb_item_names
    └──► infra/persistence (MongoDB) 写入元数据
```

### 9.2 查询流程（Query Service）

```
用户提问
    │
    ▼
api/http 接收请求 ──► schemas 校验
    │
    ▼
process/query/agent 编排 (LangGraph)
    │
    ├──► 查询理解 / 改写
    │
    ├──► rag/query 多路召回
    │       ├──► Milvus: kb_chunks      (细粒度语义)
    │       ├──► Milvus: kb_entities    (实体精确)
    │       └──► Milvus: kb_item_names  (文档级)
    │
    ├──► BGE Reranker 重排
    │
    ├──► (可选) shared/tool: WebSearch MCP 联网搜索
    │
    ▼
infra/llm (qwen-plus) 生成回答
    │
    ▼
返回结果（含引用来源 / 上下文）
```

---

## 10. 项目运行方式

### 10.1 环境准备

**Python 版本**：≥ 3.11

**包管理工具**：[uv](https://github.com/astral-sh/uv)

### 10.2 安装依赖

项目使用清华镜像源（已在 `pyproject.toml` 中配置 `tsinghua` 为默认源）：

```bash
# 安装 uv（如未安装）
pip install uv

# 同步项目依赖（依据 uv.lock 锁定版本）
uv sync
```

### 10.3 配置环境变量

复制示例配置并按实际环境修改：

```bash
cp .env.example .env
```

需重点配置以下外部依赖服务：

1. **LLM 服务**：填写 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY`；
2. **Milvus**：启动 Milvus 实例并配置 `MILVUS_URL`；
3. **MongoDB**：启动 MongoDB 实例并配置 `MONGO_URL`；
4. **MinIO**：启动 MinIO 实例并配置 endpoint / 凭证；
5. **MinerU**：填写 `MINERU_BASE_URL` 与 `MINERU_API_TOKEN`；
6. **本地模型**：将 BGE-M3 与 BGE Reranker 模型放置到 `./models/` 对应目录（或通过 `modelscope` 下载）。

### 10.4 启动服务

系统包含两个独立服务，分别启动：

```bash
# 启动导入服务（端口 8000）
uv run uvicorn app.api.http:import_app --host 0.0.0.0 --port 8000

# 启动查询服务（端口 8001）
uv run uvicorn app.api.http:query_app --host 0.0.0.0 --port 8001
```

> 注：以上入口名称（`import_app` / `query_app`）为依据 `.env` 中双服务配置的推断；实际 ASGI app 对象以 `app/api/http` 模块最终实现为准。

### 10.5 开发与测试

```bash
# 运行测试
uv run pytest

# 开发模式（热重载）
uv run uvicorn app.api.http:import_app --reload --port 8000
```

### 10.6 部署架构示意

```
                       ┌───────────────┐
                       │   负载均衡 / 网关  │
                       └───┬───────┬───┘
                           │       │
              ┌────────────▼─┐  ┌──▼────────────┐
              │ Import Service│  │ Query Service  │
              │  (FastAPI)    │  │  (FastAPI)     │
              │  :8000        │  │  :8001         │
              └──────┬────────┘  └────────┬───────┘
                     │                    │
        ┌────────────┴────────────────────┴────────────┐
        │              共享基础设施层                    │
        ├──────────┬──────────┬──────────┬──────────────┤
        │  Milvus  │ MongoDB  │  MinIO   │ MinerU / LLM │
        │ :19530   │ :27017   │ :9000    │  (远程 API)  │
        └──────────┴──────────┴──────────┴──────────────┘
```

---

## 附录：架构设计要点小结

1. **导入 / 查询分离（CQRS）**：两条链路独立部署、独立扩容，读写职责清晰。
2. **分层解耦**：`api → process → rag → infra` 单向依赖，`shared` 横向复用，便于维护与测试。
3. **基础设施可替换**：`infra` 层屏蔽 Milvus / MongoDB / MinIO / MinerU 的具体实现，上层只面向抽象接口。
4. **多粒度向量检索**：三 Collection 设计支撑多路召回，配合 Reranker 提升准确率。
5. **Agent 编排 + 工具增强**：基于 LangGraph / OpenAI Agents 编排查询流程，通过 MCP 接入联网搜索等外部能力。
6. **本地模型 + 远程 LLM 混合**：Embedding / Reranker 本地部署（可 CPU/GPU 切换），LLM / VL 走远程 API，兼顾成本与可控性。
7. **配置集中化**：全部配置通过 `.env` 管理，支持自动定位项目根目录。
