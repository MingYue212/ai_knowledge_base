# HANDOFF — knowledge-zhanggui 无缝衔接交接文档

> 写于 2026-09-29。供新会话（或人工）继续开发使用。
> **新会话第一步：读完本文档，执行第 2 节验证命令确认真实状态，再动代码。**

## 1. 项目一句话

企业知识库 RAG（包名 `ai_0119_rag`）：LangGraph 双图（导入 `import_` / 查询 `query`）+ Milvus 混合检索 + MinIO + BGE-M3 + FastAPI 双服务。
用户已在别的电脑端到端跑通全部流程；本机**不起本地服务**（Milvus/MinIO/模型下载都跳过），以写代码 + 面试复习为主。

## 2. ⚠️ 先验证真实状态（上一会话末尾有一轮不可靠的写入）

在项目根目录执行：

```bash
cd ~/Documents/Code/PROJECT/knowledge-zhanggui
git status --short
find app test -name "*.py" -print0 | xargs -0 python3 -m py_compile && echo COMPILE_OK
grep -n "hyde" app/process/query/agent/state.py
tail -3 app/shared/config/settings_config.py     # 不应再有 print(settings)
uv run python -c "from app.process.query.agent.main_graph import query_app; print(sorted(query_app.get_graph().nodes))"
uv run python -c "from app.process.import_.agent.main_graph import import_app; print('IMPORT OK')"
```

预期：编译全过；查询图 5 个业务节点注册（若 4.3/4.4 已做则 7 个）；settings_config 无 print。
**若发现垃圾文件**（错误路径下的 state.py / item_name_service.py 等，如 `app/rag/query/agent/` 这种不存在的路径），直接删除。

## 3. 已完成且验证过的部分

### 第 0 步 导入链（全部完成，主图 7 节点注册冒烟通过）
- `app/process/import_/agent/nodes/`：`node_bge_embedding`、`node_import_milvus`（新增）；`node_item_name_recognition`（补齐空文件，薄节点）
- `app/rag/import_/`：`embedding_service`、`milvus_import_service`、`item_name_service`（**直通版**，见 4.2）
- `app/shared/clients/milvus_utils.py`（客户端单例，新增）；`app/infra/vectorstore/milvus_gateway.py`（重写：幂等建集合 + 混合向量插入，HNSW+SPARSE_INVERTED_INDEX，IP 度量）

### 第 2 步 查询链核心（完成，5 节点冒烟通过）
- 线性图：`query_entry → item_name_confirm → search_embedding → rerank → answer_output`
- 服务层 `app/rag/query/`：`config`（SEARCH_TOP_K=20 / RERANK_TOP_N=5 / MAX_HISTORY_TURNS=5 / ANSWER_CONTEXT_MAX_CHARS=4000）、`query_rewrite_service`（指代消解+改写，json_mode，失败降级）、`search_service`（混合检索）、`rerank_service`（交叉编码器精排，compute_score 单条返回 float 需包装）、`answer_service`（上下文拼装+兜底话术）
- `embedding_utils.generate_query_embeddings`（encode_queries，与文档侧 encode_documents 对应）
- `milvus_utils.create_hybrid_search_requests + hybrid_search`（RRFRanker k=60，服务端融合稠密+稀疏）
- `test/02_test_query_graph_running.py` 冒烟

## 4. 剩余工作规格（按序做，一次一个文件，写完立即 py_compile）

### 4.1 settings_config.py 删 print
末行 `print(settings)` 删除（若第 2 节验证发现已删则跳过）。

### 4.2 主体识别做实（重写 `app/rag/import_/item_name_service.py`）
- 现状：`recognize_item_name(state)` 直通，item_name 留空
- 目标：取前 `ITEM_NAME_CONTEXT_CHUNK_K=5` 个切片（常量在 `app/rag/import_/config.py`，累计上限 `ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS=2000` 字符）→ `load_prompt("product_recognition_system")` 做 system（`SystemMessage`）、`load_prompt("item_name_recognition", file_title=..., context=...)` 做 user（`HumanMessage`，均来自 `langchain_core.messages`）→ `get_llm_client().invoke([...])` → 清洗（取首行、去引号/句号）→ 写 `state["item_name"]`
- 防御：LLM 失败/识别为空 → `item_name=""`，**不抛异常**（`node_import_milvus` 的主体索引分支按非空判断自动跳过）
- 节点不用改（已委托本服务）

### 4.3 HyDE 路（新文件 `app/rag/query/hyde_service.py` + `node_search_embedding_hyde.py`）
- `search_by_hyde(state)`：用 rewritten_query（回退 query）→ `load_prompt("hyde_prompt", rewritten_query=query)` → LLM 生成假设性回答 → 写 `state["hyde_answer"]` → `generate_query_embeddings` → `create_hybrid_search_requests` → `milvus_gateway.search_chunks` → 写 `state["hyde_context"]`
- 防御：失败 → `hyde_context=[]`，不抛异常（"以文搜文"增强路）
- 节点：薄节点 `@node_log("node_search_embedding_hyde")`（task_utils 中文映射已有：切片搜索(假设性文档)）

### 4.4 RRF 融合（新文件 `app/rag/query/rrf_service.py` + `node_rrf.py`）
- `fuse_search_results(state)`：主路 `search_context` + HyDE 路 `hyde_context`，`score = Σ 1/(60+rank)`（rank 从 1 起），**以 Milvus 主键 id 去重**（同 id 多路命中分数累加），降序截断 `SEARCH_TOP_K` 写回 `state["search_context"]`；单路为空退化为另一路
- ⚠️ 前置：检索结果需带 id —— `milvus_gateway.search_chunks` 的 `output_fields` 和 `milvus_utils.hybrid_search` 的返回字典都要加 `"id"`（第 2 节验证时顺带确认是否已加）
- 状态字段：`state.py` 需有 `hyde_answer: str` / `hyde_context: list[dict]`（TypedDict 与默认字典两处）
- 节点：`@node_log("node_rrf")`（映射已有：倒排融合）

### 4.5 查询主图接成 7 节点（重写 `app/process/query/agent/main_graph.py`）
`entry → confirm → search_embedding → search_embedding_hyde → rrf → rerank → answer_output`

### 4.6 FastAPI 层（`app/api/http/` 两文件 + `app/api/schemas/` 两文件）
- 复用 `app/shared/utils/sse_utils.py` 三件套：`create_sse_queue(session_id)` / `push_to_session(session_id, event, data)` / `sse_generator(session_id, request)`（queue.Queue 线程安全，配合后台线程；`SSEEvent.CLOSE = "__close__"` 是关闭信号）
- **SSE 模式**：请求进来 → `create_sse_queue(task_id)` → `threading.Thread(daemon=True)` 里用 `graph.stream(state, stream_mode="updates")` 逐节点 `push_to_session(PROGRESS, {"node": 节点名})`，结束推 `FINAL` / 异常推 `ERROR`，最后推 `CLOSE` → 立即返回 `StreamingResponse(sse_generator(task_id, request), media_type="text/event-stream")`
- `import_app.py`（端口 `settings.import_app_port`）：`POST /import`（UploadFile 存 `output/uploads/`，校验 .md/.pdf，文件名用 `Path(...).name` 防穿越）+ `GET /import/progress/{task_id}`（task_utils 轮询：`get_task_status` / `get_done_task_list` / `get_running_task_list`，中文映射直接可用）+ `GET /import/progress/stream/{task_id}`（SSE）
- `query_app.py`（端口 `settings.query_app_port`）：`POST /query`（同步 invoke，返回 answer/item_names/rewritten_query）+ `POST /query/sse`（SSE，FINAL 事件带答案）
- schemas：`QueryRequest{query, history:[{role,content}]}` / `QueryResponse` / `ImportResponse{task_id}` / `ProgressResponse{status, done_list, running_list}`
- CORS：`settings.cors_origins`；各文件 `__main__` 里 `uvicorn.run(app, host=settings.app_host, port=...)`
- **联网搜索那一路（node_web_search_mcp）用户明确不做**；KG（node_query_kg/node_join）也不做

## 5. 代码约定（严格遵守）

- 分层：节点薄（`@node_log("节点名")` + `add_running_task`/`add_done_task` + 委托服务）→ 服务在 `app/rag/{import_,query}/`（`@step_log` + 防御性校验 + 中文日志）→ 网关在 `app/infra/` 与 `app/shared/clients/`
- 节点名必须与 `task_utils._NODE_NAME_TO_CN` 中文映射一致（缺失才补映射）
- state：TypedDict + 模块级默认字典 + `create_default_state(**kwargs)`（深拷贝+update）
- `generate_embeddings(texts)` → `{"dense": [[float]], "sparse": [{int: float}]}`；查询侧用 `generate_query_embeddings`
- chunk 字段：`title / content / file_title / parent_title / part`
- 每个文件：中文 docstring + `if __name__ == '__main__'` 测试块（风格照抄现有文件）
- prompt 模板在 `app/resources/prompts/*.prompt`，`load_prompt(name, **kwargs)`（str.format 渲染；`{{}}` 是转义花括号）
- 日志：`from app.shared.runtime.logger import logger, node_log, step_log`；任务进度：`from app.shared.utils.task_utils import add_running_task, add_done_task`

## 6. ⚠️ 工作纪律（上一会话的血泪教训，必须遵守）

1. **显示层会混淆**：工具输出的路径拼写、文件内容可能被污染/串扰（本会话实锤发生过）。判断文件好坏**只认** `python3 -m py_compile` + 真实 `import` + `grep`，绝不按显示内容判断文件损坏，绝不因此触发删除→重写循环
2. **一次一个文件**：写完立即编译验证，通过才写下一个；**禁止**大批量凭记忆生成代码
3. **删除前三重确认**：只删本会话创建的文件；`git status` + `ls` 交叉验证后再删
4. 路径以 `git ls-files` 为权威

## 7. 关键文件地图

```
app/
├── process/import_/agent/     # 导入图：main_graph.py + state.py + nodes/（7节点全）
├── process/query/agent/       # 查询图：main_graph.py + state.py + nodes/（5节点，待扩7）
├── rag/import_/               # 导入服务：entry/pdf_parse/split/enrich_markdown_images/embedding/milvus_import/item_name(+config)
├── rag/query/                 # 查询服务：query_rewrite/search/rerank/answer(+config)（待加 hyde/rrf）
├── infra/vectorstore/         # milvus_gateway.py（集合schema+插入+混合检索）
├── infra/config/providers.py  # infra_config 聚合（infra_config.milvus_config.*）
├── shared/clients/            # milvus_utils.py（客户端单例+hybrid_search）、minio_utils.py（自动建桶）
├── shared/model/              # embedding_utils（generate_embeddings/generate_query_embeddings）、reranker_utils、lm_utils（get_llm_client(model=None, json_mode=False)）
├── shared/utils/              # task_utils（进度+中文映射）、sse_utils（SSE三件套）、load_prompt 在 shared/runtime/
├── shared/config/             # settings_config（AppSettings：import_app_port=8000/query_app_port=8001/cors_origins）、milvus_config、embedding_config、reranker_config
├── resources/prompts/         # rewritten_query_and_itemnames / hyde_prompt / answer_out / item_name_recognition / product_recognition_system / rerank_text_refine
└── api/http/ + api/schemas/   # 空，待建 FastAPI 层
test/                          # 01 导入冒烟 / 02 查询冒烟
```

## 8. 面试考点速查（复习用）

1. **两阶段检索**：召回段双编码器（快、可全库、向量离线）Top20 → 精排段交叉编码器（准、逐对、贵）Top5
2. **混合检索**：dense 管语义、sparse 管精确词（型号/错误码）；BGE-M3 一次出两种向量；encode_documents/encode_queries 分侧
3. **RRF**：`Σ 1/(k+rank)`，k=60；只看排名不看分数 → 规避两路量纲不可比；服务端融稠密+稀疏、客户端融多路查询
4. **查询改写**：多轮指代消解（json_mode）；失败降级回退原 query —— "LLM 挂了怎么办"的标准答案
5. **HyDE**：答案与答案的距离 < 问题与答案的距离；以文搜文补主路漏召回
6. **工程防御**：幂等建集合、空召回兜底话术、上下文截断+来源标注、进度 SSE（queue.Queue + graph.stream(updates)）
