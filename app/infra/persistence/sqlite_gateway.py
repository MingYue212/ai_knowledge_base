"""
SQLite 元数据网关：知识组 / 人员 / 授权关系的持久化（权限体系的元数据层）。

设计说明：
- 只存"轻量元数据"（组、人员、授权关系），向量与正文仍在 Milvus
- 连接按次开关（每次操作新建连接），天然线程安全，适配 FastAPI 后台线程
- knowledge_groups 建库即种子「默认知识组」，所有人可见（检索时恒定并入可见组）
- 授权模型：group_grants(group_id, subject_type ∈ {department, position, user}, subject_value)，
  一个组可按部门 / 职位 / 具体人三种方式授权，命中任一即可访问
- db_path 可注入（单测用临时库），默认 PROJECT_ROOT/data/metadata.db
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.shared.runtime.logger import logger
from app.shared.utils.path_util import PROJECT_ROOT

# 授权主体类型白名单
SUBJECT_TYPES = ("department", "position", "user")
DEFAULT_GROUP_NAME = "默认知识组"

# 回答类型（引擎明确上报，不靠猜文案；借鉴管理平台讲义模块十的口径）
ANSWER_TYPE_NORMAL = 0           # 正常回答
ANSWER_TYPE_NO_KNOWLEDGE = 1     # 资料缺失拒答（计入知识缺失）
ANSWER_TYPE_PERMISSION_DENIED = 2  # 权限拒答（预留，本期引擎不产生）
ANSWER_TYPE_CLARIFY = 3          # 澄清追问（预留，本期引擎不产生）
# 反馈状态
FEEDBACK_NONE = 0                # 未反馈
FEEDBACK_HELPFUL = 1             # 有帮助
FEEDBACK_UNHELPFUL = 2           # 无帮助（计入知识缺失）


class QueryLogNotFoundError(ValueError):
    """查询日志不存在（API 层转 404）。"""


class FeedbackNotAllowedError(ValueError):
    """反馈不被允许（仅正常回答可反馈，API 层转 400）。"""


class FeedbackForbiddenError(ValueError):
    """只能反馈本人的问答记录（API 层转 403）。"""

# 默认库路径（data/ 已在 .gitignore）
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "metadata.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_groups (
    group_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS users (
    user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    department TEXT NOT NULL DEFAULT '',
    position   TEXT NOT NULL DEFAULT '',
    is_admin   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS group_grants (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id      INTEGER NOT NULL REFERENCES knowledge_groups(group_id) ON DELETE CASCADE,
    subject_type  TEXT NOT NULL CHECK (subject_type IN ('department','position','user')),
    subject_value TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    UNIQUE(group_id, subject_type, subject_value)
);
CREATE TABLE IF NOT EXISTS query_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL DEFAULT '',
    user_id         INTEGER,
    query           TEXT NOT NULL,
    rewritten_query TEXT NOT NULL DEFAULT '',
    item_names      TEXT NOT NULL DEFAULT '[]',
    answer          TEXT NOT NULL DEFAULT '',
    answer_type     INTEGER NOT NULL DEFAULT 0,
    feedback_status INTEGER NOT NULL DEFAULT 0,
    gap_handled     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_query_logs_gap
    ON query_logs(answer_type, feedback_status, gap_handled);
"""


class SqliteGateway:
    """知识组 / 人员 / 授权元数据的 SQLite 网关（单例，连接按次开关）。"""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._default_group_id: int | None = None
        self._init_schema()
        logger.info(f"SQLite元数据网关就绪,库文件:{self.db_path}")

    @contextmanager
    def _connect(self):
        """按次开关的连接（row_factory 行转字典，外键级联开启）。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """建表 + 种子默认知识组（幂等）。"""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("INSERT OR IGNORE INTO knowledge_groups(name, description) VALUES (?, ?)",
                         (DEFAULT_GROUP_NAME, "所有用户可见的公共知识组"))

    # ---------------- 知识组 ----------------

    def get_default_group_id(self) -> int:
        """默认知识组 id（进程内缓存）。"""
        if self._default_group_id is None:
            with self._connect() as conn:
                row = conn.execute("SELECT group_id FROM knowledge_groups WHERE name = ?",
                                   (DEFAULT_GROUP_NAME,)).fetchone()
            if row is None:
                raise ValueError("默认知识组缺失,元数据库异常!")
            self._default_group_id = int(row["group_id"])
        return self._default_group_id

    def list_groups(self) -> list[dict]:
        """列出全部知识组。"""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM knowledge_groups ORDER BY group_id").fetchall()
        return [dict(row) for row in rows]

    def get_group(self, group_id: int) -> dict | None:
        """按 id 取知识组；不存在返回 None。"""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM knowledge_groups WHERE group_id = ?",
                               (int(group_id),)).fetchone()
        return dict(row) if row else None

    def create_group(self, name: str, description: str = "") -> dict:
        """新建知识组；组名重复抛 ValueError。"""
        name = (name or "").strip()
        if not name:
            raise ValueError("知识组名称不能为空!")
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO knowledge_groups(name, description) VALUES (?, ?)",
                    (name, (description or "").strip()),
                )
                group_id = int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            raise ValueError(f"知识组名称已存在:{name}") from e
        logger.info(f"知识组创建完成,id:{group_id},名称:{name}")
        return self.get_group(group_id)

    def update_group(self, group_id: int, name: str | None = None, description: str | None = None) -> dict:
        """更新知识组名称/描述；默认组不允许改名。"""
        group = self.get_group(group_id)
        if group is None:
            raise ValueError(f"知识组不存在:id={group_id}")
        new_name = (name if name is not None else group["name"]).strip()
        new_desc = description if description is not None else group["description"]
        if not new_name:
            raise ValueError("知识组名称不能为空!")
        if int(group_id) == self.get_default_group_id() and new_name != DEFAULT_GROUP_NAME:
            raise ValueError("默认知识组不允许改名!")
        try:
            with self._connect() as conn:
                conn.execute("UPDATE knowledge_groups SET name = ?, description = ? WHERE group_id = ?",
                             (new_name, new_desc, int(group_id)))
        except sqlite3.IntegrityError as e:
            raise ValueError(f"知识组名称已存在:{new_name}") from e
        logger.info(f"知识组更新完成,id:{group_id},名称:{new_name}")
        return self.get_group(group_id)

    def delete_group(self, group_id: int) -> None:
        """删除知识组（授权关系级联删除）；默认组与仍持有切片的组由 API 层拦截。"""
        group = self.get_group(group_id)
        if group is None:
            raise ValueError(f"知识组不存在:id={group_id}")
        if int(group_id) == self.get_default_group_id():
            raise ValueError("默认知识组不允许删除!")
        with self._connect() as conn:
            conn.execute("DELETE FROM knowledge_groups WHERE group_id = ?", (int(group_id),))
        if self._default_group_id is not None and int(group_id) == self._default_group_id:
            self._default_group_id = None
        logger.info(f"知识组删除完成,id:{group_id},名称:{group['name']}")

    # ---------------- 人员 ----------------

    @staticmethod
    def _user_dict(row) -> dict:
        data = dict(row)
        data["is_admin"] = bool(data.get("is_admin"))
        return data

    def list_users(self) -> list[dict]:
        """列出全部人员。"""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY user_id").fetchall()
        return [self._user_dict(row) for row in rows]

    def get_user(self, user_id: int) -> dict | None:
        """按 id 取人员；不存在返回 None。"""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (int(user_id),)).fetchone()
        return self._user_dict(row) if row else None

    def create_user(self, name: str, department: str = "", position: str = "", is_admin: bool = False) -> dict:
        """新建人员；姓名重复抛 ValueError。"""
        name = (name or "").strip()
        if not name:
            raise ValueError("人员姓名不能为空!")
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users(name, department, position, is_admin) VALUES (?, ?, ?, ?)",
                    (name, (department or "").strip(), (position or "").strip(), int(bool(is_admin))),
                )
                user_id = int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            raise ValueError(f"人员姓名已存在:{name}") from e
        logger.info(f"人员创建完成,id:{user_id},姓名:{name},部门:{department},职位:{position}")
        return self.get_user(user_id)

    def update_user(self, user_id: int, name: str | None = None, department: str | None = None,
                    position: str | None = None, is_admin: bool | None = None) -> dict:
        """更新人员信息；改名时同步更新按"具体人"授权的历史记录。"""
        user = self.get_user(user_id)
        if user is None:
            raise ValueError(f"人员不存在:id={user_id}")
        new_name = (name if name is not None else user["name"]).strip()
        if not new_name:
            raise ValueError("人员姓名不能为空!")
        new_dept = (department if department is not None else user["department"]).strip()
        new_pos = (position if position is not None else user["position"]).strip()
        new_admin = bool(is_admin) if is_admin is not None else user["is_admin"]
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE users SET name = ?, department = ?, position = ?, is_admin = ? WHERE user_id = ?",
                    (new_name, new_dept, new_pos, int(new_admin), int(user_id)),
                )
                if new_name != user["name"]:
                    conn.execute(
                        "UPDATE group_grants SET subject_value = ? WHERE subject_type = 'user' AND subject_value = ?",
                        (new_name, user["name"]),
                    )
        except sqlite3.IntegrityError as e:
            raise ValueError(f"人员姓名已存在:{new_name}") from e
        logger.info(f"人员更新完成,id:{user_id},姓名:{new_name}")
        return self.get_user(user_id)

    def delete_user(self, user_id: int) -> None:
        """删除人员，并清掉按"具体人"对其的授权记录。"""
        user = self.get_user(user_id)
        if user is None:
            raise ValueError(f"人员不存在:id={user_id}")
        with self._connect() as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (int(user_id),))
            conn.execute("DELETE FROM group_grants WHERE subject_type = 'user' AND subject_value = ?",
                         (user["name"],))
        logger.info(f"人员删除完成,id:{user_id},姓名:{user['name']}(授权已清理)")

    # ---------------- 授权 ----------------

    def list_grants(self, group_id: int) -> list[dict]:
        """列出某知识组的全部授权记录。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, group_id, subject_type, subject_value FROM group_grants "
                "WHERE group_id = ? ORDER BY subject_type, subject_value",
                (int(group_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_grant(self, group_id: int, subject_type: str, subject_value: str) -> dict:
        """给知识组追加一条授权（部门/职位/具体人）；重复授权抛 ValueError。"""
        subject_type = (subject_type or "").strip()
        subject_value = (subject_value or "").strip()
        if subject_type not in SUBJECT_TYPES:
            raise ValueError(f"授权类型不合法:{subject_type},仅支持{list(SUBJECT_TYPES)}")
        if not subject_value:
            raise ValueError("授权对象不能为空!")
        if self.get_group(group_id) is None:
            raise ValueError(f"知识组不存在:id={group_id}")
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO group_grants(group_id, subject_type, subject_value) VALUES (?, ?, ?)",
                    (int(group_id), subject_type, subject_value),
                )
        except sqlite3.IntegrityError as e:
            raise ValueError(f"该授权已存在:{subject_type}={subject_value}") from e
        logger.info(f"授权完成,组:{group_id},{subject_type}={subject_value}")
        return {"group_id": int(group_id), "subject_type": subject_type, "subject_value": subject_value}

    def remove_grant(self, group_id: int, subject_type: str, subject_value: str) -> None:
        """撤销一条授权；不存在时抛 ValueError。"""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM group_grants WHERE group_id = ? AND subject_type = ? AND subject_value = ?",
                (int(group_id), subject_type, subject_value),
            )
            if cur.rowcount == 0:
                raise ValueError(f"授权不存在:{subject_type}={subject_value}")
        logger.info(f"授权撤销完成,组:{group_id},{subject_type}={subject_value}")

    # ---------------- 权限解析 ----------------

    def resolve_user_group_ids(self, user_id: int) -> list[int]:
        """计算某人员可见的知识组 id 列表 = 默认组 ∪ 命中的授权组（按部门/职位/具体人）。

        防御性校验：
        - 人员不存在 —— 抛 ValueError（API 层转 400，前端身份选择器过期时触发）
        """
        user = self.get_user(user_id)
        if user is None:
            raise ValueError(f"人员不存在:id={user_id}")
        default_id = self.get_default_group_id()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT group_id FROM group_grants "
                "WHERE (subject_type = 'user' AND subject_value = ?) "
                "   OR (subject_type = 'department' AND subject_value = ?) "
                "   OR (subject_type = 'position' AND subject_value = ?)",
                (user["name"], user["department"], user["position"]),
            ).fetchall()
        group_ids = sorted({default_id, *(int(row["group_id"]) for row in rows)})
        logger.info(f"权限解析完成,用户:{user['name']},可见组:{group_ids}")
        return group_ids

    # ---------------- 查询日志 / 知识缺失 ----------------

    @staticmethod
    def _log_dict(row) -> dict:
        """日志行转字典（item_names JSON 还原为列表）。"""
        data = dict(row)
        try:
            data["item_names"] = json.loads(data.get("item_names") or "[]")
        except Exception:
            data["item_names"] = []
        return data

    def insert_query_log(self, task_id: str, user_id: int | None, query: str, rewritten_query: str,
                         item_names: list, answer: str, answer_type: int) -> int:
        """
        写入一条普通查询日志（知识缺失/后续看板的数据源头）。

        Args:
            task_id: 查询任务 id（便于对齐 SSE 过程）
            user_id: 提问人 id（未选身份时为 None）
            query: 原始问题
            rewritten_query: 改写后的问题
            item_names: 命中的商品/主体名列表（JSON 存储）
            answer: 最终答案全文
            answer_type: 回答类型（0 正常 / 1 资料缺失拒答 / 2 权限拒答 / 3 澄清追问）

        Returns:
            int: 日志 id（前端反馈按钮需要）
        """
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO query_logs(task_id, user_id, query, rewritten_query, item_names, answer, answer_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, user_id, query, rewritten_query or "",
                 json.dumps(list(item_names or []), ensure_ascii=False), answer, int(answer_type)),
            )
        log_id = int(cur.lastrowid)
        logger.info(f"查询日志已写入,id:{log_id},answer_type:{answer_type}")
        return log_id

    def get_query_log(self, log_id: int) -> dict | None:
        """按 id 取查询日志；不存在返回 None。"""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM query_logs WHERE id = ?", (int(log_id),)).fetchone()
        return self._log_dict(row) if row else None

    def set_feedback(self, log_id: int, feedback_status: int, user_id: int | None) -> None:
        """
        提交问答帮助度反馈（借鉴讲义模块十的边界规则）。

        防御性校验：
        1. 仅正常回答（answer_type=0）可反馈——拒答直接计入知识缺失，不需要用户再点无帮助
        2. 仅本人日志可反馈（日志无归属时放开，适配未选身份的演示场景）
        """
        if feedback_status not in (FEEDBACK_NONE, FEEDBACK_HELPFUL, FEEDBACK_UNHELPFUL):
            raise ValueError(f"反馈状态不合法:{feedback_status}")
        log = self.get_query_log(log_id)
        if log is None:
            raise QueryLogNotFoundError(f"日志不存在:id={log_id}")
        if log["answer_type"] != ANSWER_TYPE_NORMAL:
            raise FeedbackNotAllowedError("仅正常回答支持反馈,拒答记录已直接计入知识缺失!")
        if log["user_id"] is not None and user_id is not None and int(log["user_id"]) != int(user_id):
            raise FeedbackForbiddenError("只能反馈本人的问答记录!")
        with self._connect() as conn:
            conn.execute("UPDATE query_logs SET feedback_status = ? WHERE id = ?",
                         (int(feedback_status), int(log_id)))
        logger.info(f"反馈已提交,log:{log_id},feedback:{feedback_status}")

    def list_gaps(self, page: int, page_size: int, gap_filter: str) -> dict:
        """
        知识缺失运营视图：资料缺失拒答（answer_type=1）∪ 无帮助反馈（feedback_status=2）的查询日志。

        Args:
            page / page_size: 分页参数
            gap_filter: all 全部 / pending 待处理 / handled 已处理
        """
        base = "FROM query_logs WHERE (answer_type = 1 OR feedback_status = 2)"
        if gap_filter == "pending":
            base += " AND gap_handled = 0"
        elif gap_filter == "handled":
            base += " AND gap_handled = 1"
        with self._connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) {base}").fetchone()[0])
            rows = conn.execute(
                f"SELECT * {base} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (int(page_size), (int(page) - 1) * int(page_size)),
            ).fetchall()
        return {"total": total, "page": int(page), "page_size": int(page_size),
                "items": [self._log_dict(row) for row in rows]}

    def set_gap_handled(self, log_id: int, handled: bool) -> None:
        """标记/撤销一条缺失记录的处理状态（补资料或已答复后由运营操作）。"""
        if self.get_query_log(log_id) is None:
            raise QueryLogNotFoundError(f"日志不存在:id={log_id}")
        with self._connect() as conn:
            conn.execute("UPDATE query_logs SET gap_handled = ? WHERE id = ?",
                         (int(bool(handled)), int(log_id)))
        logger.info(f"缺失记录处理状态更新,log:{log_id},handled:{handled}")

    def gap_summary(self) -> dict:
        """缺失概览计数：拒答数 / 无帮助数 / 待处理数（缺失 Tab 顶部徽章用）。"""
        with self._connect() as conn:
            refusal = int(conn.execute(
                "SELECT COUNT(*) FROM query_logs WHERE answer_type = 1").fetchone()[0])
            unhelpful = int(conn.execute(
                "SELECT COUNT(*) FROM query_logs WHERE feedback_status = 2").fetchone()[0])
            pending = int(conn.execute(
                "SELECT COUNT(*) FROM query_logs WHERE gap_handled = 0 AND (answer_type = 1 OR feedback_status = 2)"
            ).fetchone()[0])
        return {"refusal_count": refusal, "unhelpful_count": unhelpful, "pending_count": pending}


if __name__ == '__main__':
    # 离线自测：临时库验证三表 CRUD、授权去重与三种命中、默认组保护
    import tempfile

    tmp_db = Path(tempfile.mkdtemp()) / "selftest.db"
    gw = SqliteGateway(db_path=tmp_db)

    # 默认组存在且受保护
    default_id = gw.get_default_group_id()
    assert default_id == 1, default_id
    try:
        gw.delete_group(default_id)
        raise AssertionError("默认组应不可删除")
    except ValueError:
        pass

    # 组 CRUD
    g1 = gw.create_group("研发部知识库", "研发专属")
    g2 = gw.create_group("销售部知识库")
    try:
        gw.create_group("研发部知识库")
        raise AssertionError("重名应抛错")
    except ValueError:
        pass
    assert len(gw.list_groups()) == 3
    gw.update_group(g2["group_id"], description="销售专属资料")

    # 人员 + 三种授权命中
    u1 = gw.create_user("张三", department="研发部", position="工程师")
    u2 = gw.create_user("李四", department="销售部", position="销售")
    gid = g1["group_id"]
    gw.add_grant(gid, "department", "研发部")
    try:
        gw.add_grant(gid, "department", "研发部")
        raise AssertionError("重复授权应抛错")
    except ValueError:
        pass
    gw.add_grant(gid, "user", "李四")
    gw.add_grant(gid, "position", "实习生")
    assert gw.resolve_user_group_ids(u1["user_id"]) == sorted({default_id, gid})
    assert gw.resolve_user_group_ids(u2["user_id"]) == sorted({default_id, gid})
    grants = gw.list_grants(gid)
    assert len(grants) == 3 and {g["subject_type"] for g in grants} == {"department", "user", "position"}

    # 改名联动"具体人"授权；撤销授权；删人清授权
    gw.update_user(u2["user_id"], name="李四明")
    assert any(g["subject_value"] == "李四明" for g in gw.list_grants(gid))
    gw.remove_grant(gid, "position", "实习生")
    assert len(gw.list_grants(gid)) == 2
    gw.delete_user(u2["user_id"])
    assert [g for g in gw.list_grants(gid) if g["subject_value"] == "李四明"] == []

    # 组删除级联授权；不存在的组/人报错
    gw.delete_group(gid)
    assert gw.list_grants(gid) == []
    for fn in (lambda: gw.get_group(999), lambda: gw.get_user(999)):
        assert fn() is None
    try:
        gw.resolve_user_group_ids(999)
        raise AssertionError("应抛人员不存在")
    except ValueError:
        pass

    # 查询日志 / 知识缺失闭环
    lid_ok = gw.insert_query_log("t-log-1", u1["user_id"], "怎么保修", "烫金机怎么保修",
                                 ["烫金机"], "正常答案", ANSWER_TYPE_NORMAL)
    lid_refusal = gw.insert_query_log("t-log-2", None, "量子芯片参数", "量子芯片参数",
                                      [], "未在知识库中检索到相关内容", ANSWER_TYPE_NO_KNOWLEDGE)

    # 反馈规则：正常回答可反馈；拒答不可反馈；只能反馈本人的
    gw.set_feedback(lid_ok, FEEDBACK_HELPFUL, u1["user_id"])
    try:
        gw.set_feedback(lid_refusal, FEEDBACK_UNHELPFUL, None)
        raise AssertionError("拒答应不可反馈")
    except FeedbackNotAllowedError:
        pass
    try:
        gw.set_feedback(lid_ok, FEEDBACK_UNHELPFUL, 99999)
        raise AssertionError("非本人应不可反馈")
    except FeedbackForbiddenError:
        pass

    # 缺失视图：拒答先计入；无帮助更新后也计入；处理状态流转
    assert [g["id"] for g in gw.list_gaps(1, 20, "all")["items"]] == [lid_refusal]
    gw.set_feedback(lid_ok, FEEDBACK_UNHELPFUL, u1["user_id"])
    assert {g["id"] for g in gw.list_gaps(1, 20, "all")["items"]} == {lid_ok, lid_refusal}
    assert gw.list_gaps(1, 20, "pending")["total"] == 2
    gw.set_gap_handled(lid_refusal, True)
    assert gw.list_gaps(1, 20, "pending")["total"] == 1
    assert gw.list_gaps(1, 20, "handled")["total"] == 1
    summary = gw.gap_summary()
    assert summary == {"refusal_count": 1, "unhelpful_count": 1, "pending_count": 1}, summary

    # 不存在的日志 → 404 语义
    try:
        gw.set_gap_handled(99999, True)
        raise AssertionError("应抛日志不存在")
    except QueryLogNotFoundError:
        pass

    print("SQLITE_GATEWAY_SELFTEST_ALL_PASSED")


# 模块级单例，业务代码统一入口
sqlite_gateway = SqliteGateway()
