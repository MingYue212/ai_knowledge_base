from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_running_task, add_done_task


@node_log("node_entry")
def node_entry(state:ImportGraphState) ->ImportGraphState:
    add_running_task(state["task_id"],"node_entry")
    state = resolve_input_file(state)
    add_done_task(state["task_id"],"node_entry")
    return state