import json

from app.process.import_.agent.main_graph import import_app
from app.process.import_.agent.state import create_default_state

input_state = create_default_state(task_id="task_001",local_file_path="./xxx.md",is_pdf_read_enabled=True)
state = import_app.invoke(input_state)

print(f"最终结果的state:\n {json.dumps(state,indent=4,ensure_ascii=False)}")

import_app.get_graph().print_ascii()