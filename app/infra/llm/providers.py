from app.shared.model.lm_utils import get_llm_client


class LLMProvider:
    def vision_model(self, vision_model_name: str):
        return get_llm_client(model=vision_model_name)

    def llm_model(self, llm_model_name, json_code):
        return get_llm_client(llm_model_name, json_code)

llm_provider = LLMProvider()