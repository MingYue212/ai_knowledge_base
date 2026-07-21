from app.shared.model import generate_embeddings
from app.shared.model.lm_utils import get_llm_client


class LLMProvider:
    def vision_model(self, vision_model_name: str):
        return get_llm_client(model=vision_model_name)

    def llm_model(self, llm_model_name, json_code):
        return get_llm_client(llm_model_name, json_code)


    def generate_embeddings(texts: list[str]) -> dict[str, list]:
        return generate_embeddings(texts)

llm_provider = LLMProvider()