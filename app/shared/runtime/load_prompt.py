from app.shared.runtime.logger import PROJECT_ROOT, logger


def load_prompt(name: str, **kwargs) -> str:
    prompt_path = PROJECT_ROOT / 'app' / 'resources' / 'prompts' / f"{name}.prompt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词不存在：{prompt_path.absolute()}")

    raw_prompt = prompt_path.read_text('utf-8')

    if kwargs:
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"提示词渲染成功，替换变量：{list(kwargs.keys())}")
        return rendered_prompt
    return raw_prompt


if __name__ == '__main__':
    root_folder = "h1370使用说明书"
    image_context = ("这是图片的上文内容", "这是图片的下文内容")

    final_prompt = load_prompt(
        name='image_summary',
        root_folder=root_folder,
        image_context=image_context
    )
    print("✅ 渲染后的最终提示词：")
    print(final_prompt)
