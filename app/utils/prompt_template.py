from llama_index.core.prompts import RichPromptTemplate


def prompt_template(context: str, path: str) -> str:
    """
    Create a prompt template for the ideal model.
    """
    template_str = open(path, "r", encoding="utf-8").read()

    if not template_str:
        print("Prompt template is empty")
        return ""

    try:
        qa_template = RichPromptTemplate(template_str)
        prompt = qa_template.format(context_str=context)
    except Exception as e:
        print(f"Error adding context: {e}. \n Using default prompt.")
        prompt = template_str

    return prompt