import os
from app.utils.generating_workflow import generate_scripts

if __name__ == "__main__":
    root_dir = "response"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    prompt = "Generate a simple Caro game using python."
    generate_scripts(prompt, root_dir)