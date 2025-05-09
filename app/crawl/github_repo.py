import os
from github import Github
from typing import List, Dict
from github.Repository import Repository
from app.config.app_config import FILE_TYPE_MAPPING, GITHUB_API_KEY


def get_files_on_repo(repo: Repository) -> List[Dict]:
    files = []
    contents = repo.get_contents("")

    while contents:
        content = contents.pop(0)
        if content.type == "dir":
            contents.extend(repo.get_contents(content.path))
        else:
            path = content.path
            extension = os.path.splitext(path)[1].lower()
            file_type = FILE_TYPE_MAPPING.get(extension, 'other')

            try:
                file_content = content.decoded_content.decode("utf-8")
                files.append({"path": path, "content": file_content, "type": file_type})
            except Exception as e:
                print(f"Cannot decode {path}: {e}")
                files.append({"path": path, "content": None, "type": file_type})

    return files


if __name__ == "__main__":
    if GITHUB_API_KEY is None:
        raise ValueError("GITHUB_API_KEY environment variable not set")
    github = Github(GITHUB_API_KEY)

    # repo = github.get_repo("biabeogo147/Magic-wand")
    # files = get_files_on_repo(repo)
    # for file in files:
    #     print(f"Path: {file['path']}")
    #     print(f"Content: {file['content'][:15] if file['content'] is not None else None}...")
    #     print(f"Type: {file['type']}")
    #     print("---")

    # repos = github.search_repositories("stars:>1000", sort="stars", order="desc")
    # for repo in repos[:10]:
    #     print(f"Repository: {repo.full_name}")
