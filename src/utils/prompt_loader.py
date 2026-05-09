"""Prompt loading utilities"""


def load_prompt(file_path: str) -> str:
    """
    Load prompt template from file

    Args:
        file_path: Path to prompt file

    Returns:
        Prompt content as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


__all__ = ['load_prompt']
