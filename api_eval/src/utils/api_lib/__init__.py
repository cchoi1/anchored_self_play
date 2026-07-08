import os
import os.path as osp
import warnings

# Get the current directory
cur_dir = osp.dirname(os.path.realpath(__file__))

# Define the outer directory path
outer_dir = osp.join(cur_dir, "..", "..", "..")

# Setup Anthropic API key
anthropic_client = None
try:
    import anthropic

    # Try to read from config file first
    api_key_path = osp.join(outer_dir, "config", "claude_api_key.txt")
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as key_file:
            api_key = key_file.read().strip()
        anthropic_client = anthropic.Anthropic(api_key=api_key)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
except Exception as e:
    pass

# Setup OpenAI API key
try:
    import openai

    # Try to read from config file first
    api_key_path = osp.join(outer_dir, "config", "openai_api_key.txt")
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as key_file:
            openai_key = key_file.read().strip()
        os.environ["OPENAI_API_KEY"] = openai_key
except Exception as e:
    pass

# Setup Gemini (Google) API key
try:
    # Try to read from config file first
    api_key_path = osp.join(outer_dir, "config", "gemini_api_key.txt")
    if os.path.exists(api_key_path):
        with open(api_key_path, "r") as key_file:
            gemini_key = key_file.read().strip()
        os.environ.setdefault("GOOGLE_API_KEY", gemini_key)
    elif os.environ.get("GEMINI_API_KEY"):
        os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
except Exception:
    pass