import os
import time
from typing import Union, List, Dict, Any

# New SDK import
try:
    from google import genai
    from google.genai import types
except ImportError:
    # Fallback for environment where google-genai isn't installed yet
    # though it's recommended to install it.
    import google.generativeai as old_genai

from src.utils.extract_json_reliable import extract_json


def _to_prompt(message: Union[str, List[Dict[str, str]]]) -> str:
    """Convert chat-style messages to a single prompt string."""
    if isinstance(message, str):
        return message
    parts: List[str] = []
    for m in message:
        role = (m.get("role") or "user").strip().lower()
        content = m.get("content") or ""
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def complete_text_gemini(message: Union[str, list], 
                         model: str = "gemini-2.0-pro",
                         json_object: bool = False,
                         max_new_tokens: int = 4096, 
                         temperature: float = 1.0, 
                         max_retry: int = 5,
                         sleep_time: int = 2,
                         request_timeout: int = 60,
                         **kwargs: Any) -> str:
    """
    Call the Gemini API (using the modern google-genai SDK) to complete a prompt.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini.")
    
    # Try using the new SDK first
    try:
        client = genai.Client(api_key=api_key)
        prompt = _to_prompt(message)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            **kwargs
        )

        for cnt in range(max_retry):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )
                
                if not resp.text:
                    print('Warning! No text returned! Could be due to safety filter.\n', resp)
                    return ' '
                    
                if json_object:
                    return extract_json(resp.text)
                else:
                    return resp.text
            except Exception as e:
                print(f"Attempt {cnt} failed: {e}. Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)

    except (NameError, ImportError):
        # Fallback to old SDK if new one isn't available
        old_genai.configure(api_key=api_key)
        prompt = _to_prompt(message)
        
        generation_config = old_genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
        )

        for cnt in range(max_retry):
            try:
                model_obj = old_genai.GenerativeModel(model_name=model, generation_config=generation_config)
                # Note: using request_options for compatibility with newer versions of the old SDK
                resp = model_obj.generate_content(prompt, request_options={"timeout": request_timeout}, **kwargs)
                
                if not resp.candidates or not resp.candidates[0].content.parts:
                    print('Warning! No candidates returned! Could be due to safety filter.\n', resp)
                    return ' '

                if json_object:
                    return extract_json(resp.text)
                else:
                    return resp.text
            except Exception as e:
                print(f"Attempt {cnt} failed: {e}. Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)
    
    raise Exception("Failed to complete Gemini text after maximum retries")
