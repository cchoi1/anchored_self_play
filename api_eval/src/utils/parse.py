import re
from typing import Dict


def parse_response(response: str) -> Dict[str, str]:
    """
    Parse the model response to extract code and explanation.
    
    Args:
        response (str): Raw model response
        
    Returns:
        Dict containing:
            - code (str): Extracted code (with comments removed)
            - explanation (str): Extracted explanation
            
    Raises:
        ValueError: If parsing fails
    """
    response = response.strip()
    
    # collapse repeated ```python markers
    while True:
        new_response = re.sub(r'```(?:python)?\s*\n```(?:python)?\s*\n', '```python\n', response)
        new_response = re.sub(r'```(?:python)?\s*```(?:python)?\s*\n', '```python\n', new_response)
        if new_response == response:
            break
        response = new_response
    
    response = re.sub(r'\n```(?:python)?\s*\n```', '\n```', response)
    response = re.sub(r'```python\s*$', '```', response)

    fenced_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    code_blocks = re.findall(fenced_pattern, response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r"```\s*\n(.*?)\n```", response, re.DOTALL)
    if not code_blocks:
        raise ValueError("No code block found in response.")
    
    code = code_blocks[-1]
    
    # Remove inline comments
    code_lines = []
    for line in code.split('\n'):
        # Remove comment portion if it exists, preserve string literals
        in_string = False
        string_char = None
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            # Handle string literals
            if char in ['"', "'"]:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                result.append(char)
            # Only consider # as comment if not in string
            elif char == '#' and not in_string:
                break
            else:
                result.append(char)
            i += 1
        
        cleaned_line = ''.join(result).rstrip()
        if cleaned_line:  # Only add non-empty lines
            code_lines.append(cleaned_line)
    
    code = '\n'.join(code_lines)
    
    # Get explanation
    matches = list(re.finditer(fenced_pattern, response, re.DOTALL))
    if not matches:
        matches = list(re.finditer(r"```\s*\n(.*?)\n```", response, re.DOTALL))
    last_code_block_match = matches[-1]
    remaining_text = response[last_code_block_match.end():]
    explanation_match = re.search(r"Explanation:\s*(.*)", remaining_text, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    
    return {'code': code, 'explanation': explanation}

