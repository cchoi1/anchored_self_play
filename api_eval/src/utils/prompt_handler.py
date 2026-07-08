from typing import Union, get_origin, get_args


class PromptHandler:
    """
    A class to represent and validate a template with expected input and output structures.

    Attributes:
        template (str): The text of the template.
        input_keys (list): A list of keys expected in the input.
        output_format (type or dict): The expected structure of the output, which can be a type (e.g., str) or a JSON-like dictionary.
        strict_input (bool): A flag indicating whether input validation should be strict (True) or lenient (False).
    """

    def __init__(self, template, input_keys, output_format, name, system_message=None, tokenizer=None, strict_input=False):
        """
        Initializes the PromptHandler object with a template text, input keys, output structure, and strict input flag.

        Args:
            template (str): The text of the template.
            input_keys (list): A list of keys expected in the input.
            output_format (type or dict): The expected structure of the output.
            strict_input (bool, optional): Flag for strict input validation. Defaults to False.
        """
        self.template = template
        self.input_keys = input_keys
        self.output_format = output_format
        self.strict_input = strict_input
        self.name = name
        self.system_message = system_message
        self.tokenizer = tokenizer
    
    def check_format(self, response):
        """
        Validates the format of the response against the expected output structure.

        Args:
            response (any): The response to validate.

        Returns:
            bool: True if the response matches the expected structure, False otherwise.
        """
        return self._check_structure(self.output_format, response)
    
    def _check_structure(self, expected_structure, actual_response):
        """
        Recursively checks the structure of the actual response against the expected structure.

        Args:
            expected_structure (type or dict): The expected structure.
            actual_response (any): The actual response to check.

        Returns:
            bool: True if the response matches the expected structure, False otherwise.
        """
        if get_origin(expected_structure) is Union:
            return any(isinstance(actual_response, t) for t in get_args(expected_structure))
        
        if isinstance(expected_structure, type):
            return isinstance(actual_response, expected_structure)
        
        if isinstance(expected_structure, dict):
            if not isinstance(actual_response, dict):
                print("Expected a dictionary but got:", actual_response)
                return False
            
            for key, expected_value in expected_structure.items():
            
                if key not in actual_response:
                    print(f"Key '{key}' not found in response.")
                    return False
                
                actual_value = actual_response[key]
                
                if not self._check_structure(expected_value, actual_value):
                    return False

            return True
        
        print("Invalid expected structure:", expected_structure)
        return False
    
    def __call__(self, **input_dict):
        """
        Validates and processes the input dictionary against the expected input keys,
        and formats the template with the given input.

        Args:
            input_dict (dict): The input dictionary to validate and process.

        Returns:
            str: The formatted template with the given input.

        Raises:
            ValueError: If input keys do not match the expected keys or if required keys are missing.
        """
        if self.strict_input:
            if set(input_dict.keys()) != set(self.input_keys):
                raise ValueError(f"Input keys do not match the expected keys: {self.input_keys}")
            subset_dict = input_dict
        else:
            subset_dict = {key: input_dict[key] for key in self.input_keys if key in input_dict}
            if len(subset_dict) != len(self.input_keys):
                missing_keys = set(self.input_keys) - set(subset_dict.keys())
                raise ValueError(f"Missing keys in input: {missing_keys}")
        
        user_prompt = self.template.format(**subset_dict)
        if self.system_message and self.tokenizer:
            chat = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_prompt}
            ]
            chat_template = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            return chat_template
        return user_prompt

