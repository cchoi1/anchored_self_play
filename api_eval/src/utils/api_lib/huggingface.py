import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List
from transformers import pipeline
from peft import PeftModelForCausalLM, PeftModel
from src.utils.extract_json_reliable import extract_json

# Dictionary to cache loaded Hugging Face models and tokenizers
loaded_hf_models = {}

def generation_pipeline_hf(message: Union[str, List[dict], List[List[dict]]],
                           model: torch.nn.Module, 
                           tokenizer: AutoTokenizer,
                           stop_sequence=[], 
                           device: str="auto", 
                           json_object=False,
                           **kwargs):

        torch.cuda.empty_cache()
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.padding_side = 'left'   
        with torch.no_grad():
            if isinstance(model, PeftModel):
                base_model = model.base_model.model  # Extract the base model
            else:
                base_model = model  # Keep original model
            generator = pipeline("text-generation", 
                                 model=base_model,
                                 tokenizer=tokenizer,
                                 model_kwargs={"torch_dtype": "auto"},
                                 device_map=device)
            
            terminators = [generator.tokenizer.eos_token_id]
            if len(stop_sequence) > 0:
                terminators = terminators + [generator.tokenizer.convert_tokens_to_ids(s) for s in stop_sequence]
                
            outputs = generator(
                message,
                eos_token_id=terminators,
                **kwargs
            )
        if isinstance(message, str):
            result = outputs[0]["generated_text"][len(message):]
            if result is None or not isinstance(result, str):
                result = ' '
        elif isinstance(message, list):
            if len(message) == 1:
                result = outputs[0][0]['generated_text'][-1]['content']
                if result is None or not isinstance(result, str):
                    result = ' '
            else:
                result = [outputs[i][0]['generated_text'][-1]['content'] for i in range(len(message))]
                # Filter out None values from list instead of replacing whole list
                result = [r if r is not None and isinstance(r, str) else ' ' for r in result]

        del outputs
        torch.cuda.empty_cache()

        if json_object:
            return extract_json(result)
        else:
            return result


def generate_text_hf(message: Union[str, List[dict]], 
                     model: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                     max_new_tokens: int = 2048, 
                     temperature: float = 0.5, 
                     json_object: bool = False,
                     stop_sequence: list = [], 
                     cached_model=False, 
                     **kwargs) -> str:
    """
    Generate text completion using a specified Hugging Face model.

    Args:
        message (str): The input text message for completion.
        model (str): The Hugging Face model to use.
        max_new_tokens (int): The maximum number of tokens to generate. Default is 2048.
        temperature (float): Sampling temperature for generation. Default is 0.5.
        json_object (bool): Whether to format the message for JSON output. Default is False.
        stop_sequence (list): List of stop sequences to halt the generation.
        **kwargs: Additional keyword arguments for the `generate` function.

    Returns:
        str: The generated text completion.
    """
    
    if json_object and not 'json' in message:
        message = "You are a helpful assistant designed to output in JSON format." + message
    
    model_name = model.split("/", 1)[1]
    
    # Load the model and tokenizer if not already loaded
    if model_name in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model_name]
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype='auto')
        tokenizer = AutoTokenizer.from_pretrained(model)

    if cached_model:
        loaded_hf_models[model_name] = (hf_model, tokenizer)

    text = generation_pipeline_hf(message, 
                                  hf_model, tokenizer, 
                                  temperature=temperature, 
                                  max_new_tokens=max_new_tokens, 
                                  **kwargs)

    if json_object:
        return extract_json(text)
    else:
        return text

