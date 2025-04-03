import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_claim_extraction_prompt(content):
    prompt = """
Instructions:
1. You are given a sentence or a short paragraph. Your task is to break the sentence or short paragraph down into a list of atomic facts.
2. An atomic fact is a sentence containing a singular piece of information.
3. Each atomic fact in the outputted list should check a different piece of information.
4. Use the previous examples to learn how to do this.
5. You should only output the atomic facts as a list, with each item starting with “- ”. Do not include
other formatting.
6. Your task is to do this for the last sentence that is given.

Please breakdown the following into independent facts:
[EXAMPLE #1]
The image showcases a well-equipped kitchen featuring a white countertop, green cabinets, and stainless steel appliances.
- [FACT-1] The image shows a kitchen.
- [FACT-2] The kitchen is well-equipped.
- [FACT-3] The countertop is white.
- [FACT-4] The cabinets are green.
- [FACT-5] The appliances are stainless steel.

Please breakdown the following into independent facts:
[EXAMPLE #2]
The image showcases a large gathering of birds, including pigeons and seagulls, on a concrete sidewalk. They appear to be looking for food and are of various colors, such as black and brown. The birds cover most of the sidewalk, extending from the foreground to the background of the scene. A food vendor cart is visible in the background, indicating that the birds may be searching for scraps. The scene is bordered by a stone wall and some shrubbery on the right side, adding a touch of greenery to the concrete environment.
- [FACT-1] The image shows a large gathering of birds.
- [FACT-2] The birds include pigeons.
- [FACT-3] The birds include seagulls.
- [FACT-4] The birds are on a concrete sidewalk.
- [FACT-5] The birds appear to be looking for food.
- [FACT-6] The birds are of various colors.
- [FACT-7] Some birds are black.
- [FACT-8] Some birds are brown.
- [FACT-9] The birds cover most of the sidewalk.
- [FACT-10] The birds extend from the foreground to the background.
- [FACT-11] A food vendor cart is visible in the background.
- [FACT-12] The food vendor cart indicates the birds may be searching for scraps.
- [FACT-13] The scene is bordered by a stone wall.
- [FACT-14] The scene is bordered by shrubbery on the right side.
- [FACT-15] The shrubbery adds greenery to the concrete environment.

Please breakdown the following into independent facts:
[EXAMPLE #3]
The image shows a man kneeling in front of a toilet inside a bathroom, with his head close to the toilet bowl as if he is throwing up. The overall scene suggests that he may be feeling unwell or experiencing nausea. The bathroom also contains a sink situated towards the right side of the room. There are several bottles of various sizes scattered throughout the bathroom, some near the sink, and others on the floor around the individual and the toilet. A towel is seen hanging on the bathroom wall, indicating that someone may have tried to clean up the mess.
- [FACT-1] The image shows a man kneeling in front of a toilet.
- [FACT-2] The man is inside a bathroom.
- [FACT-3] The man's head is close to the toilet bowl.
- [FACT-4] The man appears to be throwing up.
- [FACT-5] The man may be feeling unwell.
- [FACT-6] The man may be experiencing nausea.
- [FACT-7] The bathroom contains a sink.
- [FACT-8] The sink is situated towards the right side of the room.
- [FACT-9] There are several bottles of various sizes scattered throughout the bathroom.
- [FACT-10] Some bottles are near the sink.
- [FACT-11] Some bottles are on the floor around the man.
- [FACT-12] Some bottles are on the floor around the toilet.
- [FACT-13] A towel is hanging on the bathroom wall.
- [FACT-14] Someone may have tried to clean up the mess.

Please breakdown the following into independent facts:
[EXAMPLE #4]
The image features a woman wearing an apron standing in a kitchen, carefully using a rolling pin to flatten dough in a mixing bowl.
- [FACT-1] The image features a woman.
- [FACT-2] The woman is wearing an apron.
- [FACT-3] The woman is standing in a kitchen.
- [FACT-4] The woman is using a rolling pin.
- [FACT-5] The woman is carefully using the rolling pin.
- [FACT-6] The woman is flattening dough.
- [FACT-7] The dough is in a mixing bowl.

Please breakdown the following into independent facts:
[EXAMPLE #5]
The image shows a group of cows gathered together on a farm, standing next to each other on a dirt field that lacks grass.
- [FACT-1] The image shows a group of cows.
- [FACT-2] The cows are gathered together.
- [FACT-3] The cows are on a farm.
- [FACT-4] The cows are standing next to each other.
- [FACT-5] The cows are on a dirt field.
- [FACT-6] The dirt field lacks grass.

Please breakdown the following into independent facts:

"""
    return prompt + f"{content}\n"

class Qwen_2_5_LLM_14B_Instruct:
    def __init__(self, ):
        
        model_name="Qwen/Qwen2.5-14B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="balanced_low_0",
            cache_dir="../hf_models/",
            attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../hf_models/")

    # this function is to get generic text response from qwen
    def get_response(self, query):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = model_inputs.to("cuda")
        # Generate the text response
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    # this function is to extract a 'list of claims' from a given text --> developed for HAL detection
    def extract_claims(self, content):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": get_claim_extraction_prompt(content)}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = model_inputs.to("cuda")
        # Generate the text response
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Extract claims
        # Updated regex pattern to capture the contents of each fact
        _pattern = r"- \[FACT-\d+\] (.+)"
        _matches = re.findall(_pattern, response)
        fact_list = [match.strip() for match in _matches]
        return fact_list



"""
==========================================================================================================================================================================
==========================================================================================================================================================================
"""


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen_2_5_VL_7B_Instruct:
    def __init__(self, ):
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map="balanced_low_0",
            cache_dir="../hf_models/",
            attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="../hf_models/", use_fast=True)

    # this function is to get generic text response from qwen
    def get_response(self, img_path, query):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0]