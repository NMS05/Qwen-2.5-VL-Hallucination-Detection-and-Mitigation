import json
import re
from tqdm import tqdm

from qwen_wrapper import Qwen_2_5_LLM_14B_Instruct, Qwen_2_5_VL_7B_Instruct

# ---------------------------
# ---------------------------
        # ALL PROMPTS
# ---------------------------
# ---------------------------

def get_annotation_prompt(sentences):
    prompt = """You are a visual expert tasked with verifying the accuracy of statements based on a provided image. Use the visual evidence and your knowledge to evaluate each statement according to the categories below.

### Evaluation Categories:
- **non-hallucination**: The statement is factually correct and fully supported by the image.
- **hallucination**: The statement contains factual errors or contradicts the visual evidence.
- **subjective**: The statement expresses personal opinions, interpretations, or subjective descriptions that cannot be objectively verified using the image.

### Evaluation Guidelines:
1. **Object**: Confirm that any mentioned objects exist in the image and that their quantity matches the statement.
2. **Attributes**: Verify that attributes such as color, size, or position of objects are accurately described.
3. **Relationship**: Assess if the relationships between objects depicted in the image align with the statement.
4. **Scene Text**: Determine if any text within the image supports or contradicts the statement.
5. **Fact**: Evaluate the overall factual correctness based on visual evidence and relevant knowledge.

### Input Format:
- **Input Image**: A visual image provided as context.
- **Statements**: A list of claims to be evaluated. Each statement is prefixed with `[STATEMENT X]`, where `X` is its sequence number.

### Output Format:
- For each statement, provide your evaluation result and the rationale behind your evaluation within the `[EVALUATION X]` and `[REASON X]` tags

Respond strictly using the following template. Do not include any additional text or formatting outside the specified tags.

### Example:

**Input**
input_image = <an image of two cats sleeping on a pink couch>
[STATEMENT 1]: The image features two cats lying on a red couch. 
[EVALUATION 1]: 
[REASON 1]: 

[STATEMENT 2]: Both cats are sleeping peacefully. 
[EVALUATION 2]: 
[REASON 2]: 

**Expected Output**
[STATEMENT 1]: The image features two cats lying on a red couch. 
[EVALUATION 1]: hallucination 
[REASON 1]: The couch is pink, not red.

[STATEMENT 2]: Both cats are sleeping peacefully. 
[EVALUATION 2]: non-hallucination 
[REASON 2]: The cats clearly appear to be resting comfortably.

Now, carefully analyze the image and evaluate the following statements.

"""
    for i, sentence in enumerate(sentences):
        prompt += f"[STATEMENT {i+1}]: {sentence}\n"
        prompt += f"[EVALUATION {i+1}]: \n"
        prompt += f"[REASON {i+1}]: \n\n"
    return prompt

def get_additional_verification_prompt(annotations):
    prompt = """You are a visual expert tasked with verifying the accuracy of annotations provided for a set of image-related statements. The annotations follow the exact format:

[STATEMENT X]: <a statement text related to the given image image>
[EVALUATION X]: <one of the following categories: non-hallucination, hallucination, subjective>
[REASON X]: <a clear explanation behind the evaluation>

These annotations were done based on the following guidelines:

        ### Evaluation Categories:
        - **non-hallucination**: The statement is factually correct and fully supported by the image.
        - **hallucination**: The statement contains factual errors or contradicts the visual evidence.
        - **subjective**: The statement expresses personal opinions, interpretations, or subjective descriptions that cannot be objectively verified using the image.

        ### Evaluation Guidelines:
        1. **Object**: Confirm that any mentioned objects exist in the image and that their quantity matches the statement.
        2. **Attributes**: Verify that attributes such as color, size, or position of objects are accurately described.
        3. **Relationship**: Assess if the relationships between objects depicted in the image align with the statement.
        4. **Scene Text**: Determine if any text within the image supports or contradicts the statement.
        5. **Fact**: Evaluate the overall factual correctness based on visual evidence and relevant knowledge.

Your task is to carefully review each triplet (statement, evaluation, reason) in the input below. For each triplet:
- If the evaluation and reason are correct based on the statement, simply output the exact same triplet.
- If the evaluation and/or reason are incorrect, correct them and output the corrected triplet while maintaining the exact format.

Please ensure your verification considers the logical consistency between the statement, evaluation, and reason. Maintain clarity and strict adherence to the format in your output.

Now, verify the following annotations:

"""
    return prompt + annotations

def get_error_rectification_prompt(initial_description, annotations):
    prompt = f"""You are provided with an initial description of the given image. This description may contain inaccuracies, hallucinations, or subjective details. 

Along with it, you are provided with annotations for each claim extracted from the description. Each annotation includes:

- **Statement**: An extracted claim from the initial description.
- **Evaluation**: The assessment of the claim (i.e., non-hallucination, hallucination, subjective).
- **Reason**: A clear explanation supporting the evaluation.

The annotations are provided in the following format:

    [STATEMENT 1]: 
    [EVALUATION 1]:  
    [REASON 1]: 
    ..
    ..
    [STATEMENT N]: 
    [EVALUATION N]: 
    [REASON N]: 

You are a visual expert. Your task is to carefully review the initial description alongside these annotations. Then, generate a revised and accurate description that:
- Seamlessly incorporates the corrections indicated by the annotations.
- Maintains the original structure and coherent flow.
- Rectifies all identified errors or hallucinations.

Provide only the corrected response as your final output without any additional commentary or formatting.

### Examples:

#### Example 1:
**Initial Response:**
The image shows two young boys walking through an apple orchard, carrying a basket for collecting apples. Both boys are wearing baseball caps aleads through the orchard.

**Annotations:**
[STATEMENT 1]: The image shows two young boys.  
[EVALUATION 1]: non-hallucination  
[REASON 1]: There are two children visible in the image, and they appear to be young boys based on their size and clothing.

[STATEMENT 2]: The boys are walking through an apple orchard.  
[EVALUATION 2]: non-hallucination  
[REASON 2]: The background consists of rows of trees with apples scattered on the ground, which is typical of an apple orchard.

[STATEMENT 3]: The boys are carrying a basket.  
[EVALUATION 3]: hallucination  
[REASON 3]: There are no visible baskets in the image; the boys do not appear to be carrying anything.

[STATEMENT 4]: The basket is for collecting apples.  
[EVALUATION 4]: hallucination  
[REASON 4]: As there is no basket visible, this statement cannot be verified.

[STATEMENT 5]: Both boys are wearing baseball caps.  
[EVALUATION 5]: non-hallucination  
[REASON 5]: Both boys are wearing hats, and one appears to be a baseball cap.

[STATEMENT 6]: The boys are on a path leading through the orchard.  
[EVALUATION 6]: non-hallucination  
[REASON 6]: The boys are walking on a grassy path that runs through the orchard.

**Final Corrected Response:**
The image shows two young boys walking through an apple orchard. Both boys are wearing baseball caps as they stroll along a grassy path.

#### Example 2:
**Initial Response:**
The main colors of the jet in the image are blue, yellow, and black.

**Annotations:**
[STATEMENT 1]: The main color of the jet is blue.  
[EVALUATION 1]: hallucination  
[REASON 1]: The main color of the jet is white with green and red accents, not blue.

[STATEMENT 2]: The main color of the jet is yellow.  
[EVALUATION 2]: hallucination  
[REASON 2]: The main color of the jet is white with green and red accents, not yellow.

[STATEMENT 3]: The main color of the jet is black.  
[EVALUATION 3]: hallucination  
[REASON 3]: The main color of the jet is white with green and red accents, not black.

**Final Corrected Response:**
The main colours of the jet are green and red.


Now, carefully review the initial description alongside these annotations, then generate a revised and accurate description

---
{initial_description}
---

The annotations are as follows:

---
{annotations}
---
"""
    return prompt


# ---------------------------
# ---------------------------
        # Qwen HAL Detector
# ---------------------------
# ---------------------------

def main():
    
    # Initialize the Qwen models
    qwen_vlm = Qwen_2_5_VL_7B_Instruct()
    qwen_llm = Qwen_2_5_LLM_14B_Instruct()
    print(f"\nModels loaded ...\n")

    # Load the JSON file
    json_path = "..filtered_povid.json"
    with open(json_path, "r") as f:
        val_data = json.load(f)

    # List to store all results
    all_results = []

    for entry in tqdm(val_data):
        
        # load image
        img_path = '... path to coco images ...' + entry['image']
        
        # initial (potentially hallucinated) response
        initial_response = entry['initial_response']

        # STEP 1: Claim Extraction
        statements = []
        for statement in qwen_llm.extract_claims(initial_response): statements.append(statement)
        
        # STEP 2: HAL annotations
        annotation_prompt = get_annotation_prompt(statements)
        initial_annotations = qwen_vlm.get_response(img_path, annotation_prompt)

        # STEP 3: double-check HAL annotations
        additional_verification_prompt = get_additional_verification_prompt(initial_annotations)
        final_annotations = qwen_vlm.get_response(img_path, additional_verification_prompt)

        # STEP 4: get refined response (error-corrected response)
        error_rectification_prompt = get_error_rectification_prompt(initial_response, final_annotations)
        refined_response = qwen_vlm.get_response(img_path, error_rectification_prompt)
        
        # extract claims(statements), labels(evaluation), reason -->> save them to a json file
        eval_pattern = r"(?i)evaluation(?:(?!hallucination|non-hallucination|subjective).)*?(hallucination|non-hallucination|subjective)"
        reason_pattern = r"(?i)reason(.+)"

        eval_matches = re.findall(eval_pattern, final_annotations)
        reason_matches = re.findall(reason_pattern, final_annotations)

        eval_dict = {i + 1: match.strip() for i, match in enumerate(eval_matches)}
        reason_dict = {i + 1: re.sub(r"^[^:]*:\s*(?:\*\* )?", "", match).strip() for i, match in enumerate(reason_matches)}

        # Append the evaluation and reason to each claim's information
        evaluated_claims = []

        for i, claim in enumerate(statements, start=1):            
            _claim = {
                "claim": claim,
                "evaluation": eval_dict.get(i, ""),
                "reason": reason_dict.get(i, "")
            }
            evaluated_claims.append(_claim)
        
        # Compile the final evaluation result
        final_result = {
            "image": entry['image'],
            "prompt": entry['prompt'],
            "initial_response": initial_response,
            "initial_annotations": initial_annotations,
            "final_annotations": final_annotations,
            "evaluated_claims": evaluated_claims,
            "refined_response": refined_response,
        }

        # Add the final result to the list of all results
        all_results.append(final_result)

    # Save all results to a single JSON file
    with open("Qwen_HAL_Annotations.json", "w") as outfile:
        json.dump(all_results, outfile, indent=4)
    print(f"\nSaved to Qwen_HAL_Annotations.json\n")

if __name__ == "__main__":
    main()