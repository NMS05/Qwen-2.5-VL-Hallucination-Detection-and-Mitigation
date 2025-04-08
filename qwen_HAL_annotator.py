import json
import re
from tqdm import tqdm
import random

from qwen_wrapper import Qwen_2_5_LLM_7B_Instruct, Qwen_2_5_VL_7B_Instruct
from utils import IMAGE_DIR, DATA_PATH, SAVE_PATH

# ---------------------------
# ---------------------------
        # ALL PROMPTS
# ---------------------------
# ---------------------------

def get_annotation_prompt(statement):
    prompt = """You are a visual expert tasked with verifying the accuracy of a statement based on the provided image. Use the visual evidence and your knowledge to evaluate each statement according to the categories below.

### Evaluation Categories:
- **non-hallucination**: The statement is correct and fully supported by the image.
- **hallucination**: The statement contains errors and contradicts the visual evidence.

### Evaluation Guidelines:
1. **Object**: Confirm that any mentioned objects exist in the image and that their quantity matches the statement.
2. **Attributes**: Verify that attributes such as color, size, or position of objects are accurately described.
3. **Relationship**: Assess if the relationships between objects depicted in the image align with the statement.
4. **Scene Text**: Determine if any text within the image supports or contradicts the statement.

### Input Format:
- **Input Image**: A visual image provided as context.
- **Statement**: A statement that needs to be evaluated. This statement is prefixed with the tag `[STATEMENT]`

### Output Format:
- Provide your evaluation result and the rationale behind your evaluation within the `[EVALUATION]` and `[REASON]` tags

Respond strictly using the following template. Do not include any additional text or formatting outside the specified tags.

### Example:

input_image = <an image of two cats sleeping on a pink couch>

**Input**
[STATEMENT]: The image features two cats lying on a red couch.
[EVALUATION]:
[REASON]:

**Expected Output**
[STATEMENT]: The image features two cats lying on a red couch.
[EVALUATION]: hallucination
[REASON]: The couch is pink, not red.

**Input**
[STATEMENT]: Both cats are sleeping peacefully.
[EVALUATION]:
[REASON]:

**Expected Output**
[STATEMENT]: Both cats are sleeping peacefully.
[EVALUATION]: non-hallucination
[REASON]: The cats clearly appear to be resting comfortably.


Now, carefully analyze the image and evaluate the following statement.

"""
    prompt += f"[STATEMENT]: {statement}\n"
    prompt += f"[EVALUATION]: \n"
    prompt += f"[REASON]: \n\n"
    return prompt

def get_error_rectification_prompt(initial_description, annotations):
    prompt = f"""You are provided with an initial description of the given image. This description may contain inaccuracies or hallucinations. 

Along with it, you are provided with annotations for each claim extracted from the description. Each annotation includes:

- **Statement**: An extracted claim from the initial description.
- **Evaluation**: The assessment of the claim i.e., non-hallucination (accurate) or hallucination (inaccurate).
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

### Example:

** Input **

Initial Response:
The image shows two young boys walking through an apple orchard, carrying a basket for collecting apples. Both boys are wearing baseball caps aleads through the orchard.

Annotations:
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

**Expected Output**

Final Corrected Response:
The image shows two young boys walking through an apple orchard. Both boys are wearing baseball caps as they stroll along a grassy path.


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
    qwen_llm = Qwen_2_5_LLM_7B_Instruct()
    print(f"\nModels loaded ...\n")

    # Load the JSON file
    with open(DATA_PATH, "r") as f:
        val_data = json.load(f)

    # List to store all results
    all_results = []

    # for entry in tqdm(random.sample(val_data,k=10)): # for troubleshooting
    for entry in tqdm(val_data):
        
        # load image
        img_path = IMAGE_DIR + entry['image']
        
        # initial (potentially hallucinated) response
        initial_response = entry['initial_response']

        # STEP 1: Claim Extraction
        statements = qwen_llm.extract_claims(initial_response)
        
        # STEP 2: HAL annotations (each claim MUST be evaluated independently)
        image_paths = []
        statements_to_be_evaluated = []
        for s in statements:
            image_paths.append(img_path)
            statements_to_be_evaluated.append(get_annotation_prompt(s))
        qwen_batch_annotations = qwen_vlm.get_batch_response(image_paths, statements_to_be_evaluated)

        # extract the annotations for each claim and combine them
        evaluations = []
        reasons = []
        eval_pattern = r"(?i)evaluation(?:(?!hallucination|non-hallucination).)*?(hallucination|non-hallucination)"
        reason_pattern = r"(?i)reason(.+)"

        for response in qwen_batch_annotations:
            # Extract evaluation text
            eval_match = re.search(eval_pattern, response, flags=re.IGNORECASE)
            evaluation = eval_match.group(1) if eval_match else "Not Found"
            evaluations.append(evaluation)
            # Extract reason text
            reason_match = re.search(reason_pattern, response, flags=re.IGNORECASE | re.DOTALL)
            # clean up unnecessary characters
            reason = re.sub(r"^[^:]*:\s*(?:\*\* )?", "", reason_match.group(1)).strip() if reason_match else "Not Found"
            reasons.append(reason)

        final_annotations = ""
        idx = 1
        for s,e,r in zip(statements, evaluations, reasons):
            final_annotations += f"[STATEMENT {idx}]: {s}\n"
            final_annotations += f"[EVALUATION {idx}]: {e}\n"
            final_annotations += f"[REASON {idx}]: {r}\n\n"
            idx += 1

        # STEP 3: get refined response (error-corrected response)
        error_rectification_prompt = get_error_rectification_prompt(initial_response, final_annotations)
        refined_response = qwen_vlm.get_response(img_path, error_rectification_prompt)

        # Append the evaluation and reason to each claim's information
        evaluated_claims = []

        for i, claim in enumerate(statements):            
            _claim = {
                "claim": claim,
                "evaluation": evaluations[i],
                "reason": reasons[i]
            }
            evaluated_claims.append(_claim)
        
        # Compile the final evaluation result
        final_result = {
            "image": entry['image'],
            "prompt": entry['prompt'],
            "initial_response": initial_response, # A
            "qwen_annotations": final_annotations, # B
            "evaluated_claims": evaluated_claims, # C
            "refined_response": refined_response, # D
        }
        # A and D are most important, since they can be used for preference tuning methods like DPO
        # B and C are just for your reference

        # Add the final result to the list
        all_results.append(final_result)

    # Save all results to a single JSON file
    with open(SAVE_PATH, "w") as outfile:
        json.dump(all_results, outfile, indent=4)
    print(f"\nSaved to {SAVE_PATH}\n")

if __name__ == "__main__":
    main()