import json
from PIL import Image
import random
import matplotlib.pyplot as plt

with open("Qwen_HAL_Annotations.json", "r") as f:
    annotations = json.load(f)

while True:

    sample = random.choice(annotations)

    # load image
    img_path = '/home/DTC_SSD/HAL_Detection/POVID/mini_povid_images/' + sample['image']
    image = Image.open(img_path)

    print(f"\n + Prompt: \n{sample['prompt']}")
    print(f"\n + Initial response: \n{sample['initial_response']}\n")

    for claim in sample['evaluated_claims']:
        print(f"  - Claim: {claim['claim']}")
        print(f"    Evaluation: {claim['evaluation']}")
        print(f"    Reason: {claim['reason']}\n")

    print(f"\n + Refined response:\n{sample['refined_response']}\n")

    print("\n\t\t\t\t\t===========================================================================\n")

    plt.imshow(image)
    plt.show()