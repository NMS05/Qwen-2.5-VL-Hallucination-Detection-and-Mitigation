import json
from PIL import Image
import random
import matplotlib.pyplot as plt
from utils import IMAGE_DIR, SAVE_PATH

with open(SAVE_PATH, "r") as f:
    annotations = json.load(f)

while True:

    sample = random.choice(annotations)

    # load image
    img_path = IMAGE_DIR + sample['image']
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