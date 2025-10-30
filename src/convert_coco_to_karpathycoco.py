import json
import os
import random
#from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# === Paths ===
train_anno_path = "/pscratch/sd/m/megha89/ddl-project/data/annotations/captions_train2014.json"
val_anno_path   = "/pscratch/sd/m/megha89/ddl-project/data/annotations/captions_val2014.json"
output_path     = "/pscratch/sd/m/megha89/ddl-project/data/annotations/karpathy_coco_format.json"

# === Helper: Convert one split (train/val) ===
def convert_to_karpathy_format(anno_path, split_name):
    print(f"Loading {split_name} annotations from {anno_path}")
    with open(anno_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: {
        "split": split_name,
        "filepath": split_name + "2014",
        "filename": img["file_name"],
        "sentences": [],
        "cocoid": img["id"]
    } for img in data["images"]}

    for ann in data["annotations"]:
        sent = ann["caption"].strip()
        tokens = word_tokenize(sent.lower())
        img_id = ann["image_id"]

        if img_id in images:
            images[img_id]["sentences"].append({
                "tokens": tokens,
                "raw": sent
            })

    print(f"Converted {len(images)} {split_name} images.")
    return list(images.values())

# === Convert both splits ===
train_data = convert_to_karpathy_format(train_anno_path, "train")
val_data   = convert_to_karpathy_format(val_anno_path, "val")

# === Combine ===
all_data = {"images": train_data + val_data}

# === Optional: Create Karpathy-style mini splits ===
# e.g., 5000 val, 5000 test from the original val set
random.seed(42)
val_ids = random.sample(range(len(val_data)), 5000)
test_ids = random.sample([i for i in range(len(val_data)) if i not in val_ids], 5000)

for i, img in enumerate(val_data):
    if i in val_ids:
        img["split"] = "val"
    elif i in test_ids:
        img["split"] = "test"
    else:
        img["split"] = "restval"

# === Save final JSON ===
with open(output_path, "w") as f:
    json.dump(all_data, f, indent=2)

print(f"\n✅ Karpathy-style dataset saved to: {output_path}")

