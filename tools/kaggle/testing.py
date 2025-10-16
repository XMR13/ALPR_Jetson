import json

# Test apakah file bisa di-parse
with open('yolox_plate_train.ipynb', 'r', encoding='utf-8') as f:
    try:
        data = json.load(f)
        print("JSON valid!")
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
        print(f"Line: {e.lineno}, Column: {e.colno}")