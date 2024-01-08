import sys
import os
import mii
import torch

model_path = "/home/star/models/opt-1.3b"
pipe = mii.pipeline(model_name_or_path=model_path)
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)

