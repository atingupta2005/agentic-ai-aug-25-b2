# Using `device_map="auto"` + Offloading to Prevent OOM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "big-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",              # automatically split layers
    offload_folder="offload_dir",   # folder to store offloaded weights
    offload_state_dict=True,        # offload state dict if GPU is full
    torch_dtype="auto"
)
```

# Notes – Handling OOM in Training

## ZeRO / DeepSpeed Offload

* It shards optimizer states + params across devices.
* You can offload to CPU RAM or even disk if desperate.
* Hugging Face + Accelerate makes it easy.

```bash
pip install accelerate deepspeed
accelerate config
accelerate launch train.py
```

Config (ds_config.json):

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu" },
    "offload_param": { "device": "cpu" }
  }
}
```

