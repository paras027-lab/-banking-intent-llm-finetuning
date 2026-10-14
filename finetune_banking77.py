"""
Fine-tune Qwen2.5-1.5B-Instruct with QLoRA to classify banking customer
messages into 77 intents, and compare against a zero-shot baseline.

Run on a free Colab/Kaggle T4 GPU:
    pip install -q transformers datasets peft trl bitsandbytes accelerate scikit-learn
    python finetune_banking77.py
"""
import argparse
import json
import random

import torch
from datasets import load_dataset
from peft import LoraConfig
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=3000, help="training examples (max 10003)")
parser.add_argument("--n_eval", type=int, default=500, help="test examples (max 3080)")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--skip_baseline", action="store_true", help="skip the zero-shot baseline eval")
cli = parser.parse_args()

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_TRAIN = cli.n_train
N_EVAL = cli.n_eval
SEED = 42
TAG = f"n{N_TRAIN}_ep{cli.epochs}_r{cli.lora_r}_eval{N_EVAL}"
OUT_DIR = f"banking77-qwen-lora-{TAG}"

random.seed(SEED)

# ---------------------------------------------------------------- data
# The original PolyAI repo uses a loading script, which newer `datasets`
# versions no longer support, so we use a parquet mirror of the same data.
ds = load_dataset("mteb/banking77")
_pairs = sorted(set(zip(ds["train"]["label"], ds["train"]["label_text"])))
LABELS = [name for _, name in _pairs]  # index == label id
print(f"{len(LABELS)} intents, {len(ds['train'])} train / {len(ds['test'])} test examples")
train = ds["train"].shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds["train"]))))
test = ds["test"].shuffle(seed=SEED).select(range(min(N_EVAL, len(ds["test"]))))


def zero_shot_prompt(text):
    # Baseline gets the full label list, which makes it a fair (strong) baseline.
    return (
        "Classify the banking customer message into exactly one intent.\n"
        f"Allowed intents: {', '.join(LABELS)}\n"
        f"Message: {text}\n"
        "Answer with the intent name only."
    )


def short_prompt(text):
    # Fine-tuned model learns the labels, so the prompt stays short.
    return (
        "Classify the banking customer message into an intent.\n"
        f"Message: {text}\nIntent:"
    )


def to_sft(example):
    return {
        "prompt": [{"role": "user", "content": short_prompt(example["text"])}],
        "completion": [{"role": "assistant", "content": LABELS[example["label"]]}],
    }


train_sft = train.map(to_sft, remove_columns=train.column_names)

# --------------------------------------------------------------- model
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto"
)


# ---------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, prompt_fn, name, batch_size=16):
    model.eval()
    preds, golds = [], []
    for i in range(0, len(test), batch_size):
        batch = test[i : i + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": prompt_fn(t)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in batch["text"]
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc, max_new_tokens=15, do_sample=False, pad_token_id=tok.pad_token_id
        )
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for g, y in zip(gen, batch["label"]):
            g = g.strip().split("\n")[0].strip()
            preds.append(g if g in LABELS else "INVALID")
            golds.append(LABELS[y])
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro", labels=LABELS, zero_division=0)
    invalid = sum(p == "INVALID" for p in preds) / len(preds)
    print(f"[{name}] accuracy={acc:.3f}  macro_f1={f1:.3f}  invalid_rate={invalid:.3f}")
    return {"accuracy": acc, "macro_f1": f1, "invalid_rate": invalid}


results = {"config": vars(cli)}
if not cli.skip_baseline:
    print("Evaluating zero-shot baseline...")
    results["baseline_zero_shot"] = evaluate(model, zero_shot_prompt, "baseline")

# ------------------------------------------------------------ training
lora = LoraConfig(
    r=cli.lora_r,
    lora_alpha=2 * cli.lora_r,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
args = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=cli.epochs,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    gradient_checkpointing=True,
    max_length=256,
    report_to="none",  # switch to "wandb" to track runs for your write-up
    seed=SEED,
)
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_sft,
    peft_config=lora,
    processing_class=tok,
)
# fp16 mixed precision needs trainable weights in float32 (the GradScaler
# can't unscale bf16/fp16 gradients). Newer libraries may load adapters in bf16.
for p in trainer.model.parameters():
    if p.requires_grad:
        p.data = p.data.float()

print("Training...")
trainer.train()
trainer.model.config.use_cache = True

# ----------------------------------------------------------- evaluate
print("Evaluating fine-tuned model...")
results["finetuned"] = evaluate(trainer.model, short_prompt, "fine-tuned")

trainer.model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
with open(f"results_{TAG}.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
