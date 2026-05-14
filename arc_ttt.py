"""
Test-Time Training orchestrator for SN5 Hone miner.

Strategy (from NVARC 2025 / ARChitects 2024 recipe):
- Train ONE global LoRA adapter on ALL eval tasks' train_examples
- Augmentations: transpose × rotation × color permutation × example shuffle × n_repeat
- Adapter saved to /app/models/ttt_adapter/ for vLLM to load in inference phase

This is the load-bearing technical recipe ported from
da-fr/arc-prize-2024/training_code/run_evaluation_Llama-rearc_with_ttt.py.

Compute budget (H200 80GB):
- Mistral-NeMo-8B-bnb-4bit + LoRA r=64 = ~10GB total VRAM
- TTT_REPEAT=24 (vs NVARC's 48) → 100 tasks × 24 = 2400 sequences
  → ~1200 optimizer steps → ~30-40 min on H200
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List


# ── Config ──────────────────────────────────────────────────────────────────

BASE_MODEL = os.getenv(
    "TTT_BASE_MODEL",
    "da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit",
)
ADAPTER_DIR = Path(os.getenv("TTT_ADAPTER_DIR", "/app/models/ttt_adapter"))
TTT_REPEAT = int(os.getenv("TTT_REPEAT", "24"))   # NVARC uses 48; we cut to fit 1h
TTT_LORA_RANK = int(os.getenv("TTT_LORA_RANK", "64"))
TTT_MAX_TOKENS = int(os.getenv("TTT_MAX_TOKENS", "8192"))


# ── SN5 → ARC format converter ──────────────────────────────────────────────

def sn5_to_arc_format(sn5_tasks: List[Dict]) -> Dict:
    """
    Convert SN5 task list to Kaggle ARC challenges format.

    SN5 format:
        {"task_hash": "...", "train_examples": [...], "test_input": [...]}

    ARC format:
        {"task_id": {"train": [...], "test": [{"input": [...]}, ...]}}
    """
    arc = {}
    for task in sn5_tasks:
        task_id = task.get("task_hash") or f"task_{len(arc)}"
        train = [
            {"input": ex["input"], "output": ex["output"]}
            for ex in task.get("train_examples", [])
        ]
        # SN5 gives single test_input — wrap in a list per ARC convention
        test_inp = task.get("test_input")
        test = [{"input": test_inp}] if test_inp is not None else []
        arc[task_id] = {"train": train, "test": test}
    return arc


# ── Main entry ──────────────────────────────────────────────────────────────

def run_ttt(input_json_path: Path) -> bool:
    """
    Train ONE LoRA adapter across all SN5 eval tasks' train_examples.
    Returns True if adapter saved successfully.

    Heavy imports (unsloth, torch, peft) are done inside the function so that
    if TTT is disabled the prep phase doesn't even import them.
    """
    print("\n" + "=" * 60)
    print(f"TTT PHASE - Train one LoRA adapter on all tasks")
    print("=" * 60)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Adapter dir: {ADAPTER_DIR}")
    print(f"  Repeat n:    {TTT_REPEAT}")
    print(f"  LoRA rank:   {TTT_LORA_RANK}")

    # MERGED_DIR is where vLLM will look for the model. We MUST produce something
    # there, even if TTT fails entirely — otherwise validator's vLLM crashes the job.
    merged_dir = Path(os.getenv("TTT_MERGED_DIR", "/app/models/mistral-ttt-merged"))

    # 1. Load SN5 tasks
    if not input_json_path.exists():
        print(f"⚠️ Input file not found: {input_json_path} — skipping TTT")
        _save_base_as_merged(merged_dir)
        return False

    data = json.loads(input_json_path.read_text())
    sn5_tasks = data.get("tasks", [])
    print(f"  SN5 tasks:   {len(sn5_tasks)}")
    if not sn5_tasks:
        print("⚠️ No tasks in input — skipping TTT")
        _save_base_as_merged(merged_dir)
        return False

    # 1b. Optionally mix in public ARC-AGI-2 dataset for more training signal.
    # NVARC winner used 103K sequences; we have 2400 from SN5 alone. Adding
    # 200-500 hard public tasks (after augmentation × n=24) brings us to ~14K.
    # Empirically: TTT generalizes better with diverse public examples.
    public_n = int(os.getenv("PUBLIC_ARC_N", "0"))
    if public_n > 0:
        try:
            from arc_public_loader import load_public_arc, stats
            public_tasks = load_public_arc(
                n=public_n,
                filter_by=os.getenv("PUBLIC_ARC_FILTER", "hard"),
            )
            if public_tasks:
                # Convert to SN5-task shape (task_hash, train_examples, test_input)
                # so the same sn5_to_arc_format works.
                for pt in public_tasks:
                    sn5_tasks.append({
                        "task_hash":      f"public:{pt['task_id']}",
                        "train_examples": pt["train_examples"],
                        "test_input":     pt.get("test_input"),
                    })
                pstats = stats(public_tasks)
                print(f"  +Public ARC: {pstats['n']} tasks "
                      f"(diff range {pstats['difficulty_range']}, "
                      f"avg pairs {pstats['train_pairs_per_task']['avg']:.1f})")
        except Exception as e:
            print(f"  ⚠️ Public ARC load failed (proceeding with SN5 only): {e}")

    print(f"  Total tasks: {len(sn5_tasks)}")

    # 2. Convert to ARC format
    arc_challenge = sn5_to_arc_format(sn5_tasks)

    # 3. Heavy imports
    try:
        from unsloth import (
            FastLanguageModel,
            UnslothTrainer as Trainer,
            unsloth_train,
            is_bfloat16_supported,
            UnslothTrainingArguments as TrainingArguments,
        )
        from datasets import Dataset

        from arc_loader import ArcDataset
        from model_tools import (
            InputMaskingDataCollator,
            load_unsloth_4bit,
            save_model_and_tokenizer,
        )
    except Exception as e:
        print(f"❌ TTT dependencies not installed ({e}) — skipping")
        _save_base_as_merged(merged_dir)
        return False

    # 4. Load base model
    print(f"  Loading base model: {BASE_MODEL}...")
    try:
        model, tokenizer = load_unsloth_4bit(BASE_MODEL)
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        _save_base_as_merged(merged_dir)
        return False

    # 5. Build ARC eval set
    arc_eval_set = ArcDataset(challenge=arc_challenge, solutions={}, is_orig=True)

    # 6. NVARC/ARChitects exact LoRA config (verbatim from
    #    da-fr/arc-prize-2024/training_code/run_evaluation_Llama-rearc_with_ttt.py)
    print("  Building LoRA adapter (r=64, alpha=16, rslora)...")
    model = FastLanguageModel.get_peft_model(
        model=model,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ],
        r=TTT_LORA_RANK,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    # 7. Format prompts (ARChitects 2024 default fmt_opts)
    fmt_opts = dict(
        preprompt="ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz",
        query_beg="I",
        reply_beg="\n+/-=O",
        reply_end="\n" + tokenizer.eos_token,
        lines_sep="\n",
        max_tokens=TTT_MAX_TOKENS,
    )

    # 8. Augment data — TTT trains on ALL tasks together, not per-task
    print(f"  Augmenting (n={TTT_REPEAT})...")
    train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
    train_set = (
        arc_eval_set
        .remove_test_data()
        .repeat(n=TTT_REPEAT, seed=0)
        .augment(**train_aug_opts)
    )
    train_list = train_set.as_list(len_name="text", **fmt_opts)
    print(f"  Training examples: {len(train_list)}")

    # 9. Run TTT
    print("  Starting TTT training...")
    FastLanguageModel.for_training(model)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(train_list),
        dataset_text_field="text",
        max_seq_length=fmt_opts["max_tokens"],
        data_collator=InputMaskingDataCollator(
            instruction_template=fmt_opts["query_beg"],
            response_template=fmt_opts["reply_beg"],
            mlm=False,
            tokenizer=tokenizer,
            mask_first_n_examples=0,
        ),
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_ratio=0.25,
            num_train_epochs=1,
            learning_rate=1e-4,
            embedding_learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.00,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir="/tmp/ttt_output",
            save_strategy="no",
            report_to="none",
        ),
    )
    try:
        trainer_stats = unsloth_train(trainer)
        print(f"  Training stats: {trainer_stats}")
    except Exception as e:
        print(f"❌ TTT training failed: {e}")
        _save_base_as_merged(merged_dir)
        return False

    # 10. Save adapter (intermediate — for backup, not used by vLLM directly)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(ADAPTER_DIR))
        tokenizer.save_pretrained(str(ADAPTER_DIR))
        print(f"  Adapter saved (intermediate): {ADAPTER_DIR}")
    except Exception as e:
        print(f"❌ Adapter save failed: {e}")
        _save_base_as_merged(merged_dir)
        return False

    # 11. Merge LoRA into base model → save as full model for vLLM.
    # WHY: sandbox passes vllm extra_args as `--key value` (always 2 tokens),
    # but `--enable-lora` is argparse store_true. Cannot enable LoRA via
    # extra_args, so we MUST ship a merged model.
    #
    # HOW: use unsloth's save_pretrained_merged which properly handles
    # bnb-4bit dequantization + LoRA merge → FP16 output (vLLM-compatible).
    # Manual PEFT merge on bnb-4bit is unreliable: torch_dtype=fp16 does NOT
    # override the saved quantization_config, so weights stay quantized and
    # merge produces broken output.
    MERGED_DIR = merged_dir
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Merging via unsloth.save_pretrained_merged() → {MERGED_DIR}")
    try:
        # Primary path: unsloth handles bnb dequant + LoRA merge internally
        model.save_pretrained_merged(
            str(MERGED_DIR),
            tokenizer,
            save_method="merged_16bit",  # auto dequant 4bit → fp16 + merge LoRA
        )
        print(f"✅ Merged model saved (unsloth) to {MERGED_DIR}")
    except Exception as e1:
        print(f"⚠️ unsloth save_pretrained_merged failed: {e1}")
        print("    Trying PEFT manual merge fallback...")
        try:
            # Fallback: use PEFT's merge_and_unload (works in peft 0.13+ on QLoRA,
            # but may need bitsandbytes config to load base correctly first)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_bnb = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
            )
            peft_model = PeftModel.from_pretrained(base_bnb, str(ADAPTER_DIR))
            merged = peft_model.merge_and_unload()  # dequantizes + merges → fp16
            merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
            AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(str(MERGED_DIR))
            print(f"✅ Merged model saved (PEFT fallback) to {MERGED_DIR}")
        except Exception as e2:
            print(f"❌ Both merge paths failed: unsloth={e1!r}, peft={e2!r}")
            import traceback
            traceback.print_exc()
            # Last resort: copy base model to merged path so vLLM still starts
            _save_base_as_merged(MERGED_DIR)
            return False

    print("=" * 60)
    print("TTT PHASE COMPLETED - Status: success (merged model ready for vLLM)")
    print("=" * 60)
    return True


def _save_base_as_merged(merged_dir: Path) -> None:
    """
    Fallback when TTT/merge fails: save base model to merged path so vLLM doesn't crash.
    Without this, validator's vLLM looks for /app/models/mistral-ttt-merged and fails the whole job.
    """
    try:
        import shutil
        save_dir = Path(os.getenv("MODEL_SAVE_DIR", "/app/models"))
        base_path = save_dir / BASE_MODEL.replace("/", "--")
        if not base_path.exists():
            print(f"⚠️ Base model not found at {base_path}, cannot fallback")
            return
        merged_dir.mkdir(parents=True, exist_ok=True)
        # Copy / hardlink files
        for f in base_path.iterdir():
            dest = merged_dir / f.name
            if not dest.exists():
                try:
                    shutil.copy2(f, dest)
                except Exception:
                    pass
        print(f"✅ Fallback: base model copied to {merged_dir}")
    except Exception as e:
        print(f"❌ Fallback copy also failed: {e}")


if __name__ == "__main__":
    # CLI usage: python3 arc_ttt.py /input/miner_current_dataset.json
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/input/miner_current_dataset.json")
    ok = run_ttt(p)
    sys.exit(0 if ok else 1)
