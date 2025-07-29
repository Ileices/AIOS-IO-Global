#!/usr/bin/env python3
# json_to_aios_dataset_wizard.py
# Full-UX wizard to convert your giant chat-history JSON into trainable datasets
# for instruction-tuning + code generation with AIOS IO metadata (RBY/AE).
# - Large-file streaming (ijson if present; optional guided install)
# - Interactive wizard (no coding knowledge needed)
# - Clean/normalize/deflatten content, extract code blocks, build datasets:
#     * sft_messages.jsonl (single-turn rows)
#     * packed_conversations.jsonl (messages[])
#     * nl2code.jsonl (instruction -> code)
#     * code_infill.jsonl (prefix/middle/suffix)
# - RBY labels, AE lineage IDs, deterministic train/val/test split
# - Optional PII scrubbing (emails, phones, API-like keys)
# - Manifest + trainer recipes + HPC shard map
# - Re-run .bat and saved_config.json

import argparse, hashlib, json, os, re, sys, time, random, subprocess, shutil
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Terminal UX helpers (ANSI colors, spinners, prompts)
# ──────────────────────────────────────────────────────────────────────────────


def supports_color() -> bool:
    return sys.stdout.isatty()


def c(txt, color):
    if not supports_color():
        return txt
    colors = {
        "g": "\x1b[32m",
        "y": "\x1b[33m",
        "b": "\x1b[34m",
        "c": "\x1b[36m",
        "r": "\x1b[31m",
        "m": "\x1b[35m",
        "w": "\x1b[37m",
        "bold": "\x1b[1m",
        "reset": "\x1b[0m",
    }
    return f"{colors.get(color, '')}{txt}{colors['reset']}"


def hr():
    print(
        c(
            "──────────────────────────────────────────────────────────────────────────────",
            "w",
        )
    )


def prompt_str(q, default=None, required=True):
    while True:
        s = input(
            c(f"› {q}" + (f" [{default}]" if default else "") + ": ", "c")
        ).strip()
        if not s and default is not None:
            return default
        if s or not required:
            return s


def prompt_yesno(q, default=True):
    d = "Y/n" if default else "y/N"
    while True:
        s = input(c(f"› {q} [{d}]: ", "c")).strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def info(msg):
    print(c(f"[INFO] {msg}", "g"))


def warn(msg):
    print(c(f"[WARN] {msg}", "y"))


def err(msg):
    print(c(f"[ERROR] {msg}", "r"))


# ──────────────────────────────────────────────────────────────────────────────
# Optional streaming parser (ijson); offer guided install if missing.
# ──────────────────────────────────────────────────────────────────────────────

HAS_IJSON = False
try:
    import ijson  # type: ignore

    HAS_IJSON = True
except Exception:
    HAS_IJSON = False


def maybe_install_ijson():
    global HAS_IJSON
    if HAS_IJSON:
        return
    want = prompt_yesno(
        "ijson not found. Install it now for safer streaming on huge files?", True
    )
    if not want:
        warn(
            "Proceeding without ijson (the script will try non-streaming or line-by-line modes)."
        )
        return
    try:
        python = sys.executable or "python"
        subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "ijson"])
        import ijson  # retry import

        info("ijson installed successfully.")
        HAS_IJSON = True
    except Exception as e:
        warn(f"Could not install ijson automatically ({e}). Continuing without it.")


# ──────────────────────────────────────────────────────────────────────────────
# Core text/code processing
# ──────────────────────────────────────────────────────────────────────────────

LANG_HINTS = {
    "python": [r"\bdef\b", r"\bimport\b", r"print\(", r"^\s*class\s+"],
    "json": [r"^\s*\{", r"^\s*\[", r'":\s*'],
    "html": [r"<html", r"<div", r"</"],
    "bash": [r"#!/bin/bash", r"\bmkdir\b", r"\bgrep\b"],
    "javascript": [r"\bfunction\b", r"console\.log", r"=>\s*\{"],
    "typescript": [r":\s*\w+\s*(=|;)", r"interface\s+\w+"],
    "markdown": [r"^# ", r"\[[^\]]+\]\([^)]+\)"],
    "sql": [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b"],
}

CODE_FENCE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def guess_language(text: str, lang_hint: str | None) -> str | None:
    if lang_hint:
        return lang_hint.lower()
    sample = text[:2000]
    for lang, pats in LANG_HINTS.items():
        hits = sum(1 for p in pats if re.search(p, sample, re.MULTILINE))
        if hits >= 2:
            return lang
    return None


def unflatten_content(s: str) -> str:
    if s is None:
        return ""
    if "\n" in s and s.count("\\n") < 3:
        return s
    backslash_n = s.count("\\n")
    real_n = s.count("\n")
    if backslash_n >= 3 and backslash_n > real_n:
        try:
            s2 = s.encode("utf-8").decode("unicode_escape")
            if s2.count("\n") >= backslash_n:
                return s2
        except Exception:
            pass
    return s


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    return s


def extract_code_blocks(text: str):
    blocks = []
    for m in CODE_FENCE.finditer(text):
        lang = (m.group(1) or "").strip()
        code = m.group(2)
        blocks.append({"lang": lang.lower() if lang else None, "text": code})
    return blocks


def rby_label(role: str, text: str) -> str:
    role = (role or "").lower()
    if role == "user":
        return "R"  # Perception/input
    if role == "assistant":
        if CODE_FENCE.search(text) or re.search(r"```", text):
            return "Y"  # Execution
        return "B"  # Cognition
    return "B"


PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PII_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}(?!\d)"
)
PII_KEYLIKE = re.compile(r"(?:sk-|AKIA|ASIA|xoxb-|xoxp-|ghp_)[A-Za-z0-9\-=_]{8,}")


def scrub_pii(text: str) -> str:
    text = PII_EMAIL.sub("[EMAIL]", text)
    text = PII_PHONE.sub("[PHONE]", text)
    text = PII_KEYLIKE.sub("[KEY]", text)
    return text


def safe_jsonl_write(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def shard_path(base_dir: Path, name: str, shard_idx: int) -> Path:
    return base_dir / f"{name}.{shard_idx:03d}.jsonl"


def split_from_key(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    if h < 5:
        return "test"
    if h < 10:
        return "val"
    return "train"


# ──────────────────────────────────────────────────────────────────────────────
# Iteration over the input export
# ──────────────────────────────────────────────────────────────────────────────


def iter_root(input_path: Path):
    """
    Yields items that look like chat turns.
    Accepts:
      - JSON array of turns
      - Object with 'items' or 'data'
      - Newline-delimited JSON (JSONL-like)
    """
    if HAS_IJSON:
        with input_path.open("rb") as f:
            try:
                for item in ijson.items(f, "item"):
                    yield item
                return
            except Exception:
                f.seek(0)
                try:
                    for item in ijson.items(f, "data.item"):
                        yield item
                    return
                except Exception:
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue
    else:
        # Non-streaming fallback (may require RAM for huge files)
        with input_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            obj = json.loads(content)
        except Exception as e:
            err(f"Could not parse JSON: {e}")
            sys.exit(1)
        if isinstance(obj, list):
            for it in obj:
                yield it
        elif isinstance(obj, dict):
            seq = obj.get("data") or obj.get("items") or []
            for it in seq if isinstance(seq, list) else []:
                yield it


# ──────────────────────────────────────────────────────────────────────────────
# Builders
# ──────────────────────────────────────────────────────────────────────────────


def build_msg_row(turn, conv_id, turn_index, scrub=False):
    role = turn.get("role", "")
    title = turn.get("title", "")
    created_at = turn.get("create_time", None)
    raw_content = turn.get("content", "")

    if isinstance(raw_content, dict):
        parts = raw_content.get("parts", [])
        text = (
            "\n".join(str(p) for p in parts) if parts else raw_content.get("text", "")
        )
    else:
        text = str(raw_content)

    text = normalize_text(unflatten_content(text))
    if scrub:
        text = scrub_pii(text)
    code_blocks = extract_code_blocks(text)

    row = {
        "id": sha256(f"{conv_id}:{turn_index}:{role}:{text[:200]}"),
        "conv_id": conv_id,
        "turn_index": turn_index,
        "role": role,
        "content": text,
        "title": title,
        "created_at": int(created_at) if isinstance(created_at, (int, float)) else None,
        "rby": rby_label(role, text),
        "code_blocks": code_blocks if code_blocks else None,
        # AE lineage marker (deterministic)
        "ae_id": f"AE::{conv_id}::{turn_index}",
    }
    return row


def pack_conversations(rows_by_conv, out_path, shard_size):
    # Pack messages per conversation into one JSONL sample:
    # {"conv_id":..., "title":..., "messages":[{"role":"user","content":...}, ...]}
    shard_idx = 0
    wrote = 0
    curr_path = shard_path(out_path, "packed_conversations", shard_idx)
    for conv_id, rows in rows_by_conv.items():
        msgs = [{"role": r["role"], "content": r["content"]} for r in rows]
        title = next((r["title"] for r in rows if r.get("title")), "")
        sample = {
            "conv_id": conv_id,
            "title": title,
            "messages": msgs,
            "meta": {"rby_mix": list({r["rby"] for r in rows})},
        }
        safe_jsonl_write(curr_path, sample)
        wrote += 1
        if wrote % shard_size == 0:
            shard_idx += 1
            curr_path = shard_path(out_path, "packed_conversations", shard_idx)
    return wrote


# ──────────────────────────────────────────────────────────────────────────────
# Main processing pipeline (with wizard UX)
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline(cfg):
    input_path = Path(cfg["input"])
    out_base = Path(cfg["out"])
    shard_size = int(cfg["shard_size"])
    enable_pii = bool(cfg["pii_scrub"])
    build_nl2code = bool(cfg["build_nl2code"])
    build_infill = bool(cfg["build_infill"])
    build_packed = bool(cfg["build_packed"])
    dedup_assistant_code = bool(cfg["dedup_assistant_code"])
    min_code_len = int(cfg["min_code_len_chars"])

    # Prepare folders
    (out_base / "sft").mkdir(parents=True, exist_ok=True)
    (out_base / "nl2code").mkdir(parents=True, exist_ok=True)
    (out_base / "code_infill").mkdir(parents=True, exist_ok=True)
    (out_base / "packed").mkdir(parents=True, exist_ok=True)
    (out_base / "_meta").mkdir(parents=True, exist_ok=True)

    # Shard writers for SFT
    sft_counts = {"train": 0, "val": 0, "test": 0}
    sft_shard_idx = {"train": 0, "val": 0, "test": 0}
    sft_curr_paths = {
        split: shard_path(out_base / "sft", "sft_messages", sft_shard_idx[split])
        for split in sft_counts
    }

    # Conv cache for second pass builds
    conv_cache = {}  # conv_id -> [rows]
    # Dedup memory (assistant code)
    seen_code_hashes = set()

    total = 0
    t0 = time.time()

    info("Scanning and writing SFT rows (streaming)…")
    for turn in iter_root(input_path):
        title = str(turn.get("title", "") or "").strip()
        conv_id = (
            title
            if title
            else f"conv-{sha256(str(turn.get('create_time', ''))[:10])[:12]}"
        )
        rows = conv_cache.setdefault(conv_id, [])
        turn_index = len(rows)
        row = build_msg_row(turn, conv_id, turn_index, scrub=enable_pii)
        rows.append(row)

        split = split_from_key(conv_id)
        safe_jsonl_write(sft_curr_paths[split], row)
        sft_counts[split] += 1
        if sft_counts[split] % shard_size == 0:
            sft_shard_idx[split] += 1
            sft_curr_paths[split] = shard_path(
                out_base / "sft", "sft_messages", sft_shard_idx[split]
            )

        total += 1
        if total % 50000 == 0:
            rate = total / max(1.0, (time.time() - t0))
            print(c(f"  … {total} turns @ {rate:.1f} turns/s", "w"))

    info("First pass complete.")

    # Second pass: build NL→Code and Infill
    nl2_total = 0
    infill_total = 0
    if build_nl2code or build_infill:
        info("Building NL→Code and Infill datasets…")
        nl2_dir = out_base / "nl2code"
        infill_dir = out_base / "code_infill"
        nl2_shard = infill_shard = 0
        nl2_path = shard_path(nl2_dir, "nl2code", nl2_shard)
        infill_path = shard_path(infill_dir, "code_infill", infill_shard)

        for conv_id, rows in conv_cache.items():
            last_user = None
            for r in rows:
                if r["role"].lower() == "user":
                    last_user = r
                    continue
                if r["role"].lower() == "assistant" and r.get("code_blocks"):
                    instr = (
                        last_user["content"]
                        if last_user
                        else "Generate code based on prior context."
                    )
                    for cb in r["code_blocks"]:
                        code_text = cb["text"]
                        if dedup_assistant_code:
                            h = sha256(code_text)
                            if h in seen_code_hashes:
                                continue
                            seen_code_hashes.add(h)
                        lang = guess_language(code_text, cb.get("lang"))
                        if build_nl2code:
                            nl2 = {
                                "id": sha256(conv_id + instr[:128] + code_text[:128]),
                                "instruction": instr,
                                "language": (lang or cb.get("lang") or "unknown"),
                                "completion": code_text,
                                "meta": {
                                    "source_conv": conv_id,
                                    "assistant_turn": r["turn_index"],
                                    "rby": "Y",
                                    "ae_lineage": r["ae_id"],
                                },
                            }
                            safe_jsonl_write(nl2_path, nl2)
                            nl2_total += 1
                            if nl2_total % shard_size == 0:
                                nl2_shard += 1
                                nl2_path = shard_path(nl2_dir, "nl2code", nl2_shard)

                        if build_infill and len(code_text) >= max(min_code_len, 80):
                            cut1 = len(code_text) // 3
                            cut2 = 2 * len(code_text) // 3
                            infill = {
                                "id": sha256(conv_id + str(r["turn_index"]) + "infill"),
                                "language": (lang or cb.get("lang") or "unknown"),
                                "prefix": code_text[:cut1],
                                "middle": code_text[cut1:cut2],
                                "suffix": code_text[cut2:],
                                "meta": {
                                    "source_conv": conv_id,
                                    "assistant_turn": r["turn_index"],
                                    "rby": "Y",
                                    "ae_lineage": r["ae_id"],
                                },
                            }
                            safe_jsonl_write(infill_path, infill)
                            infill_total += 1
                            if infill_total % shard_size == 0:
                                infill_shard += 1
                                infill_path = shard_path(
                                    infill_dir, "code_infill", infill_shard
                                )

    # Packed conversations
    packed_total = 0
    if build_packed:
        info("Packing full conversations for long-context training…")
        packed_total = pack_conversations(conv_cache, out_base / "packed", shard_size)

    # Manifest + recipes + shard maps
    info("Writing manifest and training recipes…")
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input": str(input_path),
        "output_dir": str(out_base),
        "counts": {
            "sft_train": sft_counts["train"],
            "sft_val": sft_counts["val"],
            "sft_test": sft_counts["test"],
            "nl2code": nl2_total,
            "code_infill": infill_total,
            "packed_conversations": packed_total,
        },
        "options": cfg,
    }
    (out_base / "_meta" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Trainer recipes (Transformers, llama.cpp LoRA stub)
    transformers_recipe = {
        "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_seq_length": 4096,
        "datasets": {
            "sft": str(out_base / "sft"),
            "nl2code": str(out_base / "nl2code"),
            "code_infill": str(out_base / "code_infill"),
            "packed": str(out_base / "packed"),
        },
        "packing": True,
        "mask_user_only_loss": False,
        "train_val_test_split": [0.90, 0.05, 0.05],
        "training": {
            "epochs": 2,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.0,
            "lora": {
                "enabled": True,
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
        },
    }
    (out_base / "_meta" / "transformers_recipe.json").write_text(
        json.dumps(transformers_recipe, indent=2), encoding="utf-8"
    )

    llama_cpp_recipe = {
        "base_gguf": "path/to/base-model.gguf",
        "train_data": str(out_base / "sft"),
        "n_threads": 8,
        "n_ctx": 4096,
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 2,
    }
    (out_base / "_meta" / "llamacpp_lora_stub.json").write_text(
        json.dumps(llama_cpp_recipe, indent=2), encoding="utf-8"
    )

    # HPC shard map (simple): list all produced shards with sizes
    shard_map = {"sft": [], "nl2code": [], "code_infill": [], "packed": []}
    for dkey in shard_map.keys():
        dpath = out_base / dkey
        if dpath.exists():
            for p in sorted(dpath.glob("*.jsonl")):
                shard_map[dkey].append({"file": str(p), "bytes": p.stat().st_size})
    (out_base / "_meta" / "hpc_shard_map.json").write_text(
        json.dumps(shard_map, indent=2), encoding="utf-8"
    )

    # Re-run script config + Windows .bat
    (out_base / "_meta" / "saved_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    bat = out_base / "_meta" / "rerun_dataset_builder.bat"
    bat.write_text(
        f'@echo off\r\n"{sys.executable}" "{Path(__file__).absolute()}" --noninteractive "{(out_base / "_meta" / "saved_config.json").absolute()}"\r\npause\r\n',
        encoding="utf-8",
    )

    hr()
    info("DONE.")
    print(c(json.dumps(manifest["counts"], indent=2), "b"))
    print(c(f"Output dir: {out_base}", "g"))
    print(c(f"Re-run without prompts: {bat}", "y"))


def wizard():
    hr()
    print(c("AIOS IO Dataset Builder – Guided Setup", "bold"))
    hr()
    maybe_install_ijson()
    input_path = prompt_str(
        "Enter path to your giant JSON export (drag & drop allowed)", required=True
    )
    out_dir = prompt_str(
        "Enter output directory for datasets",
        default=str(Path.cwd() / "datasets_aios_io"),
    )
    shard_size = prompt_str("Lines per shard file", default="100000")
    pii_scrub = prompt_yesno(
        "Enable basic PII scrubbing (emails/phones/keys) on text?", True
    )
    build_nl2code = prompt_yesno("Build NL→Code pairs?", True)
    build_infill = prompt_yesno("Build Code-Infill samples?", True)
    build_packed = prompt_yesno("Build Packed Conversations (messages[])?", True)
    dedup_code = prompt_yesno("De-duplicate identical assistant code blocks?", True)
    min_code_len = prompt_str(
        "Min code length (chars) for infill samples", default="120"
    )

    cfg = {
        "input": input_path,
        "out": out_dir,
        "shard_size": int(shard_size),
        "pii_scrub": pii_scrub,
        "build_nl2code": build_nl2code,
        "build_infill": build_infill,
        "build_packed": build_packed,
        "dedup_assistant_code": dedup_code,
        "min_code_len_chars": int(min_code_len),
    }
    hr()
    print(c("Configuration:", "b"))
    print(json.dumps(cfg, indent=2))
    hr()
    if prompt_yesno("Proceed with these settings?", True):
        run_pipeline(cfg)
    else:
        warn("Canceled by user.")


def noninteractive(cfg_path: str):
    try:
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception as e:
        err(f"Could not load config: {e}")
        sys.exit(2)
    maybe_install_ijson()
    run_pipeline(cfg)


def main():
    ap = argparse.ArgumentParser(description="AIOS IO Dataset Builder (Wizard)")
    ap.add_argument(
        "--noninteractive",
        nargs="?",
        help="Path to saved_config.json to run without prompts",
    )
    args = ap.parse_args()
    if args.noninteractive:
        noninteractive(args.noninteractive)
    else:
        wizard()


if __name__ == "__main__":
    main()
