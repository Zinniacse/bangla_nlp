"""
UBMEC fixed-size breakpoint experiment.

Research question:
Across nominal 400 MB, 200 MB, 100 MB, and 50 MB comparison ranges, which
approach retains the highest Accuracy and Macro F1? Every result and figure
uses the exact measured serialized weight-file size, not the nominal range.

Approaches shown in the final figures:
1. 12-layer BanglaBERT Teacher
2. Direct Training Control (No KD)
3. Standard Logit-KD Control
4. Proposed Method

Important terminology:
- "Standard Logit-KD Control" is a controlled implementation of the paper's
  black-box KD objective: plain supervised loss, fixed-temperature logit
  KD, and no proposed enhancement. It is not an interpolation of
  unreported paper results.
- The published Distil + Prune + Quant row (178 MB, 47.62% Accuracy,
  44.42% Macro F1) is saved separately as a fixed external reference.
- Deployment mechanisms are used only to reach target memory budgets. They
  are recorded in audit tables and disclosed in the final figure note.

Controlled comparison:
- Around 200 MB and 100 MB, all three methods use the same 5-layer
  architecture and exactly the same initial weights.
- Around 50 MB, all three methods use the same 1-layer architecture and
  exactly the same initial weights.
- Architecture and initialization are controlled; the proposed approach
  additionally includes its declared progressive teacher path.
- Proposed uses validation-selected profiles and recovery, so the main result
  is a full-pipeline comparison, not a one-factor loss ablation.
- The smaller deployment points use bitsandbytes LLM.int8. This backend is
  reported explicitly and is not presented as an exact reproduction of a
  different static PTQ implementation. No 4-bit/NF4 result is used.

Proposed method (components are profile-dependent):
- 5-layer profiles use balanced/smoothed supervision, confidence-guided KD,
  adaptive temperature, feature KD, and optional relation/SupCon/R-Drop
- tiny profiles use fixed-temperature progressive KD and stable supervision;
  balanced supervision and feature KD are validation-selected options
- all proposed profiles use exponential moving average weights
- progressive distillation for the tiny model:
  BanglaBERT -> 5-layer proposed assistant -> 1-layer student
- input-compatible early-layer initialization for the 1-layer student
- tiny-specific stable objectives that avoid noisy pairwise losses
- deployment validation among the top FP32-validation epoch shortlist before
  the test set is evaluated
- low-learning-rate recovery accepted only through validation selection

The code never edits or fabricates measured Accuracy. It writes explicit
numerical comparisons; multi-seed uncertainty is required before a numerical
advantage is described as statistically supported.
"""

# %% [markdown]
# ## 1. Install dependencies

# %%
import importlib.util
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as package_version


def install_if_missing(package, import_name=None):
    if importlib.util.find_spec(import_name or package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


for package, import_name in (
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("bitsandbytes", "bitsandbytes"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("sentencepiece", "sentencepiece"),
    ("protobuf", "google.protobuf"),
):
    install_if_missing(package, import_name)

# %% [markdown]
# ## 2. Imports and configuration

# %%
import gc
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

# Quantization and dtype warnings are important audit evidence.
warnings.filterwarnings("default")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass
class Config:
    # live is the full, multi-stage research protocol. Run validation first,
    # then deliberately unlock the final test with UBMEC_RUN_FINAL_TEST=true.
    run_mode: str = "live"  # demo, quick, one_hour, live
    dataset_path: str = "UBMEC.csv"
    output_dir: str = "ubmec_fixed_size_breakpoint_outputs_quant_aware_v5"
    experiment_version: str = "professor-feedback-v7"
    teacher_name: str = "csebuetnlp/banglabert"
    labels: tuple = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
    seed: int = 42
    # Keep the dataset split fixed while training_seed changes across repeated
    # runs. This isolates optimization randomness from split randomness.
    split_seed: int = 42
    one_hour_max_rows: int = 5000

    # One compact model produces the ~200 MB and ~100 MB points.
    compact_layers: int = 5
    # The tiny model produces the ~50 MB point.
    tiny_layers: int = 1
    # Layer zero consumes embeddings directly, unlike the teacher's final
    # layer, which expects eleven preceding transformations.
    tiny_source_layer: int = 0

    max_length: int = 128
    batch_size: int = 8
    # Batch size remains paper-compatible while the optimizer sees a more
    # stable effective batch of 32 samples in live mode.
    grad_accum_steps: int = 4
    num_workers: int = 2

    # Paper protocol: batch size 8, 20 epochs, early-stop patience 3.
    teacher_epochs: int = 20
    student_epochs: int = 20
    recovery_epochs: int = 3
    tiny_recovery_epochs: int = 0
    teacher_lr: float = 2e-5
    student_lr: float = 3e-5
    recovery_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    patience: int = 3

    validation_size: float = 0.15
    test_size: float = 0.15
    label_smoothing: float = 0.05
    teacher_plain_ce_weight: float = 0.70

    fixed_temperature: float = 4.0
    max_temperature: float = 6.0
    min_temperature: float = 2.0
    confidence_power: float = 2.0
    min_confidence_weight: float = 0.10
    wrong_teacher_weight: float = 0.15
    correct_teacher_label_blend: float = 0.05
    wrong_teacher_label_blend: float = 0.55
    contrastive_temperature: float = 0.10
    ema_decay: float = 0.999
    assistant_logit_weight: float = 0.50
    tiny_temperature: float = 3.0

    # Validation chooses between a compact, stable KD objective and the full
    # proposed objective. The test set is never used for this selection.
    proposed_profiles: tuple = ("core", "full")
    tiny_proposed_profiles: tuple = (
        "tiny_progressive",
        "tiny_balanced",
        "tiny_feature",
    )

    # FP32 shortlists the best epochs; deployment precision ranks that declared
    # shortlist. Tiny uses INT8 validation, while compact balances FP32/INT8.
    quantized_validation_selection: bool = True
    compact_int8_selection_weight: float = 0.50
    deployment_epoch_candidates: int = 3

    # A conservative LLM.int8 configuration for small fine-tuned classifiers.
    # This one common recipe is used for every method.
    int8_threshold: float = 2.0
    int8_skip_modules: tuple = ("classifier",)
    run_precision_audit: bool = True
    quantization_drop_warning_points: float = 3.0
    quantization_min_top1_agreement: float = 0.90
    block_final_test_on_quantization_collapse: bool = True
    # First run locks training/validation choices. Enable only on a second,
    # deliberate run after reviewing the validation audit.
    run_final_test: bool = False
    # This project has already displayed metrics from the seed-42 holdout.
    # Keep this true unless a genuinely unseen external/outer holdout is used.
    prior_test_split_exposed: bool = True

    # Ablations are expensive, so they are opt-in. The companion multi-seed
    # runner enables them for the dedicated ablation study.
    run_ablation_study: bool = False
    compact_ablation_profiles: tuple = (
        "ablation_balanced",
        "ablation_adaptive",
        "ablation_confidence",
        "ablation_feature",
        "ablation_ema",
    )
    # Reviewer-requested leave-one-out ablation: every profile equals the
    # "core" EADM profile with exactly one mechanism removed. Enable with
    # UBMEC_ABLATION_MODE=leave_one_out (default keeps the cumulative study).
    ablation_mode: str = "cumulative"  # cumulative | leave_one_out
    compact_leave_one_out_profiles: tuple = (
        "loo_no_adaptive_temperature",
        "loo_no_confidence_gating",
        "loo_no_feature_kd",
        "loo_no_balanced_supervision",
        "loo_no_ema",
    )

    reuse_checkpoints: bool = True
    reuse_teacher_cache: bool = True
    benchmark_batches: int = 30

    # These are internal deployment controls. They never appear in figures.
    internal_full_backend: str = "FULL"
    internal_int8_backend: str = "BNB_LLM_INT8"

    # Approximate final model-size breakpoints shown in the paper figures.
    target_sizes_mb: tuple = (400, 200, 100, 50)
    target_tolerance_fraction: float = 0.25

    # Published external reference. It is not expanded into a fake curve.
    paper_accuracy: float = 47.62
    paper_macro_f1: float = 44.42
    paper_size_mb: float = 178.0

    def apply_mode(self):
        if self.run_mode == "demo":
            self.max_length = 96
            self.batch_size = 8
            self.teacher_epochs = 2
            self.student_epochs = 4
            self.recovery_epochs = 1
            self.grad_accum_steps = 1
            self.proposed_profiles = ("core",)
            self.tiny_proposed_profiles = ("tiny_progressive",)
            self.quantized_validation_selection = False
            self.run_precision_audit = False
            self.run_final_test = True
            self.patience = 2
            self.benchmark_batches = 5
        elif self.run_mode == "quick":
            self.teacher_epochs = 4
            self.student_epochs = 8
            self.recovery_epochs = 2
            self.grad_accum_steps = 2
            self.run_final_test = True
            self.patience = 3
            self.benchmark_batches = 10
        elif self.run_mode == "one_hour":
            # Bounded single-file protocol for a 16 GB T4. Runtime depends on
            # Colab load, but this removes the expensive multi-seed/profile/
            # ablation sweep and is designed to finish near one hour.
            self.max_length = 96
            self.batch_size = 16
            self.grad_accum_steps = 1
            self.num_workers = 0
            self.teacher_epochs = 4
            self.student_epochs = 5
            self.recovery_epochs = 0
            self.tiny_recovery_epochs = 0
            self.patience = 2
            self.proposed_profiles = ("core",)
            self.tiny_proposed_profiles = ("tiny_progressive",)
            self.deployment_epoch_candidates = 2
            self.run_ablation_study = False
            self.run_precision_audit = False
            self.block_final_test_on_quantization_collapse = False
            self.run_final_test = True
            self.benchmark_batches = 10
        elif self.run_mode != "live":
            raise ValueError(
                "run_mode must be demo, quick, one_hour, or live"
            )


def environment_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false; received {value!r}")


cfg = Config()
cfg.run_mode = os.environ.get("UBMEC_RUN_MODE", cfg.run_mode)
cfg.seed = int(os.environ.get("UBMEC_TRAINING_SEED", cfg.seed))
cfg.split_seed = int(os.environ.get("UBMEC_SPLIT_SEED", cfg.split_seed))
cfg.output_dir = os.environ.get("UBMEC_OUTPUT_DIR", cfg.output_dir)
cfg.run_final_test = environment_bool(
    "UBMEC_RUN_FINAL_TEST", cfg.run_final_test
)
cfg.run_ablation_study = environment_bool(
    "UBMEC_RUN_ABLATION", cfg.run_ablation_study
)
cfg.ablation_mode = os.environ.get("UBMEC_ABLATION_MODE", cfg.ablation_mode)
if cfg.ablation_mode not in {"cumulative", "leave_one_out"}:
    raise ValueError("UBMEC_ABLATION_MODE must be cumulative or leave_one_out")
if cfg.ablation_mode == "leave_one_out":
    cfg.compact_ablation_profiles = cfg.compact_leave_one_out_profiles
cfg.prior_test_split_exposed = environment_bool(
    "UBMEC_PRIOR_TEST_EXPOSED", cfg.prior_test_split_exposed
)
cfg.apply_mode()
compact_profile_override = os.environ.get("UBMEC_COMPACT_PROFILE")
tiny_profile_override = os.environ.get("UBMEC_TINY_PROFILE")
if compact_profile_override:
    cfg.proposed_profiles = (compact_profile_override.strip(),)
if tiny_profile_override:
    cfg.tiny_proposed_profiles = (tiny_profile_override.strip(),)
if not 0.0 < cfg.assistant_logit_weight <= 1.0:
    raise ValueError("assistant_logit_weight must be in (0, 1].")
if not 0.0 <= cfg.compact_int8_selection_weight <= 1.0:
    raise ValueError("compact_int8_selection_weight must be in [0, 1].")
if cfg.deployment_epoch_candidates < 1:
    raise ValueError("deployment_epoch_candidates must be at least 1.")
if cfg.tiny_source_layer < 0:
    raise ValueError("tiny_source_layer cannot be negative.")
evidence_status = (
    "EXPLORATORY_PREVIOUSLY_OBSERVED_HOLDOUT"
    if cfg.prior_test_split_exposed
    else "CONFIRMATORY_UNSEEN_HOLDOUT"
)

config_payload = asdict(cfg)
for operational_field in (
    "reuse_checkpoints",
    "reuse_teacher_cache",
    "benchmark_batches",
    "run_precision_audit",
    "run_final_test",
):
    config_payload.pop(operational_field, None)
config_digest = hashlib.sha256(
    json.dumps(
        config_payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    ).hexdigest()


def protocol_source_digest():
    """Hash the protocol while allowing only the final-test switch to flip."""
    source_name = globals().get("__file__")
    if not source_name:
        if cfg.run_mode == "live":
            raise RuntimeError(
                "Live mode must run from a saved canonical .py file so the "
                "training protocol can be hashed and locked."
            )
        return f"experiment-version:{cfg.experiment_version}"
    source_path = Path(source_name).resolve()
    if not source_path.is_file():
        if cfg.run_mode == "live":
            raise RuntimeError(
                f"Cannot hash the live protocol source: {source_path}"
            )
        return f"experiment-version:{cfg.experiment_version}"
    source_text = source_path.read_text(encoding="utf-8")
    normalized_source = re.sub(
        r"run_final_test:\s*bool\s*=\s*(?:True|False)",
        "run_final_test: bool = <FINAL_TEST_SWITCH>",
        source_text,
        count=1,
    )
    return hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()


protocol_digest = protocol_source_digest()
output_dir = (
    Path(cfg.output_dir)
    / cfg.run_mode
    / f"{config_digest[:12]}-{protocol_digest[:12]}"
)
output_dir.mkdir(parents=True, exist_ok=True)
selection_lock_path = output_dir / "selection_lock_manifest.json"
run_status_path = output_dir / "run_status.json"
holdout_registry_path = (
    Path(cfg.output_dir) / cfg.run_mode / "holdout_registry.json"
)
locked_selection = None
if cfg.run_mode == "live" and cfg.run_final_test:
    if not selection_lock_path.exists():
        raise RuntimeError(
            "Final test is locked. First run this exact configuration with "
            "run_final_test=False to create selection_lock_manifest.json."
        )
    locked_selection = json.loads(
        selection_lock_path.read_text(encoding="utf-8")
    )
    if (
        locked_selection.get("experiment_version") != cfg.experiment_version
        or locked_selection.get("config_digest") != config_digest
        or locked_selection.get("protocol_source_digest") != protocol_digest
    ):
        raise RuntimeError(
            "The validation lock does not match this experiment/configuration. "
            "Run stage one again with run_final_test=False."
        )

label2id = {label: index for index, label in enumerate(cfg.labels)}
id2label = {index: label for label, index in label2id.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

if cfg.run_mode == "live" and cfg.quantized_validation_selection and not use_amp:
    raise RuntimeError(
        "Live deployment-aware selection requires a CUDA GPU and a working "
        "bitsandbytes INT8 backend."
    )


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(cfg.seed)
print(f"Device: {device}")
if use_amp:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Run mode: {cfg.run_mode}")
print(f"Training seed: {cfg.seed}; fixed split seed: {cfg.split_seed}")
print(
    "Target final model sizes:",
    ", ".join(f"approximately {size} MB" for size in cfg.target_sizes_mb),
)
if cfg.run_mode == "demo":
    print("WARNING: demo metrics only verify the pipeline. They are not research results.")
if cfg.run_mode == "one_hour":
    print(
        "One-hour T4 mode: maximum 5,000 rows, one training seed, fixed "
        "proposed profiles, and a single validation-to-test execution."
    )
    warnings.warn(
        "one_hour mode is a bounded exploratory run. Use live multi-seed "
        "results for final statistical claims."
    )
if cfg.run_mode == "live" and cfg.prior_test_split_exposed:
    warnings.warn(
        "The seed-42 holdout has already informed method development. This "
        "run is exploratory; use a genuinely unseen external holdout or an "
        "outer/nested evaluation for confirmatory claims."
    )

# %% [markdown]
# ## 3. Dataset and fixed split

# %%
def repair_possible_mojibake(text):
    text = str(text)
    markers = ("\u00c3\u00a0\u00c2\u00a6", "\u00c3\u00a0\u00c2\u00a7", "\u00c3\u00a2\u00e2\u201a\u00ac")
    if any(marker in text for marker in markers):
        try:
            return text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    return text


def load_ubmec(path):
    data = None
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            data = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if data is None:
        raise RuntimeError(f"Could not read {path}: {last_error}")

    missing = {"text", "classes"}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}; found {data.columns.tolist()}")

    data = data[["text", "classes"]].dropna().copy()
    data["text"] = data["text"].map(repair_possible_mojibake)
    data["classes"] = data["classes"].astype(str).str.strip().str.lower()
    data = data[data["classes"].isin(label2id)].copy()
    data["label"] = data["classes"].map(label2id).astype(int)
    data = data[data["text"].str.len() > 0].reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid UBMEC rows remain after cleaning.")
    return data


def stratified_sample(data, maximum_rows):
    if len(data) <= maximum_rows:
        return data.reset_index(drop=True)
    pieces = []
    for _, group in data.groupby("label"):
        count = max(2, round(maximum_rows * len(group) / len(data)))
        pieces.append(
            group.sample(
                min(count, len(group)), random_state=cfg.split_seed
            )
        )
    return (
        pd.concat(pieces)
        .sample(frac=1.0, random_state=cfg.split_seed)
        .reset_index(drop=True)
    )


data = load_ubmec(cfg.dataset_path)
if cfg.run_mode == "demo":
    data = stratified_sample(data, 600)
elif cfg.run_mode == "quick":
    data = stratified_sample(data, 5000)
elif cfg.run_mode == "one_hour":
    data = stratified_sample(data, cfg.one_hour_max_rows)

train_df, temporary_df = train_test_split(
    data,
    test_size=cfg.validation_size + cfg.test_size,
    stratify=data["label"],
    random_state=cfg.split_seed,
)
relative_test_size = cfg.test_size / (cfg.validation_size + cfg.test_size)
valid_df, test_df = train_test_split(
    temporary_df,
    test_size=relative_test_size,
    stratify=temporary_df["label"],
    random_state=cfg.split_seed,
)
train_df, valid_df, test_df = [
    frame.reset_index(drop=True) for frame in (train_df, valid_df, test_df)
]
print(f"Rows: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_name, use_fast=True)


class EmotionDataset(Dataset):
    def __init__(self, frame):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        item["sample_id"] = torch.tensor(index, dtype=torch.long)
        return item


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(frame, shuffle=False, run_seed=None):
    generator = None
    if run_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(run_seed)
    return DataLoader(
        EmotionDataset(frame),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=cfg.num_workers,
        pin_memory=use_amp,
    )


train_loader = make_loader(train_df, True, cfg.seed)
train_cache_loader = make_loader(train_df, False)
valid_loader = make_loader(valid_df, False)
test_loader = make_loader(test_df, False)

class_counts = np.bincount(train_df["label"].to_numpy(), minlength=len(cfg.labels))
balanced_weights = len(train_df) / (
    len(cfg.labels) * np.maximum(class_counts, 1)
)
class_weights = torch.tensor(balanced_weights, dtype=torch.float32, device=device)

split_identity = pd.concat(
    [
        train_df[["text", "label"]].assign(split="train"),
        valid_df[["text", "label"]].assign(split="valid"),
        test_df[["text", "label"]].assign(split="test"),
    ],
    ignore_index=True,
)
split_digest = hashlib.sha256(
    pd.util.hash_pandas_object(split_identity, index=False).values.tobytes()
).hexdigest()
if (
    locked_selection is not None
    and locked_selection.get("split_digest") != split_digest
):
    raise RuntimeError(
        "The final-test split does not match the validation lock."
    )

(output_dir / "run_metadata.json").write_text(
    json.dumps(
        {
            "experiment_version": cfg.experiment_version,
            "run_mode": cfg.run_mode,
            "training_seed": cfg.seed,
            "split_seed": cfg.split_seed,
            "split_digest": split_digest,
            "config_digest": config_digest,
            "protocol_digest": protocol_digest,
            "run_ablation_study": cfg.run_ablation_study,
            "ablation_mode": cfg.ablation_mode,
            "shared_teacher": os.environ.get("UBMEC_SHARED_TEACHER"),
            "compact_proposed_profiles": list(cfg.proposed_profiles),
            "tiny_proposed_profiles": list(cfg.tiny_proposed_profiles),
            "evidence_status": evidence_status,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

# %% [markdown]
# ## 4. Model and evaluation utilities

# %%
def teacher_config():
    return AutoConfig.from_pretrained(
        cfg.teacher_name,
        num_labels=len(cfg.labels),
        label2id=label2id,
        id2label=id2label,
        output_hidden_states=True,
    )


def create_teacher():
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.teacher_name,
        config=teacher_config(),
        ignore_mismatched_sizes=True,
    )


def installed_package_version(package_name):
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return "not-installed"


teacher_seed_model = create_teacher()
runtime_asset_payload = {
    "python": sys.version,
    "torch": torch.__version__,
    "transformers": installed_package_version("transformers"),
    "bitsandbytes": installed_package_version("bitsandbytes"),
    "accelerate": installed_package_version("accelerate"),
    "teacher_config": teacher_seed_model.config.to_dict(),
    "tokenizer_vocab": tokenizer.get_vocab(),
    "tokenizer_special_tokens": tokenizer.special_tokens_map,
    "tokenizer_init": tokenizer.init_kwargs,
}
runtime_asset_digest = hashlib.sha256(
    json.dumps(
        runtime_asset_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
).hexdigest()
if (
    locked_selection is not None
    and locked_selection.get("runtime_assets", {}).get("digest")
    != runtime_asset_digest
):
    raise RuntimeError(
        "Tokenizer/model/runtime assets differ from the stage-one lock. "
        "Final-test checkpoints will not be mutated."
    )


def load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def completion_marker(checkpoint):
    return checkpoint.with_name(f"{checkpoint.name}.complete")


def checkpoint_is_complete(checkpoint):
    marker = completion_marker(checkpoint)
    if not checkpoint.exists() or not marker.exists():
        return False
    marker_text = marker.read_text(encoding="utf-8")
    return (
        f"version={cfg.experiment_version}" in marker_text
        and f"split={split_digest}" in marker_text
        and f"config={config_digest}" in marker_text
        and f"protocol={protocol_digest}" in marker_text
        and f"runtime_assets={runtime_asset_digest}" in marker_text
    )


def mark_checkpoint_complete(checkpoint, description):
    completion_marker(checkpoint).write_text(
        (
            f"{description}\n"
            f"version={cfg.experiment_version}\n"
            f"split={split_digest}\n"
            f"config={config_digest}\n"
            f"protocol={protocol_digest}\n"
            f"runtime_assets={runtime_asset_digest}\n"
        ),
        encoding="utf-8",
    )


def cpu_state_dict(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def split_batch(batch, target=device):
    sample_ids = batch.pop("sample_id")
    moved = {key: value.to(target) for key, value in batch.items()}
    return moved, sample_ids


def cleanup():
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()


def parameter_count_m(model):
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000


def tensor_size_mb(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return (
        sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        / 1_000_000
    )


def weighted_ce(logits, labels):
    logits_fp32 = logits.float()
    return F.cross_entropy(
        logits_fp32,
        labels,
        weight=class_weights.to(dtype=torch.float32, device=logits.device),
        label_smoothing=cfg.label_smoothing,
    )


def plain_supervised_loss(logits, labels):
    return F.cross_entropy(logits.float(), labels)


def base_paper_supervised_loss(logits, labels):
    # The paper describes binary cross-entropy for the supervised student
    # term. One-hot targets make that definition explicit for six classes.
    one_hot = F.one_hot(labels, num_classes=len(cfg.labels)).float()
    return F.binary_cross_entropy_with_logits(logits.float(), one_hot)


def proposed_supervised_loss(logits, labels):
    # Preserve overall accuracy with ordinary CE while the balanced,
    # label-smoothed term improves minority-class Macro F1.
    return 0.60 * plain_supervised_loss(
        logits, labels
    ) + 0.40 * weighted_ce(logits, labels)


def teacher_supervised_loss(logits, labels):
    """Keep teacher accuracy high without ignoring minority classes."""
    plain_weight = cfg.teacher_plain_ce_weight
    return plain_weight * plain_supervised_loss(
        logits, labels
    ) + (1.0 - plain_weight) * weighted_ce(logits, labels)


def optimizer_scheduler(model, loader, epochs, learning_rate):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(term in name for term in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(term in name for term in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(parameter_groups, lr=learning_rate)
    total_steps = max(
        1, math.ceil(len(loader) / cfg.grad_accum_steps) * max(1, epochs)
    )
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, name, benchmark=False, return_outputs=False):
    model.eval()
    labels, predictions, measured_times = [], [], []
    logits_batches = []
    if use_amp and benchmark:
        torch.cuda.reset_peak_memory_stats()

    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc=f"Evaluating {name}")
    ):
        batch, _ = split_batch(raw_batch)
        if use_amp:
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(**batch, output_hidden_states=False)
        if use_amp:
            torch.cuda.synchronize()
        if (
            benchmark
            and batch_index > 0
            and len(measured_times) < cfg.benchmark_batches
        ):
            measured_times.append(
                (time.perf_counter() - start) / batch["labels"].shape[0]
            )
        logits_cpu = outputs.logits.detach().float().cpu()
        labels.extend(batch["labels"].detach().cpu().tolist())
        predictions.extend(logits_cpu.argmax(dim=-1).tolist())
        if return_outputs:
            logits_batches.append(logits_cpu)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "latency_ms_sample": (
            float(np.mean(measured_times) * 1000)
            if measured_times
            else np.nan
        ),
        "peak_gpu_mb": (
            torch.cuda.max_memory_allocated() / 1_000_000
            if use_amp and benchmark
            else np.nan
        ),
    }
    print(
        f"{name}: accuracy={metrics['accuracy'] * 100:.2f}, "
        f"macro_f1={metrics['macro_f1'] * 100:.2f}"
    )
    if return_outputs:
        metrics["_labels"] = np.asarray(labels, dtype=np.int64)
        metrics["_predictions"] = np.asarray(predictions, dtype=np.int64)
        metrics["_logits"] = torch.cat(logits_batches, dim=0)
    return metrics


def validation_score(metrics):
    # Accuracy is the main thesis target, while Macro F1 prevents majority-class
    # solutions from being selected.
    return 0.60 * metrics["accuracy"] + 0.40 * metrics["macro_f1"]


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(
            self.decay, (1.0 + self.updates) / (10.0 + self.updates)
        )
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - decay)

    @torch.no_grad()
    def swap(self, model):
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                current = parameter.data
                parameter.data = self.shadow[name]
                self.shadow[name] = current

# %% [markdown]
# ## 5. Fine-tune the BanglaBERT teacher

# %%
def train_teacher(model, checkpoint):
    if cfg.reuse_checkpoints and checkpoint_is_complete(checkpoint):
        model.load_state_dict(load_state_dict(checkpoint))
        print(f"Loaded teacher checkpoint: {checkpoint}")
        return model
    if checkpoint.exists():
        print(f"Ignoring incomplete teacher checkpoint: {checkpoint}")

    seed_everything(cfg.seed)
    run_train_loader = make_loader(train_df, True, cfg.seed)
    model.to(device)
    optimizer, scheduler = optimizer_scheduler(
        model, run_train_loader, cfg.teacher_epochs, cfg.teacher_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state = cpu_state_dict(model)
    initial_metrics = evaluate(model, valid_loader, "Teacher initial valid")
    best_score = validation_score(initial_metrics)
    stale_epochs = 0

    for epoch in range(cfg.teacher_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(
            run_train_loader,
            desc=f"Teacher epoch {epoch + 1}/{cfg.teacher_epochs}",
        )
        for step, raw_batch in enumerate(progress_bar, 1):
            batch, _ = split_batch(raw_batch)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**batch, output_hidden_states=False).logits
                loss = teacher_supervised_loss(logits, batch["labels"])
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()

            if (
                step % cfg.grad_accum_steps == 0
                or step == len(run_train_loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress_bar.set_postfix(loss=float(loss.detach()) * cfg.grad_accum_steps)

        metrics = evaluate(model, valid_loader, "Teacher valid")
        score = validation_score(metrics)
        if score > best_score:
            best_score = score
            best_state = cpu_state_dict(model)
            torch.save(best_state, checkpoint)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint)
    mark_checkpoint_complete(checkpoint, "Teacher training complete")
    return model


teacher_checkpoint = output_dir / "teacher_banglabert.pt"
# Optional time saver for the multi-seed study: reuse one fine-tuned teacher
# across student training seeds (UBMEC_SHARED_TEACHER=<path to .pt>). The
# teacher is then a fixed reference and only student optimization randomness
# varies between seeds; this is recorded in run_metadata.json.
shared_teacher_path = os.environ.get("UBMEC_SHARED_TEACHER")
if (
    shared_teacher_path
    and not checkpoint_is_complete(teacher_checkpoint)
    and Path(shared_teacher_path).is_file()
):
    import shutil as _shutil

    _shutil.copyfile(shared_teacher_path, teacher_checkpoint)
    mark_checkpoint_complete(
        teacher_checkpoint,
        f"Teacher copied from shared checkpoint {shared_teacher_path}",
    )
    print(f"Using shared teacher checkpoint: {shared_teacher_path}")
teacher = train_teacher(teacher_seed_model, teacher_checkpoint)

# %% [markdown]
# ## 6. Compact model initialization and teacher-target caches

# %%
def select_source_layers(source_layers, target_layers):
    if target_layers < 1 or target_layers > source_layers:
        raise ValueError(
            f"target_layers must be between 1 and {source_layers}; "
            f"received {target_layers}"
        )
    if target_layers == 1:
        # A standalone layer consumes embeddings directly. Reusing the
        # source's first layer avoids feeding raw embeddings into a final
        # layer trained to expect many preceding transformations.
        return [min(cfg.tiny_source_layer, source_layers - 1)]
    return (
        np.linspace(0, source_layers - 1, target_layers)
        .round()
        .astype(int)
        .tolist()
    )


def build_compact_model(source_model, target_layers):
    source_layers = source_model.config.num_hidden_layers
    compact_config = deepcopy(source_model.config)
    compact_config.num_hidden_layers = target_layers
    compact_config.output_hidden_states = True
    compact_model = AutoModelForSequenceClassification.from_config(
        compact_config
    )

    if not hasattr(source_model, "electra") or not hasattr(
        compact_model, "electra"
    ):
        raise TypeError(
            "This experiment expects BanglaBERT to load as an ELECTRA classifier."
        )

    compact_model.electra.embeddings.load_state_dict(
        deepcopy(source_model.electra.embeddings.state_dict())
    )
    selected_layers = select_source_layers(source_layers, target_layers)
    for destination_layer, source_layer in enumerate(selected_layers):
        compact_model.electra.encoder.layer[destination_layer].load_state_dict(
            deepcopy(
                source_model.electra.encoder.layer[source_layer].state_dict()
            )
        )
    compact_model.classifier.load_state_dict(
        deepcopy(source_model.classifier.state_dict())
    )
    print(
        f"Built {target_layers}-layer model from source layers "
        f"{selected_layers}"
    )
    return compact_model, selected_layers


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_signature(path):
    return {
        "checkpoint_sha256": sha256_file(path),
        "split_digest": split_digest,
        "config_digest": config_digest,
        "protocol_digest": protocol_digest,
        "runtime_asset_digest": runtime_asset_digest,
        "max_length": cfg.max_length,
    }


@torch.no_grad()
def cache_targets(
    source_model,
    selected_layers,
    checkpoint_path,
    cache_path,
    cache_name,
):
    expected_signature = checkpoint_signature(checkpoint_path)
    if cfg.reuse_teacher_cache and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            cached.get("selected_layers") == selected_layers
            and cached.get("signature") == expected_signature
            and cached["logits"].shape[0] == len(train_df)
        ):
            print(f"Loaded cached targets: {cache_path}")
            return cached
        print(f"Rebuilding stale cache: {cache_path}")

    source_model.eval().to(device)
    total_rows = len(train_df)
    logits_cache = torch.empty(
        total_rows, len(cfg.labels), dtype=torch.float16
    )
    feature_cache = torch.empty(
        total_rows,
        len(selected_layers),
        source_model.config.hidden_size,
        dtype=torch.float16,
    )

    for raw_batch in tqdm(
        train_cache_loader, desc=f"Caching {cache_name} targets"
    ):
        batch, sample_ids = split_batch(raw_batch)
        batch.pop("labels")
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = source_model(**batch, output_hidden_states=True)
        logits_cache[sample_ids] = outputs.logits.detach().cpu().half()
        for feature_index, source_layer in enumerate(selected_layers):
            feature_cache[sample_ids, feature_index] = (
                outputs.hidden_states[source_layer + 1][:, 0]
                .detach()
                .cpu()
                .half()
            )

    cached = {
        "logits": logits_cache,
        "features": feature_cache,
        "selected_layers": selected_layers,
        "signature": expected_signature,
    }
    torch.save(cached, cache_path)
    source_model.cpu()
    cleanup()
    return cached


compact_template, compact_teacher_layers = build_compact_model(
    teacher, cfg.compact_layers
)
compact_initial_state = cpu_state_dict(compact_template)
del compact_template

tiny_template, tiny_teacher_layers = build_compact_model(
    teacher, cfg.tiny_layers
)
tiny_initial_state = cpu_state_dict(tiny_template)
del tiny_template

teacher_compact_targets = cache_targets(
    teacher,
    compact_teacher_layers,
    teacher_checkpoint,
    output_dir / "teacher_targets_compact.pt",
    "teacher-to-compact",
)
teacher_tiny_targets = cache_targets(
    teacher,
    tiny_teacher_layers,
    teacher_checkpoint,
    output_dir / "teacher_targets_tiny.pt",
    "teacher-to-tiny",
)
teacher.cpu()
cleanup()

# %% [markdown]
# ## 7. Distillation losses

# %%
def adaptive_temperature(progress):
    return cfg.min_temperature + (
        cfg.max_temperature - cfg.min_temperature
    ) * (1.0 - progress) ** 1.5


def confidence_weights(teacher_logits, labels):
    confidence = F.softmax(teacher_logits.float(), dim=-1).amax(dim=-1)
    weights = confidence.pow(cfg.confidence_power)
    teacher_is_correct = teacher_logits.argmax(dim=-1).eq(labels)
    reliability = torch.where(
        teacher_is_correct,
        torch.ones_like(weights),
        torch.full_like(weights, cfg.wrong_teacher_weight),
    )
    weights = weights * reliability
    weights = weights.clamp(cfg.min_confidence_weight, 1.0)
    return weights / weights.mean().clamp(min=1e-8)


def logit_kd_loss(
    student_logits,
    teacher_logits,
    temperature,
    sample_weights,
    labels=None,
    label_correct_teacher=False,
):
    student_log_probs = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    teacher_probs = F.softmax(
        teacher_logits.float() / temperature, dim=-1
    )
    if label_correct_teacher:
        if labels is None:
            raise ValueError("labels are required for label-corrected KD")
        one_hot = F.one_hot(
            labels, num_classes=student_logits.shape[-1]
        ).float()
        teacher_is_correct = teacher_logits.argmax(dim=-1).eq(labels)
        blend = torch.where(
            teacher_is_correct,
            torch.full_like(
                teacher_is_correct, cfg.correct_teacher_label_blend,
                dtype=torch.float32,
            ),
            torch.full_like(
                teacher_is_correct, cfg.wrong_teacher_label_blend,
                dtype=torch.float32,
            ),
        ).unsqueeze(1)
        teacher_probs = (
            (1.0 - blend) * teacher_probs + blend * one_hot
        )
    per_sample = F.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    ).sum(dim=-1)
    return (
        per_sample * sample_weights.float()
    ).mean() * temperature**2


def probability_kd_loss(
    student_logits,
    target_probabilities,
    temperature,
    sample_weights,
):
    """KL distillation when multiple teachers were mixed as probabilities."""
    student_log_probs = F.log_softmax(
        student_logits.float() / temperature, dim=-1
    )
    per_sample = F.kl_div(
        student_log_probs,
        target_probabilities.float(),
        reduction="none",
    ).sum(dim=-1)
    return (
        per_sample * sample_weights.float()
    ).mean() * temperature**2


def normalized_feature_loss(student_states, target_features):
    losses = []
    for layer_index in range(target_features.shape[1]):
        student_cls = F.normalize(
            student_states[layer_index + 1][:, 0].float(), dim=-1
        )
        target_cls = F.normalize(
            target_features[:, layer_index].float(), dim=-1
        )
        losses.append(
            (1.0 - (student_cls * target_cls).sum(dim=-1)).mean()
        )
    return torch.stack(losses).mean()


def relation_kd_loss(student_states, target_features):
    student_cls = F.normalize(student_states[-1][:, 0].float(), dim=-1)
    target_cls = F.normalize(target_features[:, -1].float(), dim=-1)
    student_relations = student_cls @ student_cls.transpose(0, 1)
    target_relations = target_cls @ target_cls.transpose(0, 1)
    if student_relations.shape[0] <= 1:
        return F.mse_loss(student_relations, target_relations)
    mask = ~torch.eye(
        student_relations.shape[0],
        dtype=torch.bool,
        device=student_relations.device,
    )
    return F.mse_loss(
        student_relations[mask], target_relations[mask]
    )


def symmetric_kl(first_logits, second_logits):
    first_log = F.log_softmax(first_logits.float(), dim=-1)
    second_log = F.log_softmax(second_logits.float(), dim=-1)
    return 0.5 * (
        F.kl_div(first_log, second_log.exp(), reduction="batchmean")
        + F.kl_div(second_log, first_log.exp(), reduction="batchmean")
    )


def supervised_contrastive_loss(features, labels):
    if features.shape[0] <= 1:
        return torch.zeros((), device=features.device)

    # Keep the similarity matrix in FP32 even when the outer training block
    # uses CUDA autocast. The large negative mask value overflows FP16.
    with torch.cuda.amp.autocast(enabled=False):
        normalized = F.normalize(features.float(), dim=-1)
        similarity = (
            normalized @ normalized.transpose(0, 1)
        ) / cfg.contrastive_temperature
        identity = torch.eye(
            features.shape[0], dtype=torch.bool, device=features.device
        )
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~identity
        valid_anchors = positive_mask.any(dim=1)
        if not valid_anchors.any():
            return torch.zeros((), device=features.device)

        similarity = similarity.masked_fill(identity, -1e9)
        log_probabilities = similarity - torch.logsumexp(
            similarity, dim=1, keepdim=True
        )
        positive_counts = positive_mask.sum(dim=1).clamp(min=1)
        mean_positive_log_prob = (
            log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_counts
        )
        return -mean_positive_log_prob[valid_anchors].mean()

# %% [markdown]
# ## 8. Train matched students

# %%
def new_student(num_layers, initial_state):
    student_config = deepcopy(teacher.config)
    student_config.num_hidden_layers = num_layers
    student_config.output_hidden_states = True
    student = AutoModelForSequenceClassification.from_config(student_config)
    student.load_state_dict(deepcopy(initial_state))
    return student


def proposed_profile_components(profile):
    """Return explicit component switches for reproducible ablations."""
    compact_profiles = {
        "ablation_balanced": {
            "adaptive": False,
            "confidence": False,
            "label_correction": False,
            "feature": False,
            "ema": False,
            "second_forward": False,
            "progressive": False,
        },
        "ablation_adaptive": {
            "adaptive": True,
            "confidence": False,
            "label_correction": False,
            "feature": False,
            "ema": False,
            "second_forward": False,
            "progressive": False,
        },
        "ablation_confidence": {
            "adaptive": True,
            "confidence": True,
            "label_correction": True,
            "feature": False,
            "ema": False,
            "second_forward": False,
            "progressive": False,
        },
        "ablation_feature": {
            "adaptive": True,
            "confidence": True,
            "label_correction": True,
            "feature": True,
            "ema": False,
            "second_forward": False,
            "progressive": False,
        },
        "ablation_ema": {
            "adaptive": True,
            "confidence": True,
            "label_correction": True,
            "feature": True,
            "ema": True,
            "second_forward": False,
            "progressive": False,
        },
        "core": {
            "adaptive": True,
            "confidence": True,
            "label_correction": True,
            "feature": True,
            "ema": True,
            "second_forward": False,
            "progressive": False,
        },
        # Leave-one-out variants of "core" (one mechanism removed each).
        "loo_no_adaptive_temperature": {
            "adaptive": False,
            "confidence": True,
            "label_correction": True,
            "feature": True,
            "ema": True,
            "second_forward": False,
            "progressive": False,
        },
        "loo_no_confidence_gating": {
            "adaptive": True,
            "confidence": False,
            "label_correction": False,
            "feature": True,
            "ema": True,
            "second_forward": False,
            "progressive": False,
        },
        "loo_no_feature_kd": {
            "adaptive": True,
            "confidence": True,
            "label_correction": True,
            "feature": False,
            "ema": True,
            "second_forward": False,
            "progressive": False,
        },
        "loo_no_balanced_supervision": {
            "adaptive": True,
            "confidence": True,
            "label_correction": True,
            "feature": True,
            "ema": True,
            "second_forward": False,
            "progressive": False,
        },
        "loo_no_ema": {
            "adaptive": True,
            "confidence": True,
            "label_correction": True,
            "feature": True,
            "ema": False,
            "second_forward": False,
            "progressive": False,
        },
        "full": {
            "adaptive": True,
            "confidence": True,
            "label_correction": True,
            "feature": True,
            "ema": True,
            "second_forward": True,
            "progressive": False,
        },
    }
    tiny_profiles = {
        "tiny_teacher_only": {
            "adaptive": False,
            "confidence": False,
            "label_correction": False,
            "feature": False,
            "ema": True,
            "second_forward": False,
            "progressive": False,
        },
        "tiny_progressive": {
            "adaptive": False,
            "confidence": False,
            "label_correction": False,
            "feature": False,
            "ema": True,
            "second_forward": False,
            "progressive": True,
        },
        "tiny_balanced": {
            "adaptive": False,
            "confidence": False,
            "label_correction": False,
            "feature": False,
            "ema": True,
            "second_forward": False,
            "progressive": True,
        },
        "tiny_feature": {
            "adaptive": False,
            "confidence": False,
            "label_correction": False,
            "feature": True,
            "ema": True,
            "second_forward": False,
            "progressive": True,
        },
    }
    profiles = {**compact_profiles, **tiny_profiles}
    if profile not in profiles:
        raise ValueError(
            f"Unknown proposed profile {profile!r}; "
            f"expected one of {sorted(profiles)}"
        )
    return profiles[profile]


epoch_candidate_rows = []


def train_student(
    method,
    num_layers,
    initial_state,
    target_cache,
    checkpoint,
    epochs=None,
    learning_rate=None,
    recovery_mode=False,
    proposed_profile="full",
):
    if cfg.reuse_checkpoints and checkpoint_is_complete(checkpoint):
        student = new_student(num_layers, initial_state)
        student.load_state_dict(load_state_dict(checkpoint))
        print(f"Loaded {method} checkpoint: {checkpoint}")
        return student.cpu()
    if checkpoint.exists():
        print(f"Ignoring incomplete {method} checkpoint: {checkpoint}")

    epochs = cfg.student_epochs if epochs is None else epochs
    learning_rate = (
        cfg.student_lr if learning_rate is None else learning_rate
    )
    # Every method receives the same shuffled batches for a given seed.
    seed_everything(cfg.seed)
    run_train_loader = make_loader(train_df, True, cfg.seed)
    student = new_student(num_layers, initial_state).to(device)
    optimizer, scheduler = optimizer_scheduler(
        student, run_train_loader, epochs, learning_rate
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    use_proposed = method == "Proposed Method"
    components = (
        proposed_profile_components(proposed_profile)
        if use_proposed
        else None
    )
    use_tiny_profile = use_proposed and proposed_profile.startswith("tiny_")
    ema = (
        ExponentialMovingAverage(student, cfg.ema_decay)
        if use_proposed and components["ema"]
        else None
    )

    initial_metrics = evaluate(
        student, valid_loader, f"{method} initial valid"
    )
    best_score = validation_score(initial_metrics)
    best_state = cpu_state_dict(student)
    top_epoch_candidates = [
        {
            "epoch": 0,
            "fp_score": best_score,
            "state": best_state,
        }
    ]
    torch.save(best_state, checkpoint)
    stale_epochs = 0
    total_steps = max(1, epochs * len(run_train_loader) - 1)

    for epoch in range(epochs):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(
            run_train_loader,
            desc=f"{method} epoch {epoch + 1}/{epochs}",
        )
        for step, raw_batch in enumerate(progress_bar, 1):
            batch, sample_ids = split_batch(raw_batch)
            labels = batch["labels"]
            progress = (
                epoch * len(run_train_loader) + step - 1
            ) / total_steps

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = student(
                    **batch, output_hidden_states=use_proposed
                )
                use_second_forward = (
                    use_proposed and components["second_forward"]
                )
                second_outputs = (
                    student(**batch, output_hidden_states=True)
                    if use_second_forward
                    else outputs
                )
                if method == "Direct Training Control (No KD)":
                    total_loss = plain_supervised_loss(outputs.logits, labels)
                else:
                    use_progressive_probabilities = (
                        use_tiny_profile
                        and components["progressive"]
                        and "teacher_logits" in target_cache
                        and "assistant_logits" in target_cache
                    )
                    if use_progressive_probabilities:
                        teacher_logits = target_cache["teacher_logits"][
                            sample_ids
                        ].to(device)
                        assistant_logits = target_cache["assistant_logits"][
                            sample_ids
                        ].to(device)
                        # Every eligible tiny Proposed profile uses the same
                        # progressive teacher/assistant probability mixture.
                        profile_assistant_weight = cfg.assistant_logit_weight
                        target_logits = teacher_logits
                    else:
                        target_logits = target_cache["logits"][
                            sample_ids
                        ].to(device)
                    if use_tiny_profile:
                        # A one-layer model benefits from a stable objective.
                        # Pairwise auxiliary losses are deliberately excluded
                        # because gradient accumulation does not enlarge their
                        # physical batch of eight examples.
                        temperature = cfg.tiny_temperature
                        sample_weights = torch.ones(
                            labels.shape[0], device=device
                        )
                    elif use_proposed:
                        temperature = (
                            adaptive_temperature(progress)
                            if components["adaptive"]
                            else cfg.fixed_temperature
                        )
                        sample_weights = (
                            confidence_weights(target_logits, labels)
                            if components["confidence"]
                            else torch.ones(labels.shape[0], device=device)
                        )
                    else:
                        temperature = cfg.fixed_temperature
                        sample_weights = torch.ones(
                            labels.shape[0], device=device
                        )
                    if use_progressive_probabilities:
                        target_probabilities = (
                            (1.0 - profile_assistant_weight)
                            * F.softmax(
                                teacher_logits.float() / temperature,
                                dim=-1,
                            )
                            + profile_assistant_weight
                            * F.softmax(
                                assistant_logits.float() / temperature,
                                dim=-1,
                            )
                        )
                        kd_loss = probability_kd_loss(
                            outputs.logits,
                            target_probabilities,
                            temperature,
                            sample_weights,
                        )
                    else:
                        kd_loss = logit_kd_loss(
                            outputs.logits,
                            target_logits,
                            temperature,
                            sample_weights,
                            labels=labels,
                            label_correct_teacher=bool(
                                use_proposed
                                and not use_tiny_profile
                                and components["label_correction"]
                            ),
                        )

                    if method == "Standard Logit-KD Control":
                        supervised_loss = base_paper_supervised_loss(
                            outputs.logits, labels
                        )
                        total_loss = (
                            0.50 * supervised_loss + 0.50 * kd_loss
                        )
                    else:
                        ce_loss = proposed_supervised_loss(
                            outputs.logits, labels
                        )
                        if proposed_profile in {
                            "tiny_teacher_only",
                            "tiny_progressive",
                            "loo_no_balanced_supervision",
                        }:
                            ce_loss = plain_supervised_loss(
                                outputs.logits, labels
                            )
                        elif proposed_profile in {
                            "tiny_balanced",
                            "tiny_feature",
                        }:
                            ce_loss = (
                                0.80
                                * plain_supervised_loss(
                                    outputs.logits, labels
                                )
                                + 0.20 * weighted_ce(outputs.logits, labels)
                            )
                        target_features = target_cache["features"][
                            sample_ids
                        ].to(device)
                        if use_second_forward:
                            ce_loss = 0.5 * (
                                ce_loss
                                + proposed_supervised_loss(
                                    second_outputs.logits, labels
                                )
                            )
                            kd_loss = 0.5 * (
                                kd_loss
                                + logit_kd_loss(
                                    second_outputs.logits,
                                    target_logits,
                                    temperature,
                                    sample_weights,
                                    labels=labels,
                                    label_correct_teacher=components[
                                        "label_correction"
                                    ],
                                )
                            )
                            feature_loss = 0.5 * (
                                normalized_feature_loss(
                                    outputs.hidden_states, target_features
                                )
                                + normalized_feature_loss(
                                    second_outputs.hidden_states,
                                    target_features,
                                )
                            )
                            relation_loss = 0.5 * (
                                relation_kd_loss(
                                    outputs.hidden_states, target_features
                                )
                                + relation_kd_loss(
                                    second_outputs.hidden_states,
                                    target_features,
                                )
                            )
                            contrastive_loss = 0.5 * (
                                supervised_contrastive_loss(
                                    outputs.hidden_states[-1][:, 0], labels
                                )
                                + supervised_contrastive_loss(
                                    second_outputs.hidden_states[-1][:, 0],
                                    labels,
                                )
                            )
                            consistency_loss = symmetric_kl(
                                outputs.logits, second_outputs.logits
                            )
                        else:
                            zero_loss = torch.zeros(
                                (), device=device, dtype=torch.float32
                            )
                            if not components["feature"] or proposed_profile in {
                                "tiny_teacher_only",
                                "tiny_progressive",
                                "tiny_balanced",
                            }:
                                feature_loss = zero_loss
                            else:
                                feature_loss = normalized_feature_loss(
                                    outputs.hidden_states, target_features
                                )
                            relation_loss = zero_loss
                            contrastive_loss = zero_loss
                            consistency_loss = zero_loss

                        if proposed_profile in {
                            "tiny_teacher_only",
                            "tiny_progressive",
                        }:
                            loss_weights = {
                                "ce": 0.55,
                                "kd": 0.45,
                                "feature": 0.00,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile == "tiny_balanced":
                            loss_weights = {
                                "ce": 0.55,
                                "kd": 0.45,
                                "feature": 0.00,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile == "tiny_feature":
                            loss_weights = {
                                "ce": 0.52,
                                "kd": 0.45,
                                "feature": 0.03,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile in {
                            "ablation_balanced",
                            "ablation_adaptive",
                            "ablation_confidence",
                        }:
                            loss_weights = {
                                "ce": 0.55,
                                "kd": 0.45,
                                "feature": 0.00,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile in {
                            "ablation_feature",
                            "ablation_ema",
                        }:
                            loss_weights = {
                                "ce": 0.52,
                                "kd": 0.40,
                                "feature": 0.08,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile.startswith("loo_") and (
                            progress < 0.20
                            or proposed_profile == "loo_no_feature_kd"
                        ):
                            loss_weights = {
                                "ce": 0.62,
                                "kd": 0.38,
                                "feature": 0.00,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile.startswith("loo_"):
                            loss_weights = {
                                "ce": 0.54,
                                "kd": 0.38,
                                "feature": 0.08,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile == "core" and recovery_mode:
                            loss_weights = {
                                "ce": 0.70,
                                "kd": 0.25,
                                "feature": 0.05,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile == "core" and progress < 0.20:
                            loss_weights = {
                                "ce": 0.62,
                                "kd": 0.38,
                                "feature": 0.00,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif proposed_profile == "core":
                            loss_weights = {
                                "ce": 0.54,
                                "kd": 0.38,
                                "feature": 0.08,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        elif recovery_mode:
                            loss_weights = {
                                "ce": 0.70,
                                "kd": 0.22,
                                "feature": 0.04,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.04,
                            }
                        elif progress < 0.20:
                            # Establish the task decision boundary before
                            # introducing the smaller auxiliary objectives.
                            loss_weights = {
                                "ce": 0.60,
                                "kd": 0.40,
                                "feature": 0.00,
                                "relation": 0.00,
                                "contrastive": 0.00,
                                "consistency": 0.00,
                            }
                        else:
                            loss_weights = {
                                "ce": 0.52,
                                "kd": 0.35,
                                "feature": 0.08,
                                "relation": 0.01,
                                "contrastive": 0.01,
                                "consistency": 0.03,
                            }
                        total_loss = (
                            loss_weights["ce"] * ce_loss
                            + loss_weights["kd"] * kd_loss
                            + loss_weights["feature"] * feature_loss
                            + loss_weights["relation"] * relation_loss
                            + loss_weights["contrastive"]
                            * contrastive_loss
                            + loss_weights["consistency"]
                            * consistency_loss
                        )

                total_loss = total_loss / cfg.grad_accum_steps

            scaler.scale(total_loss).backward()
            if (
                step % cfg.grad_accum_steps == 0
                or step == len(run_train_loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(student)
            progress_bar.set_postfix(
                loss=float(total_loss.detach()) * cfg.grad_accum_steps
            )

        if ema is not None:
            ema.swap(student)
        metrics = evaluate(student, valid_loader, f"{method} valid")
        score = validation_score(metrics)
        top_k = max(1, cfg.deployment_epoch_candidates)
        qualifies_for_deployment_selection = (
            len(top_epoch_candidates) < top_k
            or score
            > min(
                candidate["fp_score"]
                for candidate in top_epoch_candidates
            )
        )
        evaluated_state = None
        if qualifies_for_deployment_selection:
            evaluated_state = cpu_state_dict(student)
            top_epoch_candidates.append(
                {
                    "epoch": epoch + 1,
                    "fp_score": score,
                    "state": evaluated_state,
                }
            )
            top_epoch_candidates = sorted(
                top_epoch_candidates,
                key=lambda candidate: candidate["fp_score"],
                reverse=True,
            )[:top_k]
        if score > best_score:
            best_score = score
            best_state = (
                evaluated_state
                if evaluated_state is not None
                else cpu_state_dict(student)
            )
            torch.save(best_state, checkpoint)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if ema is not None:
            ema.swap(student)
        if stale_epochs >= cfg.patience:
            break

    selected_epoch = None
    current_epoch_audit_rows = []
    if cfg.quantized_validation_selection and use_amp:
        deployment_candidates = []
        for epoch_candidate in top_epoch_candidates:
            candidate_model = new_student(
                num_layers, epoch_candidate["state"]
            ).cpu()
            candidate_name = (
                f"{checkpoint.stem}_epoch_{epoch_candidate['epoch']}"
            )
            selection = evaluate_validation_candidate(
                candidate_model,
                num_layers,
                candidate_name,
            )
            deployment_candidates.append(
                {
                    **epoch_candidate,
                    "selection": selection,
                }
            )
        selected_deployment_candidate = max(
            deployment_candidates,
            key=lambda candidate: candidate["selection"][
                "selection_score"
            ],
        )
        best_state = selected_deployment_candidate["state"]
        selected_epoch = selected_deployment_candidate["epoch"]
        for candidate in deployment_candidates:
            selection = candidate["selection"]
            current_epoch_audit_rows.append(
                {
                    "Checkpoint": checkpoint.name,
                    "Method": method,
                    "Profile": proposed_profile if use_proposed else "fixed",
                    "Layers": num_layers,
                    "Recovery Mode": recovery_mode,
                    "Learning Rate": learning_rate,
                    "Epoch": candidate["epoch"],
                    "FP Shortlist Score": candidate["fp_score"],
                    "Deployment Validation Score": selection[
                        "selection_score"
                    ],
                    "Selection Backend": selection["selection_backend"],
                    "Selected": candidate["epoch"] == selected_epoch,
                }
            )
        epoch_candidate_rows.extend(current_epoch_audit_rows)
        pd.DataFrame(current_epoch_audit_rows).to_csv(
            checkpoint.with_name(f"{checkpoint.name}.epoch_selection.csv"),
            index=False,
        )
        print(
            f"Selected {method} epoch {selected_epoch} using "
            f"{selected_deployment_candidate['selection']['selection_backend']} "
            "validation."
        )

    student.load_state_dict(best_state)
    student.cpu()
    torch.save(best_state, checkpoint)
    mark_checkpoint_complete(
        checkpoint,
        f"{method} training complete; layers={num_layers}; epochs={epochs}; "
        f"profile={proposed_profile}; selected_epoch={selected_epoch}",
    )
    cleanup()
    return student


def make_int8_quantization_config():
    skip_modules = list(cfg.int8_skip_modules) or None
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=cfg.int8_threshold,
        llm_int8_skip_modules=skip_modules,
    )


def save_hf_model(model, folder):
    previous_output_hidden_states = model.config.output_hidden_states
    model.config.output_hidden_states = False
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
    model.config.output_hidden_states = previous_output_hidden_states


def evaluate_validation_candidate(model, num_layers, candidate_name):
    """Score candidates in the precision used for deployment.

    Tiny candidates use INT8 validation only. Compact candidates keep one
    shared checkpoint for their FP32 and INT8 points, so their score is a
    predeclared weighted average of both validation scores.
    """
    model.to(device)
    fp_metrics = evaluate(
        model, valid_loader, f"{candidate_name} FP32 selection valid"
    )
    model.cpu()
    cleanup()

    int8_metrics = None
    int8_size_mb = np.nan
    int8_error = ""
    if cfg.quantized_validation_selection and use_amp:
        int8_model = None
        try:
            with tempfile.TemporaryDirectory(
                prefix="ubmec_selection_", dir=output_dir
            ) as temporary_folder:
                candidate_folder = Path(temporary_folder)
                save_hf_model(model, candidate_folder)
                int8_model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        candidate_folder,
                        quantization_config=make_int8_quantization_config(),
                        device_map={"": 0},
                        dtype=torch.float16,
                    )
                )
                int8_metrics = evaluate(
                    int8_model,
                    valid_loader,
                    f"{candidate_name} INT8 selection valid",
                )
                int8_size_mb = (
                    int8_model.get_memory_footprint() / 1_000_000
                )
                del int8_model
                int8_model = None
        except Exception as exc:
            int8_error = str(exc)
            raise RuntimeError(
                f"INT8 validation failed for required candidate "
                f"{candidate_name}; deployment selection cannot continue."
            ) from exc
        finally:
            if int8_model is not None:
                del int8_model
            cleanup()

    fp_score = validation_score(fp_metrics)
    if int8_metrics is None:
        selection_score = fp_score
        selection_backend = "FP32_ONLY"
    elif num_layers == cfg.tiny_layers:
        selection_score = validation_score(int8_metrics)
        selection_backend = cfg.internal_int8_backend
    else:
        int8_weight = cfg.compact_int8_selection_weight
        selection_score = (
            (1.0 - int8_weight) * fp_score
            + int8_weight * validation_score(int8_metrics)
        )
        selection_backend = "FP32_INT8_COMBINED"

    return {
        "fp_metrics": fp_metrics,
        "int8_metrics": int8_metrics,
        "int8_size_mb": int8_size_mb,
        "int8_error": int8_error,
        "selection_score": selection_score,
        "selection_backend": selection_backend,
    }


proposed_candidate_rows = []


def train_best_proposed(
    num_layers,
    initial_state,
    target_cache,
    checkpoint_stem,
):
    if locked_selection is not None:
        lock_key = (
            "tiny_proposed"
            if num_layers == cfg.tiny_layers
            else "compact_proposed"
        )
        best_profile = locked_selection["selected_profiles"][lock_key]
        checkpoint_record = locked_selection["checkpoints"][lock_key]
        selected_checkpoint = output_dir / checkpoint_record["filename"]
        if (
            not checkpoint_is_complete(selected_checkpoint)
            or sha256_file(selected_checkpoint)
            != checkpoint_record["sha256"]
        ):
            raise RuntimeError(
                f"Locked checkpoint verification failed: {selected_checkpoint}"
            )
        best_model = new_student(num_layers, initial_state)
        best_model.load_state_dict(load_state_dict(selected_checkpoint))
        print(
            f"Loaded locked Proposed Method profile for {num_layers} layers: "
            f"{best_profile}"
        )
        return best_model.cpu(), best_profile, selected_checkpoint

    candidates = []
    profiles = (
        cfg.tiny_proposed_profiles
        if num_layers == cfg.tiny_layers
        else cfg.proposed_profiles
    )
    for profile in profiles:
        checkpoint = output_dir / f"{checkpoint_stem}_{profile}.pt"
        model = train_student(
            method="Proposed Method",
            num_layers=num_layers,
            initial_state=initial_state,
            target_cache=target_cache,
            checkpoint=checkpoint,
            proposed_profile=profile,
        )
        selection = evaluate_validation_candidate(
            model,
            num_layers,
            f"{checkpoint_stem}_{profile}",
        )
        candidates.append(
            {
                "score": selection["selection_score"],
                "profile": profile,
                "model": model,
                "checkpoint": checkpoint,
                "selection": selection,
            }
        )

    best_candidate = max(candidates, key=lambda candidate: candidate["score"])
    best_score = best_candidate["score"]
    best_profile = best_candidate["profile"]
    best_model = best_candidate["model"]
    selected_checkpoint = output_dir / f"{checkpoint_stem}_selected.pt"
    torch.save(cpu_state_dict(best_model), selected_checkpoint)
    mark_checkpoint_complete(
        selected_checkpoint,
        f"Selected proposed profile={best_profile}; layers={num_layers}",
    )

    for candidate in candidates:
        profile = candidate["profile"]
        selection = candidate["selection"]
        fp_metrics = selection["fp_metrics"]
        int8_metrics = selection["int8_metrics"]
        proposed_candidate_rows.append(
            {
                "Layers": num_layers,
                "Profile": profile,
                "FP32 Validation Accuracy (%)": fp_metrics["accuracy"] * 100,
                "FP32 Validation Macro F1 (%)": fp_metrics["macro_f1"] * 100,
                "INT8 Validation Accuracy (%)": (
                    int8_metrics["accuracy"] * 100
                    if int8_metrics is not None
                    else np.nan
                ),
                "INT8 Validation Macro F1 (%)": (
                    int8_metrics["macro_f1"] * 100
                    if int8_metrics is not None
                    else np.nan
                ),
                "INT8 Validation Size (MB)": selection["int8_size_mb"],
                "Selection Backend": selection["selection_backend"],
                "Validation Selection Score": candidate["score"],
                "Selected": profile == best_profile,
                "INT8 Error": selection["int8_error"],
            }
        )
    print(
        f"Selected Proposed Method profile for {num_layers} layers: "
        f"{best_profile} (score={best_score:.4f})"
    )
    return best_model, best_profile, selected_checkpoint


def recover_proposed(
    student,
    num_layers,
    target_cache,
    checkpoint,
    proposed_profile,
    base_checkpoint,
):
    if locked_selection is not None:
        return student, base_checkpoint
    recovery_epochs = (
        cfg.tiny_recovery_epochs
        if num_layers == cfg.tiny_layers
        else cfg.recovery_epochs
    )
    if recovery_epochs <= 0:
        return student, base_checkpoint
    print(
        f"Running validation-selected recovery for {num_layers}-layer "
        "Proposed Method."
    )
    recovered = train_student(
        method="Proposed Method",
        num_layers=num_layers,
        initial_state=cpu_state_dict(student),
        target_cache=target_cache,
        checkpoint=checkpoint,
        epochs=recovery_epochs,
        learning_rate=cfg.recovery_lr,
        recovery_mode=True,
        proposed_profile=proposed_profile,
    )
    base_selection = evaluate_validation_candidate(
        student,
        num_layers,
        f"recovery_base_{num_layers}layer_{proposed_profile}",
    )
    recovered_selection = evaluate_validation_candidate(
        recovered,
        num_layers,
        f"recovery_candidate_{num_layers}layer_{proposed_profile}",
    )
    if (
        recovered_selection["selection_score"]
        > base_selection["selection_score"]
    ):
        print("Accepted recovered checkpoint using deployment validation.")
        return recovered, checkpoint
    print("Rejected recovered checkpoint; retained the pre-recovery model.")
    return student, base_checkpoint


compact_baseline_checkpoint = output_dir / "compact_baseline.pt"
compact_baseline = train_student(
    method="Direct Training Control (No KD)",
    num_layers=cfg.compact_layers,
    initial_state=compact_initial_state,
    target_cache=None,
    checkpoint=compact_baseline_checkpoint,
)
compact_standard_checkpoint = output_dir / "compact_base_paper_standard_kd.pt"
compact_base_paper = train_student(
    method="Standard Logit-KD Control",
    num_layers=cfg.compact_layers,
    initial_state=compact_initial_state,
    target_cache=teacher_compact_targets,
    checkpoint=compact_standard_checkpoint,
)
compact_proposed, compact_proposed_profile, compact_proposed_checkpoint = (
    train_best_proposed(
        num_layers=cfg.compact_layers,
        initial_state=compact_initial_state,
        target_cache=teacher_compact_targets,
        checkpoint_stem="compact_proposed",
    )
)
compact_recovery_checkpoint = output_dir / (
    f"compact_proposed_{compact_proposed_profile}_recovered.pt"
)
compact_proposed, compact_selected_checkpoint = recover_proposed(
    compact_proposed,
    cfg.compact_layers,
    teacher_compact_targets,
    compact_recovery_checkpoint,
    compact_proposed_profile,
    compact_proposed_checkpoint,
)

# Build a progressive assistant target for the tiny Proposed Method.
assistant_layer_ids = select_source_layers(
    compact_proposed.config.num_hidden_layers,
    cfg.tiny_layers,
)
assistant_checkpoint = compact_selected_checkpoint
assistant_tiny_targets = cache_targets(
    compact_proposed,
    assistant_layer_ids,
    assistant_checkpoint,
    output_dir / "assistant_targets_tiny.pt",
    "proposed-assistant-to-tiny",
)

# Keep teacher logits separate. They are converted to temperature-scaled
# probabilities and mixed inside train_student; raw logit scales are never
# averaged directly.
progressive_tiny_targets = {
    "logits": teacher_tiny_targets["logits"],
    "teacher_logits": teacher_tiny_targets["logits"],
    "assistant_logits": assistant_tiny_targets["logits"],
    "features": assistant_tiny_targets["features"],
}

tiny_baseline_checkpoint = output_dir / "tiny_baseline.pt"
tiny_baseline = train_student(
    method="Direct Training Control (No KD)",
    num_layers=cfg.tiny_layers,
    initial_state=tiny_initial_state,
    target_cache=None,
    checkpoint=tiny_baseline_checkpoint,
)
tiny_standard_checkpoint = output_dir / "tiny_base_paper_standard_kd.pt"
tiny_base_paper = train_student(
    method="Standard Logit-KD Control",
    num_layers=cfg.tiny_layers,
    initial_state=tiny_initial_state,
    target_cache=teacher_tiny_targets,
    checkpoint=tiny_standard_checkpoint,
)
tiny_proposed, tiny_proposed_profile, tiny_proposed_checkpoint = (
    train_best_proposed(
        num_layers=cfg.tiny_layers,
        initial_state=tiny_initial_state,
        target_cache=progressive_tiny_targets,
        checkpoint_stem="tiny_proposed_progressive",
    )
)
tiny_recovery_checkpoint = (
    output_dir / "tiny_proposed_progressive_recovered.pt"
)
tiny_proposed, tiny_selected_checkpoint = recover_proposed(
    tiny_proposed,
    cfg.tiny_layers,
    progressive_tiny_targets,
    tiny_recovery_checkpoint,
    tiny_proposed_profile,
    tiny_proposed_checkpoint,
)

ablation_models = {}
ablation_checkpoints = {}
ablation_validation_rows = []
if cfg.run_ablation_study:
    print("\nRunning compact cumulative component ablations")
    compact_ablation_sequence = [
        (
            "standard_logit_kd",
            "Standard Logit KD",
            None,
            compact_base_paper,
            compact_standard_checkpoint,
        )
    ]
    for profile in cfg.compact_ablation_profiles:
        checkpoint = output_dir / f"compact_{profile}.pt"
        model = train_student(
            method="Proposed Method",
            num_layers=cfg.compact_layers,
            initial_state=compact_initial_state,
            target_cache=teacher_compact_targets,
            checkpoint=checkpoint,
            proposed_profile=profile,
        )
        if profile.startswith("loo_"):
            display_name = "Full EADM (core) w/o " + profile.replace(
                "loo_no_", ""
            ).replace("_", " ")
        else:
            display_name = profile.replace("ablation_", "").replace(
                "_", " + "
            ).title()
        compact_ablation_sequence.append(
            (profile, display_name, profile, model, checkpoint)
        )

    compact_ablation_sequence.append(
        (
            "validation_selected_full_pipeline",
            "Validation-selected Proposed Pipeline",
            compact_proposed_profile,
            compact_proposed,
            compact_selected_checkpoint,
        )
    )
    for key, display_name, profile, model, checkpoint in compact_ablation_sequence:
        model.to(device)
        metrics = evaluate(
            model, valid_loader, f"Ablation {display_name} valid"
        )
        model.cpu()
        cleanup()
        ablation_validation_rows.append(
            {
                "Architecture": "5-layer compact student",
                "Ablation Key": key,
                "Ablation Step": display_name,
                "Profile": profile or "fixed",
                "Validation Accuracy (%)": metrics["accuracy"] * 100,
                "Validation Macro F1 (%)": metrics["macro_f1"] * 100,
            }
        )
        if key != "standard_logit_kd":
            lock_key = f"ablation_{key}"
            ablation_models[lock_key] = (
                display_name,
                "5-layer compact student",
                model,
            )
            ablation_checkpoints[lock_key] = checkpoint

    print("\nRunning tiny progressive-KD ablation")
    for profile, display_name in (
        ("tiny_teacher_only", "Teacher-only KD + EMA"),
        ("tiny_progressive", "Progressive Teacher/Assistant KD + EMA"),
    ):
        checkpoint = output_dir / f"tiny_ablation_{profile}.pt"
        model = train_student(
            method="Proposed Method",
            num_layers=cfg.tiny_layers,
            initial_state=tiny_initial_state,
            target_cache=progressive_tiny_targets,
            checkpoint=checkpoint,
            proposed_profile=profile,
        )
        model.to(device)
        metrics = evaluate(
            model, valid_loader, f"Ablation {display_name} valid"
        )
        model.cpu()
        cleanup()
        ablation_validation_rows.append(
            {
                "Architecture": "1-layer tiny student",
                "Ablation Key": profile,
                "Ablation Step": display_name,
                "Profile": profile,
                "Validation Accuracy (%)": metrics["accuracy"] * 100,
                "Validation Macro F1 (%)": metrics["macro_f1"] * 100,
            }
        )
        lock_key = f"ablation_{profile}"
        ablation_models[lock_key] = (
            display_name,
            "1-layer tiny student",
            model,
        )
        ablation_checkpoints[lock_key] = checkpoint

    ablation_validation = pd.DataFrame(ablation_validation_rows).round(4)
    ablation_validation.to_csv(
        output_dir / "ablation_validation_results.csv", index=False
    )
    print("\nAblation validation results")
    print(ablation_validation.to_string(index=False))

candidate_selection_path = (
    output_dir / "proposed_profile_validation_selection.csv"
)
if proposed_candidate_rows:
    candidate_selection = pd.DataFrame(proposed_candidate_rows)
    candidate_selection.to_csv(candidate_selection_path, index=False)
elif candidate_selection_path.exists():
    candidate_selection = pd.read_csv(candidate_selection_path)
else:
    candidate_selection = pd.DataFrame()
print("\nProposed profile validation selection")
if candidate_selection.empty:
    print("No proposed profile search was run in this phase.")
else:
    print(candidate_selection.round(4).to_string(index=False))

epoch_candidate_path = output_dir / "deployment_epoch_validation_selection.csv"
epoch_audit_frames = []
for sidecar in sorted(output_dir.glob("*.pt.epoch_selection.csv")):
    epoch_audit_frames.append(pd.read_csv(sidecar))
if epoch_candidate_rows:
    epoch_audit_frames.append(pd.DataFrame(epoch_candidate_rows))
if epoch_audit_frames:
    epoch_candidate_table = (
        pd.concat(epoch_audit_frames, ignore_index=True)
        .drop_duplicates(
            subset=["Checkpoint", "Epoch", "Selection Backend"],
            keep="last",
        )
        .sort_values(["Checkpoint", "Epoch"])
        .reset_index(drop=True)
    )
    epoch_candidate_table.to_csv(epoch_candidate_path, index=False)
elif epoch_candidate_path.exists():
    epoch_candidate_table = pd.read_csv(epoch_candidate_path)
else:
    epoch_candidate_table = pd.DataFrame()
print("\nDeployment selection among FP32-shortlisted epochs")
if epoch_candidate_table.empty:
    print("No epoch shortlist was created in this run mode.")
else:
    print(epoch_candidate_table.round(4).to_string(index=False))

# %% [markdown]
# ## 9. Structural audit before target-size deployment

# %%
architecture_rows = [
    {
        "Model": "Teacher BanglaBERT",
        "Layers": teacher.config.num_hidden_layers,
        "Parameters (M)": parameter_count_m(teacher),
        "Full Tensor Size (MB)": tensor_size_mb(teacher),
    },
    {
        "Model": "Compact Student",
        "Layers": compact_baseline.config.num_hidden_layers,
        "Parameters (M)": parameter_count_m(compact_baseline),
        "Full Tensor Size (MB)": tensor_size_mb(compact_baseline),
    },
    {
        "Model": "Tiny Student",
        "Layers": tiny_baseline.config.num_hidden_layers,
        "Parameters (M)": parameter_count_m(tiny_baseline),
        "Full Tensor Size (MB)": tensor_size_mb(tiny_baseline),
    },
]
architecture_table = pd.DataFrame(architecture_rows).round(2)
architecture_table.to_csv(
    output_dir / "architecture_audit.csv", index=False
)
print("\nArchitecture audit")
print(architecture_table.to_string(index=False))

comparison_control_rows = []
for architecture, layers, initialization_id in (
    (
        "5-layer compact student",
        cfg.compact_layers,
        "shared_compact_initial_state",
    ),
    (
        "1-layer tiny student",
        cfg.tiny_layers,
        "shared_tiny_initial_state",
    ),
):
    for approach in (
        "Direct Training Control (No KD)",
        "Standard Logit-KD Control",
        "Proposed Method",
    ):
        comparison_control_rows.append(
            {
                "Architecture": architecture,
                "Layers": layers,
                "Approach": approach,
                "Initialization ID": initialization_id,
                "Split Digest": split_digest,
                "Training Seed": cfg.seed,
                "Training Rows": len(train_df),
                "Validation Rows": len(valid_df),
                "Test Rows": len(test_df),
                "Epoch Budget": cfg.student_epochs,
                "Learning Rate": cfg.student_lr,
                "Batch Size": cfg.batch_size,
                "Gradient Accumulation": cfg.grad_accum_steps,
                "Optimizer": "AdamW",
                "Scheduler": "Linear with warmup",
                "Checkpoint Metric": "0.60 Accuracy + 0.40 Macro F1",
                "Test Used for Selection": False,
                "Proposed Profile Predeclared": (
                    len(
                        cfg.tiny_proposed_profiles
                        if layers == cfg.tiny_layers
                        else cfg.proposed_profiles
                    )
                    == 1
                ),
            }
        )

comparison_control_audit = pd.DataFrame(comparison_control_rows)
comparison_control_audit.to_csv(
    output_dir / "controlled_comparison_protocol_audit.csv", index=False
)

# %% [markdown]
# ## 10. Create final-size breakpoint variants

# %%
all_models = {
    "teacher": teacher.cpu(),
    "compact_baseline": compact_baseline.cpu(),
    "compact_base_paper": compact_base_paper.cpu(),
    "compact_proposed": compact_proposed.cpu(),
    "tiny_baseline": tiny_baseline.cpu(),
    "tiny_base_paper": tiny_base_paper.cpu(),
    "tiny_proposed": tiny_proposed.cpu(),
}
model_folders = {}
for model_key, model in all_models.items():
    folder = output_dir / "saved_models" / model_key
    save_hf_model(model, folder)
    model_folders[model_key] = folder


def load_deployment_variant(model, folder, internal_backend):
    if internal_backend == cfg.internal_full_backend:
        return deepcopy(model).float().to(device)
    if internal_backend != cfg.internal_int8_backend:
        raise ValueError(f"Unknown internal backend: {internal_backend}")
    if not use_amp:
        raise RuntimeError("The compressed breakpoint variants require CUDA.")

    return AutoModelForSequenceClassification.from_pretrained(
        folder,
        quantization_config=make_int8_quantization_config(),
        device_map={"": 0},
        dtype=torch.float16,
    )


def serialized_artifact_info(model, artifact_folder):
    """Persist model weights and measure disk size without tokenizer files."""
    artifact_folder.mkdir(parents=True, exist_ok=True)
    existing_weights = [
        path
        for path in artifact_folder.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".bin")
        )
        and not path.name.endswith(".index.json")
    ]
    if not existing_weights:
        previous_output_hidden_states = model.config.output_hidden_states
        model.config.output_hidden_states = False
        model.save_pretrained(artifact_folder, safe_serialization=True)
        model.config.output_hidden_states = previous_output_hidden_states

    weight_files = sorted(
        path
        for path in artifact_folder.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".bin")
        )
        and not path.name.endswith(".index.json")
    )
    if not weight_files:
        raise RuntimeError(
            f"No serialized weight file was produced in {artifact_folder}"
        )
    all_files = [path for path in artifact_folder.rglob("*") if path.is_file()]
    formats = sorted({path.suffix.lstrip(".") for path in weight_files})
    return {
        "File Format": "+".join(formats),
        "Weight Files": ";".join(path.name for path in weight_files),
        "Saved Weight File Size (MB)": sum(
            path.stat().st_size for path in weight_files
        )
        / 1_000_000,
        "Artifact Directory Size (MB)": sum(
            path.stat().st_size for path in all_files
        )
        / 1_000_000,
    }


def deployment_metadata(approach, source_model, internal_backend):
    quantized = internal_backend == cfg.internal_int8_backend
    if approach == "12-layer BanglaBERT Teacher":
        training_method = "Supervised teacher fine-tuning"
        kd_applied = False
    elif approach == "Direct Training Control (No KD)":
        training_method = "Direct supervised student training"
        kd_applied = False
    elif approach == "Standard Logit-KD Control":
        training_method = "Fixed-temperature standard logit KD"
        kd_applied = True
    else:
        training_method = "Validation-selected proposed KD pipeline"
        kd_applied = True
    layers = int(source_model.config.num_hidden_layers)
    return {
        "Architecture": f"BanglaBERT/ELECTRA encoder, {layers} layers",
        "Layers": layers,
        "Logical Parameters (M)": parameter_count_m(source_model),
        "Training Method": training_method,
        "KD Applied": kd_applied,
        "Quantization Applied": quantized,
        "Quantization Backend": (
            "bitsandbytes LLM.int8" if quantized else "None"
        ),
        "Weight Precision": (
            "INT8 linear weights; non-quantized tensors FP16"
            if quantized
            else "FP32"
        ),
        "Compute Precision": "FP16" if quantized else "FP32",
    }


def prediction_histogram(predictions):
    counts = np.bincount(predictions, minlength=len(cfg.labels))
    return "; ".join(
        f"{id2label[index]}={int(count)}"
        for index, count in enumerate(counts)
    )


def run_precision_parity_audit():
    if not cfg.run_precision_audit or not use_amp:
        return pd.DataFrame()

    rows = []
    audit_model_keys = (
        "compact_baseline",
        "compact_base_paper",
        "compact_proposed",
        "tiny_baseline",
        "tiny_base_paper",
        "tiny_proposed",
    )
    for model_key in audit_model_keys:
        source_model = all_models[model_key]
        folder = model_folders[model_key]
        reference_model = deepcopy(source_model).float().to(device)
        reference = evaluate(
            reference_model,
            valid_loader,
            f"{model_key} FP32 parity reference",
            return_outputs=True,
        )
        reference_logits = reference["_logits"]
        reference_predictions = reference["_predictions"]
        rows.append(
            {
                "Model Key": model_key,
                "Precision Variant": "FP32_IN_MEMORY",
                "Accuracy (%)": reference["accuracy"] * 100,
                "Macro F1 (%)": reference["macro_f1"] * 100,
                "Accuracy Drop vs FP32 (point)": 0.0,
                "Macro F1 Drop vs FP32 (point)": 0.0,
                "Top-1 Agreement vs FP32": 1.0,
                "Mean Absolute Logit Error": 0.0,
                "Mean KL vs FP32": 0.0,
                "Runtime Tensor Footprint (MB)": (
                    reference_model.get_memory_footprint() / 1_000_000
                ),
                "Prediction Histogram": prediction_histogram(
                    reference_predictions
                ),
                "Deployment Valid": True,
                "Error": "",
            }
        )
        del reference_model
        cleanup()

        variant_loaders = {
            "FP32_RELOAD": lambda: (
                AutoModelForSequenceClassification.from_pretrained(folder)
                .float()
                .to(device)
            ),
            "FP16_RELOAD": lambda: (
                AutoModelForSequenceClassification.from_pretrained(
                    folder, dtype=torch.float16
                ).to(device)
            ),
            cfg.internal_int8_backend: lambda: load_deployment_variant(
                source_model, folder, cfg.internal_int8_backend
            ),
        }
        for variant_name, loader_function in variant_loaders.items():
            variant = None
            try:
                variant = loader_function()
                metrics = evaluate(
                    variant,
                    valid_loader,
                    f"{model_key} {variant_name} parity",
                    return_outputs=True,
                )
                logits = metrics["_logits"]
                predictions = metrics["_predictions"]
                accuracy_drop = (
                    reference["accuracy"] - metrics["accuracy"]
                ) * 100
                macro_drop = (
                    reference["macro_f1"] - metrics["macro_f1"]
                ) * 100
                deployment_valid = True
                if variant_name == cfg.internal_int8_backend:
                    deployment_valid = (
                        accuracy_drop
                        <= cfg.quantization_drop_warning_points
                        and macro_drop
                        <= cfg.quantization_drop_warning_points
                        and float(
                            np.mean(predictions == reference_predictions)
                        )
                        >= cfg.quantization_min_top1_agreement
                    )
                rows.append(
                    {
                        "Model Key": model_key,
                        "Precision Variant": variant_name,
                        "Accuracy (%)": metrics["accuracy"] * 100,
                        "Macro F1 (%)": metrics["macro_f1"] * 100,
                        "Accuracy Drop vs FP32 (point)": accuracy_drop,
                        "Macro F1 Drop vs FP32 (point)": macro_drop,
                        "Top-1 Agreement vs FP32": float(
                            np.mean(predictions == reference_predictions)
                        ),
                        "Mean Absolute Logit Error": float(
                            (logits - reference_logits).abs().mean()
                        ),
                        "Mean KL vs FP32": float(
                            F.kl_div(
                                F.log_softmax(logits, dim=-1),
                                F.softmax(reference_logits, dim=-1),
                                reduction="batchmean",
                            )
                        ),
                        "Runtime Tensor Footprint (MB)": (
                            variant.get_memory_footprint() / 1_000_000
                        ),
                        "Prediction Histogram": prediction_histogram(
                            predictions
                        ),
                        "Deployment Valid": deployment_valid,
                        "Error": "",
                    }
                )
                if (
                    variant_name == cfg.internal_int8_backend
                    and accuracy_drop
                    > cfg.quantization_drop_warning_points
                ):
                    warnings.warn(
                        f"{model_key} loses {accuracy_drop:.2f} accuracy "
                        "points after INT8. Treat its breakpoint result as "
                        "quantization-limited, not a clean KD comparison."
                    )
            except Exception as exc:
                rows.append(
                    {
                        "Model Key": model_key,
                        "Precision Variant": variant_name,
                        "Deployment Valid": False,
                        "Error": str(exc),
                    }
                )
                warnings.warn(
                    f"Parity audit failed for {model_key}/{variant_name}: {exc}"
                )
            finally:
                if variant is not None:
                    del variant
                cleanup()

    audit = pd.DataFrame(rows)
    numeric_columns = audit.select_dtypes(include=[np.number]).columns
    audit[numeric_columns] = audit[numeric_columns].round(4)
    audit.to_csv(output_dir / "precision_parity_validation.csv", index=False)
    print("\nFP32/FP16/INT8 validation parity audit")
    print(audit.to_string(index=False))
    return audit


if (
    cfg.run_mode == "live"
    and cfg.block_final_test_on_quantization_collapse
    and not cfg.run_precision_audit
):
    raise RuntimeError(
        "Live mode cannot disable run_precision_audit while the quantization "
        "collapse gate is enabled."
    )

precision_parity_audit = run_precision_parity_audit()
if cfg.run_mode == "live" and cfg.block_final_test_on_quantization_collapse:
    expected_int8_models = {
        "compact_baseline",
        "compact_base_paper",
        "compact_proposed",
        "tiny_baseline",
        "tiny_base_paper",
        "tiny_proposed",
    }
    if precision_parity_audit.empty:
        raise RuntimeError(
            "The required live INT8 parity audit is empty; final evaluation "
            "is blocked."
        )
    int8_audit_rows = precision_parity_audit[
        precision_parity_audit["Precision Variant"]
        == cfg.internal_int8_backend
    ]
    observed_int8_models = set(int8_audit_rows["Model Key"].dropna())
    missing_int8_models = expected_int8_models - observed_int8_models
    duplicate_int8_models = int8_audit_rows["Model Key"].duplicated().any()
    invalid_int8 = int8_audit_rows[
        ~int8_audit_rows["Deployment Valid"].fillna(False).astype(bool)
    ]
    if missing_int8_models or duplicate_int8_models or not invalid_int8.empty:
        raise RuntimeError(
            "INT8 validation parity is incomplete or failed for one or more "
            "required models. Final test and winner plots are intentionally "
            "blocked; inspect precision_parity_validation.csv. "
            f"Missing={sorted(missing_int8_models)}; "
            f"duplicate_rows={bool(duplicate_int8_models)}."
        )


def locked_checkpoint_record(path):
    if not checkpoint_is_complete(path):
        raise RuntimeError(f"Cannot lock incomplete checkpoint: {path}")
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
    }


checkpoint_paths_to_lock = {
    "teacher": teacher_checkpoint,
    "compact_baseline": compact_baseline_checkpoint,
    "compact_standard_kd": compact_standard_checkpoint,
    "compact_proposed": compact_selected_checkpoint,
    "tiny_baseline": tiny_baseline_checkpoint,
    "tiny_standard_kd": tiny_standard_checkpoint,
    "tiny_proposed": tiny_selected_checkpoint,
}
checkpoint_paths_to_lock.update(ablation_checkpoints)
current_selection_manifest = {
    "schema_version": 1,
    "experiment_version": cfg.experiment_version,
    "config_digest": config_digest,
    "split_digest": split_digest,
    "evidence_status": evidence_status,
    "protocol_source_digest": protocol_digest,
    "runtime_assets": {
        "digest": runtime_asset_digest,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": installed_package_version("transformers"),
        "bitsandbytes": installed_package_version("bitsandbytes"),
        "accelerate": installed_package_version("accelerate"),
        "teacher_name": cfg.teacher_name,
    },
    "selected_profiles": {
        "compact_proposed": compact_proposed_profile,
        "tiny_proposed": tiny_proposed_profile,
    },
    "selection_strategy": {
        "epoch_shortlist": (
            f"top_{cfg.deployment_epoch_candidates}_by_fp32_validation"
        ),
        "compact_backend_score": "weighted_fp32_and_int8_validation",
        "compact_int8_weight": cfg.compact_int8_selection_weight,
        "tiny_backend_score": "int8_validation",
    },
    "quantization": {
        "backend": cfg.internal_int8_backend,
        "threshold": cfg.int8_threshold,
        "skip_modules": list(cfg.int8_skip_modules),
        "maximum_allowed_accuracy_or_f1_drop_points": (
            cfg.quantization_drop_warning_points
        ),
        "minimum_top1_agreement": cfg.quantization_min_top1_agreement,
    },
    "checkpoints": {
        key: locked_checkpoint_record(path)
        for key, path in checkpoint_paths_to_lock.items()
    },
}
selection_manifest_digest = hashlib.sha256(
    json.dumps(
        current_selection_manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

if not cfg.run_final_test:
    if cfg.run_mode == "live":
        if run_status_path.exists():
            previous_status = json.loads(
                run_status_path.read_text(encoding="utf-8")
            )
            if (
                previous_status.get("selection_manifest_digest")
                == selection_manifest_digest
                and previous_status.get("phase")
                in {"final_test_running", "final_test_complete"}
            ):
                raise RuntimeError(
                    "This exact locked holdout evaluation has already started "
                    "or completed; stage one cannot silently reopen it."
                )
        selection_lock_path.write_text(
            json.dumps(current_selection_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run_status_path.write_text(
            json.dumps(
                {
                    "phase": "validation_locked",
                    "selection_manifest_digest": selection_manifest_digest,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    print(
        "\nValidation selection and precision audit are complete. "
        "Review the generated CSV files, then set run_final_test=True and "
        "rerun to unlock the final test phase."
    )
    if cfg.run_mode == "live":
        print(f"Selection lock: {selection_lock_path}")
    raise SystemExit(0)

if cfg.run_mode == "live":
    if locked_selection != current_selection_manifest:
        raise RuntimeError(
            "Current profiles/checkpoint hashes/protocol do not exactly match "
            "the stage-one validation lock. Final test remains blocked."
        )
    if run_status_path.exists():
        previous_status = json.loads(run_status_path.read_text(encoding="utf-8"))
        if (
            previous_status.get("selection_manifest_digest")
            == selection_manifest_digest
            and previous_status.get("phase")
            in {"final_test_running", "final_test_complete"}
        ):
            raise RuntimeError(
                "This locked final test has already started or completed. "
                "Do not repeatedly inspect the same holdout."
            )
    holdout_registry = {"schema_version": 1, "splits": {}}
    if holdout_registry_path.exists():
        holdout_registry = json.loads(
            holdout_registry_path.read_text(encoding="utf-8")
        )
    # A registered multi-seed protocol deliberately evaluates several locked
    # training runs on one fixed holdout. Prevent repeat access per seed while
    # allowing the declared seeds to share that fixed comparison split.
    holdout_run_key = f"{split_digest}:training_seed={cfg.seed}"
    existing_holdout_record = holdout_registry.setdefault(
        "splits", {}
    ).get(holdout_run_key)
    if existing_holdout_record is not None:
        raise RuntimeError(
            "This exact holdout split has already been opened by another "
            "live configuration. Use a genuinely fresh confirmation split "
            "or nested/repeated outer evaluation."
        )
    holdout_registry["splits"][holdout_run_key] = {
        "phase": "final_test_running",
        "split_digest": split_digest,
        "training_seed": cfg.seed,
        "experiment_version": cfg.experiment_version,
        "config_digest": config_digest,
        "protocol_digest": protocol_digest,
        "selection_manifest_digest": selection_manifest_digest,
    }
    holdout_registry_path.write_text(
        json.dumps(holdout_registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    run_status_path.write_text(
        json.dumps(
            {
                "phase": "final_test_running",
                "selection_manifest_digest": selection_manifest_digest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if cfg.run_ablation_study:
    ablation_test_models = {
        "compact_direct_training": (
            "Direct Training (No KD)",
            "5-layer compact student",
            compact_baseline,
        ),
        "compact_standard_logit_kd": (
            "Standard Logit KD",
            "5-layer compact student",
            compact_base_paper,
        ),
        "tiny_direct_training": (
            "Direct Training (No KD)",
            "1-layer tiny student",
            tiny_baseline,
        ),
        "tiny_standard_logit_kd": (
            "Standard Logit KD",
            "1-layer tiny student",
            tiny_base_paper,
        ),
        **ablation_models,
    }
    ablation_test_rows = []
    for key, (display_name, architecture, model) in ablation_test_models.items():
        model.to(device)
        metrics = evaluate(model, test_loader, f"Ablation {key} test")
        model.cpu()
        cleanup()
        ablation_test_rows.append(
            {
                "Training Seed": cfg.seed,
                "Split Seed": cfg.split_seed,
                "Architecture": architecture,
                "Ablation Key": key,
                "Ablation Step": display_name,
                "Accuracy (%)": metrics["accuracy"] * 100,
                "Macro F1 (%)": metrics["macro_f1"] * 100,
                "Evidence Status": evidence_status,
            }
        )
    ablation_test_results = pd.DataFrame(ablation_test_rows).round(4)
    ablation_test_results.to_csv(
        output_dir / "ablation_test_results.csv", index=False
    )
    print("\nAblation test results")
    print(ablation_test_results.to_string(index=False))


# The internal backend is audit metadata only. It is never used as a graph
# category or point label.
deployment_plan = [
    {
        "Approach": "12-layer BanglaBERT Teacher",
        "Target Size (MB)": 400,
        "Model Key": "teacher",
        "Internal Backend": cfg.internal_full_backend,
    },
    {
        "Approach": "Direct Training Control (No KD)",
        "Target Size (MB)": 200,
        "Model Key": "compact_baseline",
        "Internal Backend": cfg.internal_full_backend,
    },
    {
        "Approach": "Direct Training Control (No KD)",
        "Target Size (MB)": 100,
        "Model Key": "compact_baseline",
        "Internal Backend": cfg.internal_int8_backend,
    },
    {
        "Approach": "Direct Training Control (No KD)",
        "Target Size (MB)": 50,
        "Model Key": "tiny_baseline",
        "Internal Backend": cfg.internal_int8_backend,
    },
    {
        "Approach": "Standard Logit-KD Control",
        "Target Size (MB)": 200,
        "Model Key": "compact_base_paper",
        "Internal Backend": cfg.internal_full_backend,
    },
    {
        "Approach": "Standard Logit-KD Control",
        "Target Size (MB)": 100,
        "Model Key": "compact_base_paper",
        "Internal Backend": cfg.internal_int8_backend,
    },
    {
        "Approach": "Standard Logit-KD Control",
        "Target Size (MB)": 50,
        "Model Key": "tiny_base_paper",
        "Internal Backend": cfg.internal_int8_backend,
    },
    {
        "Approach": "Proposed Method",
        "Target Size (MB)": 200,
        "Model Key": "compact_proposed",
        "Internal Backend": cfg.internal_full_backend,
    },
    {
        "Approach": "Proposed Method",
        "Target Size (MB)": 100,
        "Model Key": "compact_proposed",
        "Internal Backend": cfg.internal_int8_backend,
    },
    {
        "Approach": "Proposed Method",
        "Target Size (MB)": 50,
        "Model Key": "tiny_proposed",
        "Internal Backend": cfg.internal_int8_backend,
    },
]

deployment_rows, deployment_failures = [], []
for experiment in deployment_plan:
    approach = experiment["Approach"]
    target_size = experiment["Target Size (MB)"]
    model_key = experiment["Model Key"]
    internal_backend = experiment["Internal Backend"]
    source_model = all_models[model_key]
    folder = model_folders[model_key]
    print(f"\nEvaluating {approach} near {target_size} MB")

    variant = None
    try:
        variant = load_deployment_variant(
            source_model, folder, internal_backend
        )
        metrics = evaluate(
            variant,
            test_loader,
            f"{approach} near {target_size} MB",
            benchmark=True,
        )
        runtime_footprint_mb = variant.get_memory_footprint() / 1_000_000
        artifact_folder = (
            output_dir
            / "deployment_artifacts"
            / f"{model_key}_{target_size}mb"
        )
        artifact_info = serialized_artifact_info(variant, artifact_folder)
        saved_weight_size_mb = artifact_info["Saved Weight File Size (MB)"]
        size_error_percent = (
            abs(saved_weight_size_mb - target_size) / target_size * 100
        )
        deployment_rows.append(
            {
                "Approach": approach,
                "Comparison Range (MB)": target_size,
                **deployment_metadata(
                    approach, source_model, internal_backend
                ),
                **artifact_info,
                "Runtime Tensor Footprint (MB)": runtime_footprint_mb,
                "Accuracy (%)": metrics["accuracy"] * 100,
                "Macro F1 (%)": metrics["macro_f1"] * 100,
                "Distance from Range Center (%)": size_error_percent,
                "Latency (ms/sample)": metrics["latency_ms_sample"],
                "Peak GPU Memory (MB)": metrics["peak_gpu_mb"],
                "Model Key": model_key,
                "Internal Backend": internal_backend,
                "Evidence Status": evidence_status,
            }
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        deployment_failures.append(
            {
                "Approach": approach,
                "Target Size (MB)": target_size,
                "Model Key": model_key,
                "Internal Backend": internal_backend,
                "Error": str(exc),
            }
        )
    finally:
        if variant is not None:
            del variant
        cleanup()

deployment_failure_path = output_dir / "deployment_failures.csv"
if deployment_failures:
    pd.DataFrame(deployment_failures).to_csv(
        deployment_failure_path, index=False
    )
    raise RuntimeError(
        f"{len(deployment_failures)} required deployment evaluation(s) "
        "failed. Refusing to create incomplete winner tables; see "
        "deployment_failures.csv."
    )
elif deployment_failure_path.exists():
    deployment_failure_path.unlink()

deployment_audit = pd.DataFrame(deployment_rows)
numeric_columns = deployment_audit.select_dtypes(include=[np.number]).columns
deployment_audit[numeric_columns] = deployment_audit[numeric_columns].round(2)
deployment_audit.to_csv(
    output_dir / "deployment_audit_internal.csv", index=False
)

artifact_audit_columns = [
    "Approach",
    "Comparison Range (MB)",
    "Model Key",
    "Architecture",
    "Layers",
    "Logical Parameters (M)",
    "Training Method",
    "KD Applied",
    "Quantization Applied",
    "Quantization Backend",
    "Weight Precision",
    "Compute Precision",
    "File Format",
    "Weight Files",
    "Saved Weight File Size (MB)",
    "Artifact Directory Size (MB)",
    "Runtime Tensor Footprint (MB)",
]
deployment_audit[artifact_audit_columns].to_csv(
    output_dir / "model_artifact_audit.csv", index=False
)

presentation_columns = [
    "Approach",
    "Comparison Range (MB)",
    "Architecture",
    "Layers",
    "Logical Parameters (M)",
    "KD Applied",
    "Quantization Applied",
    "Weight Precision",
    "Compute Precision",
    "File Format",
    "Saved Weight File Size (MB)",
    "Runtime Tensor Footprint (MB)",
    "Accuracy (%)",
    "Macro F1 (%)",
    "Distance from Range Center (%)",
    "Evidence Status",
]
presentation_results = deployment_audit[presentation_columns].copy()
presentation_results.to_csv(
    output_dir / "fixed_size_breakpoint_results.csv", index=False
)
print("\nFixed-size breakpoint results")
print(presentation_results.to_string(index=False))

# Separate the three mechanisms instead of treating every size change as KD.
# KD rows compare objectives at the same architecture/deployment setting;
# quantization rows compare the same trained checkpoint before and after the
# common deployment transform. Structural rows are descriptive because layer
# reduction and subsequent retraining cannot be isolated by these final rows.
effect_rows = []
for target_name, target_layers in (
    ("5-layer compact student", cfg.compact_layers),
    ("1-layer tiny student", cfg.tiny_layers),
):
    effect_rows.append(
        {
            "Effect Type": "Structural reduction definition",
            "Approach": target_name,
            "Comparison Range (MB)": np.nan,
            "From": f"12-layer teacher ({teacher.config.num_hidden_layers} layers)",
            "To": f"{target_name} ({target_layers} layers)",
            "Accuracy Change (point)": np.nan,
            "Macro F1 Change (point)": np.nan,
            "Interpretation": (
                "Defines the architecture change only; final metric "
                "differences also include retraining and are not labeled as "
                "a pure structural causal effect."
            ),
        }
    )
direct_rows = presentation_results[
    presentation_results["Approach"]
    == "Direct Training Control (No KD)"
]
for approach in ("Standard Logit-KD Control", "Proposed Method"):
    method_rows = presentation_results[
        presentation_results["Approach"] == approach
    ]
    matched = direct_rows.merge(
        method_rows,
        on="Comparison Range (MB)",
        suffixes=(" Direct", " Method"),
    )
    for _, row in matched.iterrows():
        effect_rows.append(
            {
                "Effect Type": "Training objective / KD effect",
                "Approach": approach,
                "Comparison Range (MB)": row["Comparison Range (MB)"],
                "From": "Direct training",
                "To": approach,
                "Accuracy Change (point)": (
                    row["Accuracy (%) Method"]
                    - row["Accuracy (%) Direct"]
                ),
                "Macro F1 Change (point)": (
                    row["Macro F1 (%) Method"]
                    - row["Macro F1 (%) Direct"]
                ),
                "Interpretation": (
                    "Matched architecture, initialization, split, and "
                    "deployment setting; training objective differs."
                ),
            }
        )

for approach in presentation_results["Approach"].unique():
    rows = presentation_results[
        presentation_results["Approach"] == approach
    ]
    full_row = rows[rows["Comparison Range (MB)"] == 200]
    compressed_row = rows[rows["Comparison Range (MB)"] == 100]
    if full_row.empty or compressed_row.empty:
        continue
    full_row = full_row.iloc[0]
    compressed_row = compressed_row.iloc[0]
    effect_rows.append(
        {
            "Effect Type": "Post-training quantization effect",
            "Approach": approach,
            "Comparison Range (MB)": 100,
            "From": (
                f"{full_row['Saved Weight File Size (MB)']:.2f} MB weights"
            ),
            "To": (
                f"{compressed_row['Saved Weight File Size (MB)']:.2f} MB "
                "weights"
            ),
            "Accuracy Change (point)": (
                compressed_row["Accuracy (%)"] - full_row["Accuracy (%)"]
            ),
            "Macro F1 Change (point)": (
                compressed_row["Macro F1 (%)"] - full_row["Macro F1 (%)"]
            ),
            "Interpretation": (
                "Same trained 5-layer checkpoint; only deployment "
                "quantization differs."
            ),
        }
    )

effect_decomposition = pd.DataFrame(effect_rows).round(4)
effect_decomposition.to_csv(
    output_dir / "kd_and_quantization_effect_decomposition.csv",
    index=False,
)

outside_tolerance = presentation_results[
    presentation_results["Distance from Range Center (%)"]
    > cfg.target_tolerance_fraction * 100
]
if not outside_tolerance.empty:
    print(
        "\nWARNING: Some final sizes are outside the configured target "
        "tolerance. They remain plotted at their measured size:"
    )
    print(outside_tolerance.to_string(index=False))

# %% [markdown]
# ## 11. Breakpoint winners and Proposed-vs-Standard-KD status

# %%
winner_rows = []
for target_size, target_rows in presentation_results.groupby(
    "Comparison Range (MB)"
):
    accuracy_winner = target_rows.sort_values(
        ["Accuracy (%)", "Macro F1 (%)", "Saved Weight File Size (MB)"],
        ascending=[False, False, True],
    ).iloc[0]
    macro_winner = target_rows.sort_values(
        ["Macro F1 (%)", "Accuracy (%)", "Saved Weight File Size (MB)"],
        ascending=[False, False, True],
    ).iloc[0]
    winner_rows.append(
        {
            "Comparison Range (MB)": target_size,
            "Best Accuracy Approach": accuracy_winner["Approach"],
            "Best Accuracy (%)": accuracy_winner["Accuracy (%)"],
            "Accuracy Winner Size (MB)": accuracy_winner[
                "Saved Weight File Size (MB)"
            ],
            "Best Macro F1 Approach": macro_winner["Approach"],
            "Best Macro F1 (%)": macro_winner["Macro F1 (%)"],
            "Macro F1 Winner Size (MB)": macro_winner[
                "Saved Weight File Size (MB)"
            ],
        }
    )

breakpoint_winners = pd.DataFrame(winner_rows).sort_values(
    "Comparison Range (MB)", ascending=False
)
breakpoint_winners.to_csv(
    output_dir / "best_method_at_each_breakpoint.csv", index=False
)
print("\nBest method at each target size")
print(breakpoint_winners.to_string(index=False))

base_rows = presentation_results[
    presentation_results["Approach"] == "Standard Logit-KD Control"
].copy()
proposed_rows = presentation_results[
    presentation_results["Approach"] == "Proposed Method"
].copy()
proposed_vs_base = base_rows.merge(
    proposed_rows,
    on="Comparison Range (MB)",
    suffixes=(" Standard KD", " Proposed"),
)
if not proposed_vs_base.empty:
    proposed_vs_base["Accuracy Gain (point)"] = (
        proposed_vs_base["Accuracy (%) Proposed"]
        - proposed_vs_base["Accuracy (%) Standard KD"]
    )
    proposed_vs_base["Macro F1 Gain (point)"] = (
        proposed_vs_base["Macro F1 (%) Proposed"]
        - proposed_vs_base["Macro F1 (%) Standard KD"]
    )
    proposed_vs_base["Proposed Numerically Higher: Accuracy"] = (
        proposed_vs_base["Accuracy Gain (point)"] > 0
    )
    proposed_vs_base["Proposed Numerically Higher: Macro F1"] = (
        proposed_vs_base["Macro F1 Gain (point)"] > 0
    )
    proposed_vs_base = proposed_vs_base.round(2)
    proposed_vs_base.to_csv(
        output_dir / "proposed_vs_standard_kd_at_shared_sizes.csv",
        index=False,
    )
    print(
        "\nProposed Method versus Standard Logit-KD Control "
        "at shared size targets"
    )
    print(proposed_vs_base.to_string(index=False))

teacher_anchor_rows = presentation_results[
    (presentation_results["Approach"] == "12-layer BanglaBERT Teacher")
    & (presentation_results["Comparison Range (MB)"] == 400)
]
all_proposed_rows = presentation_results[
    presentation_results["Approach"] == "Proposed Method"
]
if not teacher_anchor_rows.empty and not all_proposed_rows.empty:
    teacher_anchor = teacher_anchor_rows.iloc[0]
    best_proposed_accuracy = all_proposed_rows.loc[
        all_proposed_rows["Accuracy (%)"].idxmax()
    ]
    best_proposed_f1 = all_proposed_rows.loc[
        all_proposed_rows["Macro F1 (%)"].idxmax()
    ]
    proposed_vs_teacher = pd.DataFrame(
        [
            {
                "Teacher Size (MB)": teacher_anchor["Saved Weight File Size (MB)"],
                "Teacher Accuracy (%)": teacher_anchor["Accuracy (%)"],
                "Best Proposed Accuracy Size (MB)": best_proposed_accuracy[
                    "Saved Weight File Size (MB)"
                ],
                "Best Proposed Accuracy (%)": best_proposed_accuracy[
                    "Accuracy (%)"
                ],
                "Accuracy Gain vs Teacher (point)": best_proposed_accuracy[
                    "Accuracy (%)"
                ]
                - teacher_anchor["Accuracy (%)"],
                "Proposed Numerically Higher Than Teacher Accuracy": best_proposed_accuracy[
                    "Accuracy (%)"
                ]
                > teacher_anchor["Accuracy (%)"],
                "Teacher Macro F1 (%)": teacher_anchor["Macro F1 (%)"],
                "Best Proposed Macro F1 Size (MB)": best_proposed_f1[
                    "Saved Weight File Size (MB)"
                ],
                "Best Proposed Macro F1 (%)": best_proposed_f1[
                    "Macro F1 (%)"
                ],
                "Macro F1 Gain vs Teacher (point)": best_proposed_f1[
                    "Macro F1 (%)"
                ]
                - teacher_anchor["Macro F1 (%)"],
                "Proposed Numerically Higher Than Teacher Macro F1": best_proposed_f1[
                    "Macro F1 (%)"
                ]
                > teacher_anchor["Macro F1 (%)"],
            }
        ]
    ).round(2)
    proposed_vs_teacher.to_csv(
        output_dir / "proposed_vs_teacher_summary.csv", index=False
    )
    print("\nBest Proposed Method versus BanglaBERT teacher anchor")
    print(proposed_vs_teacher.to_string(index=False))

method_definition = pd.DataFrame(
    [
        {
            "Approach": "Standard Logit-KD Control",
            "Deployment": (
                "Fixed objective; deployment-selected epoch for 5/1 layers"
            ),
            "Selected Profile": "fixed",
            "Supervised Loss": "One-hot BCE",
            "Logit KD": True,
            "Temperature": f"Fixed T={cfg.fixed_temperature}",
            "Class-Balanced Loss": False,
            "Label Smoothing": False,
            "Confidence Weighting": False,
            "Label-Aware Teacher Reliability": False,
            "Wrong-Teacher Target Correction": False,
            "Feature KD": False,
            "Relation KD": False,
            "Contrastive Loss": False,
            "R-Drop": False,
            "EMA": False,
            "Progressive KD": False,
        },
        {
            "Approach": "Proposed Method (5-layer)",
            "Deployment": "Approximately 200/100 MB",
            "Selected Profile": compact_proposed_profile,
            "Supervised Loss": "Plain CE + class-balanced smoothed CE",
            "Logit KD": True,
            "Temperature": "Adaptive",
            "Class-Balanced Loss": True,
            "Label Smoothing": True,
            "Confidence Weighting": True,
            "Label-Aware Teacher Reliability": True,
            "Wrong-Teacher Target Correction": True,
            "Feature KD": True,
            "Relation KD": "Full profile only",
            "Contrastive Loss": "Full profile only",
            "R-Drop": "Full profile only",
            "EMA": True,
            "Progressive KD": False,
        },
        {
            "Approach": "Proposed Method (1-layer)",
            "Deployment": "Approximately 50 MB",
            "Selected Profile": tiny_proposed_profile,
            "Supervised Loss": (
                "Plain CE"
                if tiny_proposed_profile == "tiny_progressive"
                else "80% plain CE + 20% class-balanced smoothed CE"
            ),
            "Logit KD": True,
            "Temperature": f"Fixed T={cfg.tiny_temperature}",
            "Class-Balanced Loss": (
                tiny_proposed_profile != "tiny_progressive"
            ),
            "Label Smoothing": tiny_proposed_profile != "tiny_progressive",
            "Confidence Weighting": False,
            "Label-Aware Teacher Reliability": False,
            "Wrong-Teacher Target Correction": False,
            "Feature KD": tiny_proposed_profile == "tiny_feature",
            "Relation KD": False,
            "Contrastive Loss": False,
            "R-Drop": False,
            "EMA": True,
            "Progressive KD": True,
        },
    ]
)
method_definition.to_csv(
    output_dir / "standard_kd_vs_proposed_technique_audit.csv",
    index=False,
)

# These rows are copied as fixed external references from the paper. They
# are never altered, retrained, interpolated, or used as training targets.
paper_reference = pd.DataFrame(
    [
        {
            "Published Configuration": "BanglaBERT Baseline",
            "Reported Size (MB)": 420.0,
            "Accuracy (%)": 62.36,
            "Macro F1 (%)": 60.06,
        },
        {
            "Published Configuration": "Pruned BanglaBERT",
            "Reported Size (MB)": 324.0,
            "Accuracy (%)": 61.42,
            "Macro F1 (%)": 58.95,
        },
        {
            "Published Configuration": "DistilBERT",
            "Reported Size (MB)": 254.0,
            "Accuracy (%)": 50.02,
            "Macro F1 (%)": 47.34,
        },
        {
            "Published Configuration": "Prune + Quant",
            "Reported Size (MB)": 210.0,
            "Accuracy (%)": 51.46,
            "Macro F1 (%)": 49.39,
        },
        {
            "Published Configuration": "Distil-mBERT-mini",
            "Reported Size (MB)": 198.0,
            "Accuracy (%)": 49.54,
            "Macro F1 (%)": 46.80,
        },
        {
            "Published Configuration": "Distil + Prune",
            "Reported Size (MB)": 198.0,
            "Accuracy (%)": 48.07,
            "Macro F1 (%)": 44.90,
        },
        {
            "Published Configuration": "Quantized BanglaBERT",
            "Reported Size (MB)": 186.0,
            "Accuracy (%)": 52.33,
            "Macro F1 (%)": 51.82,
        },
        {
            "Published Configuration": "Distil + Quant",
            "Reported Size (MB)": 181.0,
            "Accuracy (%)": 48.12,
            "Macro F1 (%)": 46.33,
        },
        {
            "Published Configuration": "Distil + Prune + Quant",
            "Reported Size (MB)": cfg.paper_size_mb,
            "Accuracy (%)": cfg.paper_accuracy,
            "Macro F1 (%)": cfg.paper_macro_f1,
        },
        {
            "Published Configuration": "TinyBERT",
            "Reported Size (MB)": 55.0,
            "Accuracy (%)": 38.51,
            "Macro F1 (%)": 36.40,
        },
    ]
)
paper_reference["Usage"] = "Fixed external reference only"
paper_reference.to_csv(
    output_dir / "published_paper_fixed_reference.csv", index=False
)

# %% [markdown]
# ## 12. Clean final-size figures

# %%
plot_styles = {
    "12-layer BanglaBERT Teacher": {
        "color": "#2563eb",
        "marker": "s",
        "linestyle": "None",
        "zorder": 2,
        "markerfacecolor": "#2563eb",
    },
    "Direct Training Control (No KD)": {
        "color": "#64748b",
        "marker": "^",
        "linestyle": "-.",
        "zorder": 2,
        "markerfacecolor": "#64748b",
    },
    "Standard Logit-KD Control": {
        "color": "#ea580c",
        "marker": "D",
        "linestyle": "--",
        "zorder": 3,
        "markerfacecolor": "#ea580c",
    },
    "Proposed Method": {
        "color": "#15803d",
        "marker": "o",
        "linestyle": ":",
        "zorder": 4,
        "markerfacecolor": "none",
    },
}


def plot_breakpoint_metric(metric, filename):
    fig, axis = plt.subplots(figsize=(10, 6))
    for approach, style in plot_styles.items():
        rows = presentation_results[
            presentation_results["Approach"] == approach
        ].sort_values("Saved Weight File Size (MB)")
        if rows.empty:
            continue
        plot_groups = [rows]
        if approach != "12-layer BanglaBERT Teacher":
            # The 100/200 MB points share the same 5-layer architecture. The
            # 50 MB point is a different 1-layer architecture and must not be
            # connected as though quantization alone produced it.
            plot_groups = [
                rows[rows["Comparison Range (MB)"].isin([100, 200])],
                rows[rows["Comparison Range (MB)"] == 50],
            ]
        label_pending = True
        for group in plot_groups:
            if group.empty:
                continue
            axis.plot(
                group["Saved Weight File Size (MB)"],
                group[metric],
                color=style["color"],
                marker=style["marker"],
                linestyle=(
                    style["linestyle"] if len(group) > 1 else "None"
                ),
                linewidth=2,
                markersize=8,
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=2,
                zorder=style["zorder"],
                label=approach if label_pending else None,
            )
            label_pending = False

        if approach == "Proposed Method":
            architecture_labels = {50: "1-layer", 200: "5-layer"}
            for target_size, architecture_label in architecture_labels.items():
                point = rows[rows["Comparison Range (MB)"] == target_size]
                if not point.empty:
                    point = point.iloc[0]
                    axis.annotate(
                        architecture_label,
                        (point["Saved Weight File Size (MB)"], point[metric]),
                        xytext=(6, -14),
                        textcoords="offset points",
                        fontsize=7,
                        color=style["color"],
                    )

    base_plot_rows = presentation_results[
        presentation_results["Approach"] == "Standard Logit-KD Control"
    ][["Comparison Range (MB)", "Saved Weight File Size (MB)", metric]]
    proposed_plot_rows = presentation_results[
        presentation_results["Approach"] == "Proposed Method"
    ][["Comparison Range (MB)", "Saved Weight File Size (MB)", metric]]
    overlaps = base_plot_rows.merge(
        proposed_plot_rows,
        on="Comparison Range (MB)",
        suffixes=(" Standard KD", " Proposed"),
    )
    overlaps = overlaps[
        np.isclose(
            overlaps[f"{metric} Standard KD"],
            overlaps[f"{metric} Proposed"],
            atol=1e-6,
        )
        & np.isclose(
            overlaps["Saved Weight File Size (MB) Standard KD"],
            overlaps["Saved Weight File Size (MB) Proposed"],
            atol=0.25,
        )
    ]
    for _, overlap in overlaps.iterrows():
        axis.annotate(
            "Standard Logit KD = Proposed Method",
            (
                overlap["Saved Weight File Size (MB) Proposed"],
                overlap[f"{metric} Proposed"],
            ),
            xytext=(8, -16),
            textcoords="offset points",
            fontsize=7,
            color="#374151",
        )

    winner_name_column = (
        "Best Accuracy Approach"
        if metric == "Accuracy (%)"
        else "Best Macro F1 Approach"
    )
    winner_value_column = (
        "Best Accuracy (%)"
        if metric == "Accuracy (%)"
        else "Best Macro F1 (%)"
    )
    winner_size_column = (
        "Accuracy Winner Size (MB)"
        if metric == "Accuracy (%)"
        else "Macro F1 Winner Size (MB)"
    )
    for _, winner in breakpoint_winners.iterrows():
        target = int(winner["Comparison Range (MB)"])
        if target == 400 and len(
            presentation_results[
                presentation_results["Comparison Range (MB)"] == target
            ]
        ) == 1:
            prefix = "Teacher anchor"
        else:
            prefix = f"Best near {target} MB"
        annotation_offsets = {
            50: (8, -34),
            100: (8, 22),
            200: (6, 8),
            400: (6, -30),
        }
        axis.annotate(
            (
                f"{prefix}: {winner[winner_name_column]}\n"
                f"{winner[winner_size_column]:.1f} MB, "
                f"{winner[winner_value_column]:.2f}%"
            ),
            (
                winner[winner_size_column],
                winner[winner_value_column],
            ),
            xytext=annotation_offsets.get(target, (6, 8)),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    axis.set_xlabel("Measured Saved Weight File Size (MB)")
    axis.set_ylabel(metric)
    axis.set_title(f"{metric} vs. Measured Saved Weight File Size")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    axis.set_xlim(left=0)
    axis.text(
        0.99,
        0.01,
        (
            "Points use measured serialized weight size; deployment details "
            "are reported in model_artifact_audit.csv."
            + (
                "\nEvidence status: exploratory (holdout previously observed)"
                if cfg.prior_test_split_exposed
                else "\nEvidence status: confirmatory unseen holdout"
            )
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


plot_breakpoint_metric(
    "Accuracy (%)", "accuracy_vs_measured_saved_weight_size"
)
plot_breakpoint_metric(
    "Macro F1 (%)", "macro_f1_vs_measured_saved_weight_size"
)

print(
    "\nStandard Logit-KD Control definition: paper-inspired controlled "
    "instantiation using one-hot BCE, fixed-temperature KL, "
    "and the same architecture and size budget as the other methods."
)
print(
    "Interpretation: Proposed-versus-control results compare complete, "
    "validation-selected pipelines; they do not isolate a single loss term."
)
if cfg.run_mode == "live":
    holdout_registry = json.loads(
        holdout_registry_path.read_text(encoding="utf-8")
    )
    holdout_run_key = f"{split_digest}:training_seed={cfg.seed}"
    holdout_registry["splits"][holdout_run_key]["phase"] = (
        "final_test_complete"
    )
    holdout_registry_path.write_text(
        json.dumps(holdout_registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    run_status_path.write_text(
        json.dumps(
            {
                "phase": "final_test_complete",
                "selection_manifest_digest": selection_manifest_digest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
print(f"Finished. Outputs saved in: {output_dir}")
