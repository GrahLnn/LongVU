# pyre-unsafe
import hashlib
from transformers import AutoConfig
from pathlib import Path
from huggingface_hub import snapshot_download


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print(
            "You are using newer LLaVA code base, while the checkpoint of v0 is from older code base."
        )
        print(
            "You must upgrade the checkpoint to the new code base (this can be done automatically)."
        )
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def verify_file(file_path: Path, expected_sha256: str) -> bool:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower() == expected_sha256.lower()


def check_model(model_type: str) -> str:
    MODEL_CONFIGS = {
        "llamav": {
            "repo_id": "Vision-CAIR/LongVU_Llama3_2_3B",
            "folder": Path.home() / ".cache" / "longvu" / "video" / "llama",
        },
        "qwenv": {
            "repo_id": "Vision-CAIR/LongVU_Qwen2_7B",
            "folder": Path.home() / ".cache" / "longvu" / "video" / "qwen",
        },
        "llamaimg": {
            "repo_id": "Vision-CAIR/LongVU_Llama3_2_3B_img",
            "folder": Path.home() / ".cache" / "longvu" / "image" / "llama",
        },
        "qwenimg": {
            "repo_id": "Vision-CAIR/LongVU_Qwen2_7B_img",
            "folder": Path.home() / ".cache" / "longvu" / "image" / "qwen",
        },
    }

    config = MODEL_CONFIGS.get(model_type)
    if not config:
        raise ValueError(f"Unsupported model type: {model_type}")

    snapshot_download(
        repo_id=config["repo_id"],
        local_dir=config["folder"],
    )

    return str(config["folder"])
