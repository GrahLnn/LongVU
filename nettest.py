from longvu.utils import check_model
import shutil
from pathlib import Path

check_model("llamav")

path = Path.home() / ".cache" / "longvu" / "video" / "llama"


shutil.rmtree(path, ignore_errors=True)
