from pathlib import Path
from logging import DEBUG as DEFAULT_LOG_LEVEL

MAX_FILE_LOCK_WAIT = 600
VERBOSE = False
VERBOSITY = 1
LOG_FILE_NAME = "run.log"
LOG_DIR = Path.home().joinpath(".config/gitops/logs")
