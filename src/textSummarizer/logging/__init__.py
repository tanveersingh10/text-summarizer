import os
import sys
import logging

log_filepath = os.path.join("logs","running_logs.log")
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level= logging.INFO,
    format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("textSummarizerLogger")