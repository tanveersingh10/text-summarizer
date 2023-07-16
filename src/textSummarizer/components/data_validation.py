import os
from textSummarizer.logging import logging
from textSummarizer.config.configuration import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "samsum_dataset"))
            all_files_set = set(all_files)  # convert list to set for quicker membership tests

            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files_set:
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write("Validation status: False")
                    return False

            with open(self.config.STATUS_FILE, "w") as f:
                f.write("Validation status: All data files exist")

            return True

        except Exception as e:
            logging.exception(e)
            raise e
        