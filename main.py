from optuna_hyp import Optuna
from ccbdl.config_loader.loaders import ConfigurationLoader
import ccbdl
import os
import torch
import sys


ccbdl.utils.logging.del_logger(source=__file__)

config_path = os.path.join(os.getcwd(), "config.yaml")
config = ConfigurationLoader().read_config("config.yaml")

# setting configurations
network_config = config["network"]
optimizer_config = config["optimized"]
data_config = config["data"]
learner_config = config["learning"]
study_config = config["study"]

study_path = ccbdl.storages.storages.generate_train_folder(name="",
                                                           generate=False,
                                                           location=os.path.dirname(os.path.realpath(__file__)))

opti = Optuna(study_config,
              optimizer_config,
              network_config,
              data_config,
              learner_config,
              config,
              study_path,
              comment="Study for Testing",
              config_path=config_path,
              debug=False,
              logging=True)

# Run Parameter Optimizer
opti.start_study()

# Evaluate Study
opti.eval_study()

# Summarize Study
opti.summarize_study()

# Saving the study
torch.save(opti.study, os.path.join(study_path, "study.pt"))

handler = ccbdl.evaluation.additional.notebook_handler("./StudySummary.ipynb", study_location = os.path.join(study_path, "study.pt"))
handler.save_as_html(directory = study_path, html_name = "study_analysis.html")

# Calculate Metrics for all trials
if sys.platform == "win32":
    opti.eval_metrics()
