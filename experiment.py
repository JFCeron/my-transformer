import os
import json
from datetime import datetime

class Experiment():
    EXPERIMENT_DIR = "experiments/{name}"
    ARGS_PATH = os.path.join(EXPERIMENT_DIR, "args.json")
    TRAINED_MODELS_DIR = os.path.join(EXPERIMENT_DIR, "trained-models")

    def __init__(self, args, name):
        self.args = args
        self.name = name
        assert not os.path.exists(self.experiment_dir)

    def record_args(self):
        os.makedirs(os.path.dirname(self.args_path))
        str_args = json.dumps(self.args, indent=2)
        with open(self.args_path, "w") as args_file:
            args_file.write(str_args)

    def get_new_model_path(self):
        new_model_path = datetime.now().strftime("%Y-%m-%d_%M-%S.pt")
        new_model_path = os.path.join(self.trained_models_dir, new_model_path)
        return new_model_path

    @property
    def experiment_dir(self):
        return self.EXPERIMENT_DIR.format(name=self.name)
    
    @property
    def args_path(self):
        return self.ARGS_PATH.format(name=self.name)

    @property
    def trained_models_dir(self):
        return self.TRAINED_MODELS_DIR.format(name=self.name)

    @classmethod
    def load(name):
        experiment = Experiment(args={}, name=name)
        with open(experiment.args_path, "r") as args_file:
            str_args = args_file.read()
            experiment.args = json.loads(str_args)