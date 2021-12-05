import os
import json

class Experiment():
    EXPERIMENT_DIR = "experiments/{name}"
    ARGS_PATH = os.path.join(EXPERIMENT_DIR, "args.json")
    TRAINED_MODELS_DIR = os.path.join(EXPERIMENT_DIR, "trained-models")

    def __init__(self, args, name):
        self.args = args
        self.name = name

    def record_args(self):
        os.makedirs(os.path.dirname(self.args_path), exist_ok=True)
        str_args = json.dumps(self.args, indent=2)
        with open(self.args_path, "w") as args_file:
            args_file.write(str_args)

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