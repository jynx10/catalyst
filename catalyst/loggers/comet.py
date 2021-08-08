from typing import Dict, List
import numpy as np
from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS
import os
import configparser
import pickle

if SETTINGS.comet_required:
    import comet_ml

CONFIG_FILE_PATH = "/root/.comet.config"
CONFIG_FILE_PATH = os.environ.get("COMET_CONFIG_PATH", "~/.comet.config")


class CometLogger(ILogger):
    """Comet logger for parameters, metrics, images and other artifacts (videos, audio,
    model checkpoints, etc.).

    Comet documentation: https://www.comet.ml/docs/

    To start with Comet please check out: https://www.comet.ml/docs/quick-start/.
    You will need an ``api_token`` and experiment with a project name to log your Catalyst runs to.

    Args:
        project_name: Optional, ``str``, the name of the project within Comets's run.
          Default is "experiments".
        workspace : Optional, ``str``. Attach an experiment to a project the belongs to this workspace.
        Read more about workspaces in the `Comet User Interface docs <https://www.comet.ml/docs/user-interface/>`
        api_token: Optional, ``str``. Your Comet's API token. 
        Read more about it in the `Comet installation docs <https://www.comet.ml/docs/quick-start/>`.
        experiment: Optional, pass a Comet Existing Experiment object if you want to continue logging
          to the existing experiment (resume experiment).
          Read more about Existing Experiment `here <https://www.comet.ml/docs/python-sdk/ExistingExperiment/>`_.
        tags: Optional, pass a list of tags to add to the Experiment. Tags will be shows in the dashboard.  
        comet_experiment_kwargs: Optional, additional keyword arguments to be passed directly to the
          `Experiment.__init__() <https://www.comet.ml/docs/python-sdk/Experiment/#experiment__init__>`_ function.

    Python API examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...
            loggers={
                "comet": dl.CometLogger(
                    project_name="Ds's Experiment"
                )
            }
        )

    .. code-block:: python

        from catalyst import dl

        class CustomRunner(dl.IRunner):
            # ...

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "comet": dl.CommetLogger(
                        project="Ds's Experiment"
                    )
                }
            # ...

        runner = CustomRunner().run()
    """

    def __init__(self, project_name: str = None, workspace: str = None, api_key: str = None, experiment: comet_ml.ExistingExperiment = None, tags: List = None) -> None:
        if project_name is None:
            enviornment_project_name = os.environ.get('COMET_PROJECT_NAME')
            if enviornment_project_name:
                self.project_name = enviornment_project_name
                print("Using the project name stored in the 'COMET_PROJECT_NAME' enviornment variable.")
            else:
                print("A project_name has not been set. Using default project name 'general' instead.")
        else:
            self.project_name = project_name
        
        self.logging_disabled = os.getenv('COMET_AUTO_LOG_DISABLE', False)
        self.workspace = workspace
        self.api_key = api_key
        self.comet_mode = os.getenv('COMET_MODE', 'online')

        if experiment is None:
            try:
                if api_key is None:
                    parser = configparser.ConfigParser()
                    enviornment_api_key = os.environ.get('COMET_API_KEY')
                    try:
                        parser.read(CONFIG_FILE_PATH)
                        config_file_api_key = parser.get("comet", "api_key")
                    except:
                        config_file_api_key = None
                    if config_file_api_key:
                        self.api_key = config_file_api_key
                        print("Using the API key stored in the comet config file.")
                    elif enviornment_api_key:
                        self.api_key = enviornment_api_key
                        print("Using the API key stored in the COMET_API_KEY' enviornment variable.")
                
                if self.comet_mode == 'offline':
                    print("Starting an Offline Experiment")
                    self.offline_directory = os.getenv('COMET_OFFLINE_DIRECTORY')  
                    self.experiment = comet_ml.OfflineExperiment(project_name=self.project_name, workspace=self.workspace, offline_directory=self.offline_directory)
                else:
                    self.experiment = comet_ml.Experiment(
                        project_name=self.project_name, api_key=self.api_key, workspace=self.workspace
                    )

            except BaseException as e:
                experiment = None
                print(e)
        else:
            self.experiment = experiment
        
        if tags:
            for tag in tags:
                self.experiment.add_tag(tag)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        key_parameters = [stage_key, loader_key, scope]
        # Removes all None values from list.
        passed_key_parmeters = [key_parameter for key_parameter in key_parameters if key_parameter]
        keys_prefix = None
        if len(passed_key_parmeters) == 1:
            keys_prefix = passed_key_parmeters[0]
        elif len(passed_key_parmeters) > 1:
            keys_prefix = '_'.join(passed_key_parmeters)
        
        self.experiment.log_metrics(metrics, step=global_batch_step, epoch=global_batch_step, prefix=keys_prefix)

    def log_image(
        self,
        tag: str = "images",
        image: np.ndarray = [],
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs image to the logger."""
        self.image_name = tag
        key_parameters = [stage_key, loader_key, scope]
        # Removes all None values from list.
        passed_key_parmeters = [key_parameter for key_parameter in key_parameters if key_parameter]
        if len(passed_key_parmeters) == 1:
            keys_prefix = passed_key_parmeters[0]
            self.image_name = f"{keys_prefix}_{tag}"
        elif len(passed_key_parmeters) > 1:
            keys_prefix = '_'.join(passed_key_parmeters)
            self.image_name = f"{keys_prefix}_{tag}"
        self.experiment.log_image(image, name=self.image_name, step=global_batch_step)

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs hyperparameters to the logger."""
        key_parameters = [stage_key, scope]
        passed_key_parmeters = [key_parameter for key_parameter in key_parameters if key_parameter]
        if len(passed_key_parmeters) == 1:
            keys_prefix = passed_key_parmeters[0]
            self.experiment.log_parameters(hparams, prefix=keys_prefix)
        elif len(passed_key_parmeters) > 1:
            keys_prefix = '_'.join(passed_key_parmeters)
            self.experiment.log_parameters(hparams, prefix=keys_prefix)
        else:
            self.experiment.log_parameters(hparams)

    def log_artifact(
        self,
        tag: str = "artifact",
        artifact: object = None,
        path_to_artifact: str = None,
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs artifact (arbitrary file like audio, video, model weights) to the logger."""
        if path_to_artifact or artifact:
            key_parameters = {'stage_key': stage_key, 'loader_key': loader_key}
            passed_key_parameters = {k: v for k, v in key_parameters.items() if v is not None}
            if path_to_artifact:
                self.experiment.log_asset(path_to_artifact, tag,
                                          step=global_batch_step, metadata=passed_key_parameters)
            else:
                self.experiment.log_asset_data(
                    pickle.dumps(artifact), tag, step=global_batch_step, epoch=global_epoch_step, metadata=passed_key_parameters)
        else:
            print("No Artifact Data or a path to the artifact were provided.")

    def flush_log(self) -> None:
        """Flushes the loggers."""
        pass

    def close_log(self, scope: str = None) -> None:
        """Closes the logger."""
        if scope is None or scope == "experiment":
            self.experiment.end()


__all__ = ["CometLogger"]
