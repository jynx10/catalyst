from typing import Dict, List
import numpy as np
from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS
import os, sys, pickle

if SETTINGS.comet_required:
    import comet_ml

CONFIG_FILE_PATH = "/root/.comet.config"
CONFIG_FILE_PATH = os.environ.get("COMET_CONFIG_PATH", "~/.comet.config")


def _format_prefix(prefix_parameters: List) -> None:
    """Formats the prefix of the log according to the given parameters.
    
    Parameters:
        prefix_parameters (List): A list of the given arguments.

    Returns:
        prefix (str): The formatted prefix string.
    
    """

    prefix = None
    passed_prefix_parameters = [
        parameter for parameter in prefix_parameters if parameter is not None
    ]

    if len(passed_prefix_parameters) > 1:
        prefix = "_".join(passed_prefix_parameters)

    elif len(passed_prefix_parameters) == 1:
        prefix = passed_prefix_parameters[0]

    return prefix


class CometLogger(ILogger):
    """Comet logger for parameters, metrics, images and other artifacts (videos, audio,
    model checkpoints, etc.).

    Comet documentation: https://www.comet.ml/docs/

    To start with Comet please check out: https://www.comet.ml/docs/quick-start/.
    You will need an ``api_token`` and experiment with a project name to log your Catalyst runs to.

    Args:
        project_name: Optional, ``str``, the name of the project within Comets's run.
          Default is "general" and can be seen in the comet UI under "Uncategorized Experiments".
        workspace : Optional, ``str``. Attach an experiment to a project the belongs to this workspace.
        Read more about workspaces in the `Comet User Interface docs <https://www.comet.ml/docs/user-interface/>`
        api_token: Optional, ``str``. Your Comet's API token.
        Read more about it in the `Comet installation docs <https://www.comet.ml/docs/quick-start/>`.
        experiment: Optional, pass a Comet Existing Experiment object if you want to continue logging
          to the existing experiment (resume experiment).
          Read more about Existing Experiment `here <https://www.comet.ml/docs/python-sdk/ExistingExperiment/>`_.
        tags: Optional, pass a list of tags to add to the Experiment. Tags will be shows in the dashboard.

    Python API examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...
            loggers={
                "comet": dl.CometLogger(
                    project_name="my-comet-project
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
                        project_name="my-comet-project"
                    )
                }
            # ...

        runner = CustomRunner().run()
    """

    def __init__(self, project_name: str = None, workspace: str = None, api_key: str = None, experiment: comet_ml.ExistingExperiment = None, tags: List = None) -> None:
        self.project_name = project_name
        self.logging_disabled = os.getenv('COMET_AUTO_LOG_DISABLE', False)
        self.workspace = workspace
        self.api_key = api_key
        self.comet_mode = os.getenv('COMET_MODE', 'online')

        if experiment is None:
            try:
                if self.comet_mode == 'offline':
                    print("Starting an Offline Experiment")
                    self.offline_directory = os.getenv('COMET_OFFLINE_DIRECTORY')
                    self.experiment = comet_ml.OfflineExperiment(
                        project_name=self.project_name, workspace=self.workspace, offline_directory=self.offline_directory, disabled=self.logging_disabled)
                else:
                    self.experiment = comet_ml.Experiment(
                        project_name=self.project_name, api_key=self.api_key, workspace=self.workspace, disabled=self.logging_disabled
                    )

            except BaseException as e:
                print(e)
                sys.exit()
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
        """Logs the metrics to the logger."""

        prefix_parameters = [stage_key, loader_key, scope]
        prefix = _format_prefix(prefix_parameters)

        self.experiment.log_metrics(metrics, step=global_batch_step,
                                    epoch=global_batch_step, prefix=prefix)

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

        prefix_parameters = [stage_key, loader_key, scope]
        prefix = _format_prefix(prefix_parameters)
        self.image_name = f"{prefix}_{tag}"

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

        prefix_parameters = [stage_key, scope]
        prefix = _format_prefix(prefix_parameters)

        self.experiment.log_parameters(hparams, prefix=prefix)

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

        metadata_parameters = {'stage_key': stage_key, 'loader_key': loader_key}
        passed_metadata_parameters = {k: v for k,
                                      v in metadata_parameters.items() if v is not None}
        if path_to_artifact:
            self.experiment.log_asset(path_to_artifact, tag,
                                      step=global_batch_step, metadata=passed_metadata_parameters)
        else:
            self.experiment.log_asset_data(
                pickle.dumps(artifact), tag, step=global_batch_step, epoch=global_epoch_step, metadata=passed_metadata_parameters)

    def flush_log(self) -> None:
        """Flushes the loggers."""
        pass

    def close_log(self, scope: str = None) -> None:
        """Closes the logger."""
        if scope is None or scope == "experiment":
            self.experiment.end()


__all__ = ["CometLogger"]
