from logging import NullHandler
from typing import Dict, List
import numpy as np
from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS
import os

if SETTINGS.comet_required:
    import comet_ml


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
        run: Optional, pass a Comet Experiment run object if you want to continue logging
          to the existing run (resume run).
          Read more about Existing Experiment `here <https://www.comet.ml/docs/python-sdk/ExistingExperiment/>`_.
        tags: Optional, pass a list of tags to add to the Experiment. Tags will be shows in the dashboard.  
        comet_run_kwargs: Optional, additional keyword arguments to be passed directly to the
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
    
    def __init__(self, project_name: str = None, workspace: str = None, api_key: str = None, run: comet_ml.Experiment = None, tags: List = None, **comet_run_kwargs) -> None:
        if project_name is None:
            enviornment_project_name = os.environ.get('COMET_PROJECT_NAME')
            if enviornment_project_name:
                self.project_name = enviornment_project_name
                print("Using the project name stored in the 'COMET_PROJECT_NAME' enviornment variable.")
            else:
                self.project_name = 'experiments'
                print("A project_name has not been set. Using default project name 'experiments' instead.")
        else:
            self.project_name = project_name

        self.workspace = workspace
        self.api_key = api_key
        self._comet_run_kwargs = comet_run_kwargs

        if run is None:
            try:
                if api_key is None:
                    enviornment_api_key = os.environ.get('COMET_API_KEY')
                    if enviornment_api_key:
                        self.api_key = enviornment_api_key
                        print("Using the API key stored in the COMET_API_KEY' enviornment variable.")
                    else:
                        print("A Comet API key was not give and not found in the 'COMET_API_KEY' enviornment variable.")
                self.run = comet_ml.Experiment(
                    project_name = self.project_name, api_key = self.api_key, workspace = self.workspace, **self._comet_run_kwargs
                )

                if tags is not None:
                    for tag in tags:
                        self.run.add_tag(tag)
            
            except BaseException as e:
                print(e)
        else:
            self.run = run

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
        self.run.log_metrics(metrics, step=global_batch_step)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
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
        self.run.log_image(image, "images")

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs hyperparameters to the logger."""
        self.run.log_parameters(hparams)

    def close_log(self) -> None:
        """Closes the logger."""
        self.run.end()


__all__ = ["CometLogger"]
