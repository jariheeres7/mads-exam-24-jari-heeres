from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from src import datasets, models, metrics

# Definitions for shorthand use of tune's sampling methods for hyperparameters
SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

def train(config: Dict):
    """
    Train function defines the training process configured to run with Ray Tune,
    which manages hyperparameter tuning dynamically.

    Parameters:
    - config (Dict): Dictionary containing configuration parameters and hyperparameters.
    """
    data_dir = config["data_dir"]
    
    trainfile = data_dir / 'heart_train.parq'
    testfile = data_dir / 'heart_test.parq'
    
    shape = (16, 12)
    traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=shape)
    testdataset = datasets.HeartDataset2D(testfile, target="target", shape=shape)

    with FileLock(data_dir / ".lock"):
        train = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=32)
        valid = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=32)

    f1micro = metrics.F1Score(average='micro')
    f1macro = metrics.F1Score(average='macro')
    precision = metrics.Precision('micro')
    recall = metrics.Recall('macro')
    accuracy = metrics.Accuracy()
    model = models.CNN(config)
    model.to("cpu")

    trainersettings = TrainerSettings(
        epochs=10,
        metrics=[accuracy, f1micro, f1macro, precision, recall],
        logdir=Path("."),
        train_steps=len(train),  
        valid_steps=len(valid),  
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 4},
        earlystop_kwargs=None,
    )

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=train.stream(),
        validdataloader=valid.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    trainer.loop()

if __name__ == "__main__":
    ray.init()

    data_dir = Path("data").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()
    
    config = {
        "hidden": tune.qrandint(96, 160, 8),
        "num_layers": tune.choice([1, 2]),
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "num_classes": 2,
        "dropout": tune.choice([0.3, 0.4, 0.5]),
        "shape": (16, 12),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("epochs")
    reporter.add_metric_column("hidden")
    reporter.add_metric_column("num_layers")
    reporter.add_metric_column("dropout")
    
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()