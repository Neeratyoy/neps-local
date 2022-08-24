import logging

import torch
import numpy as np
import neps
from neps.search_spaces.search_space import SearchSpace
from neps_examples.multi_fidelity.model_and_optimizer import get_model_and_optimizer
from neps.optimizers.successive_halving.sampling_policy import RandomUniformPolicy, FixedPriorPolicy


def run_pipeline(working_directory, previous_working_directory, learning_rate, epoch):
    model, optimizer = get_model_and_optimizer(learning_rate)
    checkpoint_name = "checkpoint.pth"

    # Read in state of the model after the previous fidelity rung
    if previous_working_directory is not None:
        checkpoint = torch.load(previous_working_directory / checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_already_trained = checkpoint["epoch"]
        print(f"Read in model trained for {epoch_already_trained} epochs")

    # Train model here ...

    # Save model to disk
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        working_directory / checkpoint_name,
    )

    return np.log(learning_rate / epoch)  # Replace with actual error


pipeline_space = dict(
    learning_rate=neps.FloatParameter(
        lower=1e-4, upper=1e0, log=True, default=1e-1, default_confidence="high"
    ),
    epoch=neps.IntegerParameter(lower=1, upper=100, is_fidelity=True),
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    working_directory="results/sh_example",
    max_evaluations_total=50,
    searcher='successive_halving',
    sampling_policy=FixedPriorPolicy(SearchSpace(**pipeline_space))
)
previous_results, pending_configs = neps.status("results/sh_example")
