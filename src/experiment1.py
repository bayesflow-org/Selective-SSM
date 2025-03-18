import os
os.environ["KERAS_BACKEND"] = "torch"

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import keras
import bayesflow as bf

def create_fname(params: tuple) -> str:
    dt, budget = params
    fname = "dt-" + str(1 / dt).split(".")[0] + "-" + "budget-" + str(budget) + ".png"
    return fname

def generate_datasets(dt_params: list[float]):
    # TODO
    # Generate datasets by dt_params list
    # Clear data folder
    # Add completed datasets to data folder
    
    return None


if __name__ == "__main__":
    # Verify GPU is accessible and list devices
    num_devices = torch.cuda.device_count()
    assert num_devices >= 1, "GPU IS REQUIRED"
    print("Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Available devices:", num_devices)
    device = "cuda"
    
    # Define experimental params
    dt_params = [0.01, 0.001, 0.0001]
    budget_params = [1000, 10000, 1000000]
    
    experiment_params = list(itertools.product(*[dt_params, budget_params]))
    
    # Experiment workflow
    for params in experiment_params:
        # Log
        keras.backend.clear_session() # clears gradients between runs
        print("Testing params:")
        print(params)
        
        # Define simulator
        dt, budget = params
        simulator = bf.benchmarks.LotkaVolterra(subsample=None, flatten=False, dt=dt)
        
        # Summary and Inference Networks
        summary_net = bf.wrappers.MambaSSM(
            feature_dim=2,
            state_dim=32,
            conv_dim=4,
            expand=4,
            mamba_blocks=4,
            summary_dim=64,
            device=device
        )
        
        inference_net = bf.networks.FlowMatching(
            subnet_kwargs={"residual": True, "dropout": 0.1, "widths": (512,) * 4}
        )
        
        # Construct workflow (specific to LV for now)
        adapter = (
            bf.adapters.Adapter()
            .as_time_series("observables")
            .rename("parameters", "inference_variables")
            .rename("observables", "summary_variables")
            .apply(forward=lambda x: np.log1p(x), inverse=lambda x: np.expm1(x))
        )
        workflow = bf.BasicWorkflow(
            simulator=simulator,
            adapter=adapter,
            inference_network=inference_net,
            summary_network=summary_net
        )
        
        # Train
        training_set = workflow.simulate(budget)
        validation_set = workflow.simulate(500)
        history = workflow.fit_offline(training_set, epochs=1, batch_size=64, validation_data=validation_set)
        
        # Test diagnostics + save model
        test_sims = workflow.simulate(300) 
        test_obs = test_sims.pop("observables")
        samples = workflow.sample(conditions={"observables": test_obs}, num_samples=250)
        
        fig = bf.diagnostics.plots.recovery(samples, test_sims)
        fig.savefig("diagnostics/" + create_fname(params))
        print("----------------------")