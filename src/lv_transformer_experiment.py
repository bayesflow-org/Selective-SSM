import os
os.environ["KERAS_BACKEND"] = "torch"

import itertools

import matplotlib

import torch
import keras
import bayesflow as bf


def create_fname(params: tuple, prepend: str = None) -> str:
    dt, budget = params
    fname = "dt-" + str(1 / dt).split(".")[0] + "-" + "budget-" + str(budget)
    if prepend is not None:
        fname = prepend + "-" + fname
    return fname


if __name__ == "__main__":

    # Verify GPU is accessible and list devices
    num_devices = torch.cuda.device_count()
    assert num_devices >= 1, "GPU IS REQUIRED"
    print("Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Available devices:", num_devices)
    device = "cuda"
    
    # Define experimental params
    dt_params = [0.01, 0.001, 0.0001]
    budget_params = [1000, 10_000, 100_000]

    epochs = [200, 100, 50]
    epochs = {k: v for k, v in zip(budget_params, epochs)}

    dropouts = [0.2, 0.1, 0.05]
    dropouts = {k: v for k, v in zip(budget_params, dropouts)}

    experiment_params = list(itertools.product(*[dt_params, budget_params]))
    

    # Experiment workflow
    for params in experiment_params:
        # Log
        print("Testing params:")
        print(params)

        # Define simulator
        dt, budget = params
        simulator = bf.simulators.LotkaVolterra(subsample=None, flatten=False, dt=dt)

        # Summary and inference network, transformer with roughly the same # params as MAMBA
        summary_net = bf.networks.TimeSeriesTransformer(embed_dims=(32, 32))

        inference_net = bf.networks.CouplingFlow(coupling_kwargs={"subnet_kwargs": {"dropout": dropouts[budget]}})

        # Construct workflow
        adapter = (
            bf.adapters.Adapter()
            .convert_dtype("float64", "float32")
            .as_time_series("observables")
            .log("observables", p1=True)
            .rename("parameters", "inference_variables")
            .rename("observables", "summary_variables")
        )

        workflow = bf.BasicWorkflow(
            simulator=simulator,
            adapter=adapter,
            inference_network=inference_net,
            summary_network=summary_net,
            checkpoint_filepath="transformer_results/checkpoints/" + create_fname(params),
            inference_variables=["parameters"],
        )

        # Train
        training_set = workflow.simulate(budget)
        validation_set = workflow.simulate(300)
        history = workflow.fit_offline(training_set, epochs=epochs[budget], batch_size=64, validation_data=validation_set)
        
        # Evaluate
        df = workflow.compute_default_diagnostics(validation_set)
        figures = workflow.plot_default_diagnostics(validation_set)
        latex_df = df.to_latex(index=True, float_format="{:.3f}".format)
        
        # Save numeric diagnostics as a latex table
        with open("transformer_results/diagnostics/" + create_fname(params, "latex") + ".tex", "w") as f:
            f.write(latex_df)
        
        # Save numeric diagnostics as csv
        df.to_csv("transformer_results/diagnostics/" + create_fname(params, "table") + ".csv", header=True, index=True, float_format='%.3f')

        # Save figures
        for k, fig in figures.items():
            fig.savefig("transformer_results/diagnostics/" + create_fname(params, k) + ".png", dpi=300)
        
        # Clear stuff between runs
        matplotlib.pyplot.close()
        keras.backend.clear_session()
        print("----------------------\n")
