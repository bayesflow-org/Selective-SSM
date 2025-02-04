import os
os.environ["KERAS_BACKEND"] = "torch"

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import nbinom

import torch
import keras
import bayesflow as bf

# Define simulator
RNG = np.random.default_rng(2024)

def prior():
    """Generates a random draw from the joint prior."""
    lambd = RNG.lognormal(mean=np.log(0.4), sigma=0.5)
    mu = RNG.lognormal(mean=np.log(1 / 8), sigma=0.2)
    D = RNG.lognormal(mean=np.log(8), sigma=0.2)
    I0 = RNG.gamma(shape=2, scale=20)
    psi = RNG.exponential(5)
    return {"lambd": lambd, "mu": mu, "D": D, "I0": I0, "psi": psi}

def convert_params(mu, phi):
    """Helper function to convert mean/dispersion parameterization of a negative binomial to N and p,
    as expected by numpy's negative_binomial.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """

    r = phi
    var = mu + 1 / r * mu**2
    p = (var - mu) / var
    return r, 1 - p

def stationary_SIR(lambd, mu, D, I0, psi, N=83e6, T=14, eps=1e-5):
    """Performs a forward simulation from the stationary SIR model given a random draw from the prior."""

    # Extract parameters and round I0 and D
    I0 = np.ceil(I0)
    D = int(round(D))

    # Initial conditions
    S, I, R = [N - I0], [I0], [0]

    # Reported new cases
    C = [I0]

    # Simulate T-1 timesteps
    for t in range(1, T + D):
        # Calculate new cases
        I_new = lambd * (I[-1] * S[-1] / N)

        # SIR equations
        S_t = S[-1] - I_new
        I_t = np.clip(I[-1] + I_new - mu * I[-1], 0.0, N)
        R_t = np.clip(R[-1] + mu * I[-1], 0.0, N)

        # Track
        S.append(S_t)
        I.append(I_t)
        R.append(R_t)
        C.append(I_new)

    reparam = convert_params(np.clip(np.array(C[D:]), 0, N) + eps, psi)
    C_obs = RNG.negative_binomial(reparam[0], reparam[1])
    return dict(cases=C_obs)

def load_data():
    """Helper function to load cumulative cases and transform them to new cases."""

    confirmed_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    confirmed_cases = pd.read_csv(confirmed_cases_url, sep=",")

    date_data_begin = datetime.date(2020, 3, 1)
    date_data_end = datetime.date(2020, 3, 15)
    format_date = lambda date_py: f"{date_py.month}/{date_py.day}/{str(date_py.year)[2:4]}"
    date_formatted_begin = format_date(date_data_begin)
    date_formatted_end = format_date(date_data_end)

    cases_obs = np.array(confirmed_cases.loc[confirmed_cases["Country/Region"] == "Germany", date_formatted_begin:date_formatted_end])[0]
    new_cases_obs = np.diff(cases_obs)
    return new_cases_obs



if __name__ == "__main__":
    # Verify GPU is accessible and list devices
    num_devices = torch.cuda.device_count()
    assert num_devices >= 1, "NVIDIA GPU IS REQUIRED"
    print("Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Available devices:", num_devices)
    
    # Define simulator
    simulator = bf.make_simulator([prior, stationary_SIR])
    
    # Define adapter
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series("cases")
        .concatenate(["lambd", "mu", "D", "I0", "psi"], into="inference_variables")
        .rename("cases", "summary_variables")
        .apply(forward=lambda x: np.log1p(x), inverse=lambda x: np.expm1(x))
    )
    
    # Define summary and inference networks
    summary_net = bf.wrappers.MambaSSM(pooling=True, dropout=0.5)
    inference_net = bf.networks.FlowMatching(subnet_kwargs={"residual": True, "dropout": 0.1, "widths": (128, 128, 128)})
    
    # Construct workflow
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
        inference_variables=["lambd", "mu", "D", "I0", "psi"]
    )
    
    # Generate training data
    training_data = workflow.simulate(5000)
    validation_data = workflow.simulate(300)
    
    # Train model
    print("Training...")
    history = workflow.fit_offline(training_data, epochs=100, batch_size=64, validation_data=validation_data)
    
    # Diagnostics
    test_size = 300
    metrics = workflow.compute_diagnostics(test_data=test_size)
    print(metrics)
    
    figures = workflow.plot_diagnostics(
        test_data=test_size,
        loss_kwargs={"figsize": (15, 3), "label_fontsize": 12},
        recovery_kwargs={"figsize": (15, 3), "label_fontsize": 12},
        calibration_ecdf_kwargs={"figsize": (15, 3), "legend_fontsize": 8, "difference": True, "label_fontsize": 12},
        z_score_contraction_kwargs={"figsize": (15, 3), "label_fontsize": 12}
    )
        
    for figure in figures.keys():
        print("Saving", figure)
        figures[figure].savefig("diagnostics/" + figure)