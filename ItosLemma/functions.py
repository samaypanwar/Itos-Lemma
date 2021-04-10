import numpy as np
import matplotlib.pyplot as plt
import yfinance
from icecream import ic
import warnings
import pandas as pd
from sympy import *

warnings.filterwarnings("ignore")
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = 15, 5
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["font.family"] = "Times New Roman"


def plot_timeseries(timeSeries):
    """ Plots all the Time Series in a given dataframe """

    fig, axes = plt.subplots(
        nrows=len(timeSeries.columns) // 2, ncols=2, dpi=120, figsize=(16, 8)
    )
    fig.suptitle("Stock Components", size=18)

    for i, ax in enumerate(axes.flatten()):

        data = timeSeries[timeSeries.columns[i]]
        ax.plot(data, color="r", linewidth=1)
        ax.set_title(timeSeries.columns[i], size=16)
        ax.tick_params(labelsize=10)
        ax.grid(True)
        ax.set_xlabel("Days")

    plt.tight_layout()


def geometric_brownian_motion(timeSeries, numberOfSimluations=50, sizeOfTimeInterval=1, plotSimulations=False, returnNone=False):
    """ timeSeries is a one-dimensional numpy array or list.
        numberOfSimluations is the number of GBM paths we want to create
        Size of time interval is how frequent our data values are
        If plotSimulations is True then a plot is generated with all the simulations along with
        the actual timeSeries in blue
        If returnNone is True then it does not return anything

        This function calculates the drift and diffusion coefficients for the timeSeries
        and simulates a geometric brownian motion path of the same.
    """

    returns = timeSeries.pct_change()[1:]       # Calculate the daily percentage returns of the timeSeries
    So = timeSeries[0]                          # Inital value of the timeSeries
    timePeriod = len(timeSeries)
    N = int(timePeriod / sizeOfTimeInterval)    # Number of simulation points
    t = np.arange(1, N + 1)
    mu = np.mean(returns)                       # Mean returns
    sigma = np.std(returns)                     # Variance of returns

    dB = {
        str(simulation): np.random.standard_normal(size=N)
        for simulation in range(1, numberOfSimluations + 1)
    }                                           # Creates white noise sample of size N
    dW = {
        str(simulation): dB[str(simulation)].cumsum() * np.sqrt(sizeOfTimeInterval)
        for simulation in range(1, numberOfSimluations + 1)
    }                                           # Brownian motion path

    drift = (mu - 0.5 * sigma ** 2) * t         # Drift component remains same for all simulations as it is non-stochastic
    diffusion = {
        str(simulation): sigma * dW[str(simulation)]
        for simulation in range(1, numberOfSimluations + 1)
    }                                           # Diffusion component is different for simulations

    S = np.array(
        [So * np.exp(drift + diffusion[str(simulation)])
            for simulation in range(1, numberOfSimluations + 1)])   # Calculate simulated path for each simulation

    Simulations = {}
    for simulation in range(numberOfSimluations):
        Simulations[str(simulation + 1)] = S[simulation, :]
    Simulations = pd.DataFrame(data=Simulations,)

    if plotSimulations or returnNone:
        plt.plot(Simulations)
        plt.plot(timeSeries.tolist(), color="blue", linewidth=2)
        plt.title("Simulated time series", size=18)
        plt.legend(["Actual Price in Blue"], loc="upper left")

    if not returnNone:
        return Simulations


def itos_lemma(functionofX, driftFunction, diffusionFunction):
    """
    This function applies Ito's Lemma to a twice differentiable function
    (function) with drift and diffusion components as functions of (x, t)
    for a stochastic process X[t] which follows an Ito's process:

        dX = drift(X, t)dt + diffusion(X, t)dW

    This function returns a new sympy function which the derivative of X which follows (function) follows
    """

    X = symbols("X")        # Sympy requires us to define variables we will use beforehand
    t = symbols("t")
    delX = lambda function: diff(function, X)       # Partial derivative with respect to X of a function
    delT = lambda function: diff(function, t)       # Partial derivative with respect to t of a function

    # These functions create lambda functions to calculate the
    # the new drift and diffusion components of the derivative
    derivativeDrift = lambdify(
        [X, t],
        delT(function) + delX(function) * driftFunction
        + (1 / 2) * delX(delX(function)) * (diffusionFunction ** 2),
        "math",
    )
    derivativeDiffusion = lambdify([X, t], delX(function) * diffusionFunction, "math")

    return derivativeDrift, derivativeDiffusion


def euler_maruyama(derivativeDrift, derivativeDiffusion, timePeriod, sizeOfTimeInterval=1, initialValue=0, numberOfSimulations=1):
    """ In Itô calculus, the Euler–Maruyama method (also called the Euler method)
    is a method for the approximate numerical solution of a stochastic differential equation.

    Consider the differential equation:

    dX = a(X, t)dt + b(X, t)dW

    with the initial conditions X[0] = x0
    then the EM approximation can be defined as:

    Y[n+1] = Y[n] + a(Y[n], t[n])dt + b(Y[n], t[n])dW

    where dW is white noise with variance dt
    """

    N = int(timePeriod / sizeOfTimeInterval)
    dt = sizeOfTimeInterval
    # Create empty simulated series for all simulations
    derivativeTimeSeries = {str(simulation):np.zeros(N+1) for simulation in range(1, numberOfSimulations + 1)}

    for simulation in range(1, numberOfSimulations + 1):
        for t in range(N):
            dW = np.random.normal(scale = np.sqrt(dt))  # Random component dW for each interval
            Yt = derivativeTimeSeries[str(simulation)][t]
            derivativeTimeSeries[str(simulation)][t + 1] = Yt + derivativeDrift(X=Yt, t=t) * dt + derivativeDiffusion(X=Yt, t=t) * dW

    derivativeTimeSeries = pd.DataFrame(derivativeTimeSeries)

    return derivativeTimeSeries