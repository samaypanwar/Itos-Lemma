
import numpy as np
import matplotlib.pyplot as plt
import yfinance
from icecream import ic
from functions import *
from sympy import *
import os

np.random.seed(420)
START_DATE = "2018-09-20"
END_DATE = "2021-03-21"
TICKER = 'MCD'
stock = yfinance.download(TICKER, start=START_DATE, end=END_DATE)

returns = stock.Close.pct_change()[1:]
mu = np.mean(returns)
sigma = np.std(returns)
X = symbols("X")
t = symbols("t")
drift = mu * X
diffusion = sigma * X
function = X * exp(mu*(len(stock) - t))

if __name__ == '__main__':
    
    derivativeDrift, derivativeDiffusion = itos_lemma(functionofX=function, driftFunction=drift,
                                                        diffusionFunction=diffusion)
    derivativeTimeSeries = euler_maruyama(derivativeDrift, derivativeDiffusion, timePeriod=len(stock),
                                                        numberOfSimulations=50)

    Simulations = geometric_brownian_motion(stock.Close)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
    fig.suptitle('GBM simulations and Forward Price simulations', size=20)
    axes[0].plot(Simulations)
    axes[0].plot(stock.Close.tolist(), color='b', linewidth=2)
    axes[1].plot(derivativeTimeSeries)
    plt.tight_layout()
    plt.show()
