import numpy as np
import matplotlib.pyplot as plt
import yfinance
from functions import *
from sympy import *
import yaml
np.random.seed(3016)

with open('ItosLemma/config.yaml') as file:
    config = yaml.safe_load(file)

START_DATE = config.get('START_DATE')
END_DATE = config.get('END_DATE')
TICKER = config.get('TICKER')
numberOfSimulations = config.get('NUMBER_OF_SIMULATIONS')

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
                                                        numberOfSimulations=numberOfSimulations)

    Simulations = geometric_brownian_motion(stock.Close, numberOfSimluations=numberOfSimulations)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
    fig.suptitle('GBM simulations and Forward Price simulations', size=20)
    axes[0].plot(Simulations)
    axes[0].plot(stock.Close.tolist(), color='b', linewidth=2)
    axes[1].plot(derivativeTimeSeries)
    plt.tight_layout()
    plt.show()
