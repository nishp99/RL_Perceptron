# RL-Perceptron
Generalisation dynamics of policy learning in high-dimensions

## Simulations
The code for numerically solving the ODEs and their corresponding simulations can be run in ODEs_and_Simulations.ipynb

## Bossfight - Procgen

Supported Platforms:
- Windows 10
- macOS 10.14 (Mojave), 10.15 (Catalina)
- Linux (manylinux2010)

Supported Pythons:
- 3.7 64-bit
- 3.8 64-bit
- 3.9 64-bit
- 3.10 64-bit

### Install procgen from cource

```
cd procgen
conda env update --name bossfight --file bossfight.yml
conda activate bossfight
pip install -e .
# this should say "building procgen...done"
```
Additionally install the pytorch appropriate to you.

In the bossfight environment, training and evaluation can be run by `interactive_script.py` in `procgen/bossfight/reinforce`.
Training must be done in two  steps:
1. Train the preagents: run `interactive_script.py` with `preagents=False` and `save_model=True` for many times to save agents which will be used to generate the random initialisations of the game
2. Change the locations of the loaded agents in `training_reinf.py` to match the locations of the model parameters of the agents you have trained. Then run `interactive_script.py` with `preagents=True`

## Pong
Requirements: Python 3.8, 3.9, 3.10, 3.11 and 3.12 on Linux and macOS.

Necessary packages
install the `ale-py` package distributed via PyPI:
```
pip install ale-py
```
Install the gymnasium library with atari dependencies:
```
pip install gymnasium[atari]
```



