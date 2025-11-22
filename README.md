# THIX: THICK world models in JAX

Condor command
```
python run_condor.py results --config condor_cfg.yaml --env_name crafter --condor --num_trials 5 --wandb_mode online --use_gpu true  --project_name hrl --wandb_use true
```


A reimplementation of [THICK][paper], our algorithm for learning hierarchical world models with adaptive temporal abstractions, in JAX based on [DreamerV3][dreamerv3].

<p align="center">
  <img src="docs/thix.gif" alt="THICK overview" width="70%">
</p>

## Installation

The code has been tested on Linux and Mac and requires Python 3.11+.

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r embodied/requirements.txt
pip install -U -r dreamerv3/requirements.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Running experiments

Example training script for running DreamerV3 with a THICK world model

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs crafter thick \
  --loss_scales.sparse 1.0
```

The `thick` config is needed for a THICK world model. 
Otherwise a flat RSSM is used. 

THICK uses one hyperparameter $\beta^{\mathrm{sparse}}$, or `loss_scales.sparse`, which scales the sparsity regularization term in the loss function. 
This controls the frequency of context changes and the granularity of temporal abstractions.
Typically, this hyperparameter needs to be tuned for every environment or suite. 
We provide a [short tutorial](tune_sparsity.md) for determining $\beta^{\mathrm{sparse}}$ for new environments.

The following settings have been tested:

**Atari**:
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs atari100k thick size25m\
  --task atari100k_qbert
  --loss_scales.sparse 10.0
  --run.steps 1e6
```

**Pokémon Red**: For running Pokémon Red, you need to copy your legally obtained Pokémon Red ROM into the `embodied/envs/pokemon_emulator/` directory. 
You can find the ROM using Google. 
It should be 1MB of size. 
Rename it to `PokemonRed.gb`.

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs pokemon thick size25m\
  --loss_scales.sparse 1.0
  --run.steps 1e6
```

See the [DreamerV3][dreamerv3] code base for more tips on running Dreamer and [our paper][paper] for details on THICK. 

## Acknowledgements

The code was primarily developed by [Tomáš Daniš][TomasDanis], with support from [Christian Gumbsch][ChristianGumbsch]. This code was developed based on the [DreamerV3][dreamerv3] code base. From the [SENSEI][sensei] code base we took the modified version of [PokeGym][pokegym]. 
We provide the Pokémon game states from [AI plays Pokemon][AIplaysPokemon] . 


## Citation

```
@inproceedings{gumbsch2024thick,
title={Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics},
author={Christian Gumbsch and Noor Sajid and Georg Martius and Martin V. Butz},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=TjCDNssXKU}
}
```


[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://openreview.net/pdf?id=TjCDNssXKU
[dreamerv3]: https://github.com/danijar/dreamerv3
[sensei]: https://github.com/martius-lab/sensei
[pokegym]: https://pypi.org/project/pokegym/
[AIplaysPokemon]: https://github.com/PWhiddy/PokemonRedExperiments
[TomasDanis]: https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/distributed-intelligence/team/tomas-danis/
[ChristianGumbsch]: https://cgumbsch.github.io
