#### Notes

- Use the meta version (`ppaquette/meta-SuperMarioBros-Tiles-v0`) of the Super Mario environments to avoid environment restarting on death.  Instead just reset environment.   However this has the unfortunate side effect of restarting Mario from the start of the last finished level vs the beginning of World 1-1.
- Set the max timesteps to beyond the level timeout (~4000 steps) to avoid episode terminating without death, otherwise it gets frozen.
- The environment expects 6 different simultaneous inputs, but many combinations are not useful (e.g. left and right) so the action_map defines only the useful combinations.
- Random agent averages score of 40 over 10 rollouts.
- Environment does not accept a seed but is deterministic.
- Increasing frameskip makes the emulator skip rendering frames.  This only affects rendering, not training.
- For some reason a huge reward is given on the first step, distorting the reward function.  This is now removed in the training code.
- All training is done on AWS.
- I think this environment is more challenging for a pure CNN model given the long time sequence of correct actions needed to be successful.  Adding an RNN would probably help.


##### Rewards Progress
```
21   9%  : first goomba
61   22% : after second pipe
76   28% : after third (first tall) pipe
112  35% : first pit
140  45% : second pit
166  51% : first turtle
```


##### Action Indices
```
0: up
1: left
2: down
3: right
4: jump
5: run (and fire)
```

##### Action Map
```
0: [0, 0, 0, 0, 0, 0]   noop
1: [1, 0, 0, 0, 0, 0]   up
2: [0, 1, 0, 0, 0, 0]   left
3: [0, 0, 1, 0, 0, 0]   down
4: [0, 0, 0, 1, 0, 0]   right
5: [0, 0, 0, 0, 1, 0]   jump
6: [0, 0, 0, 0, 0, 1]   run
7: [0, 0, 0, 0, 1, 1]   jump and run
8: [0, 1, 0, 0, 1, 0]   left jump
9: [0, 0, 0, 1, 1, 0]   right jump
10: [0, 1, 0, 0, 0, 1]   left run
11: [0, 0, 0, 1, 0, 1]  right run
12: [0, 1, 0, 0, 1, 1]  left run jump
13: [0, 0, 0, 1, 1, 1]  right run jump
```

##### Frameskip
Wall time taken for identical rollouts
```
0: 159 secs
1: 87  secs
2: 68  secs
3: 66  secs
4: 54  secs
6: 56  secs
```

#### PPO
```
Episode:   100   Avg:   63.101   BestAvg:     -inf   σ:   36.236  |  Steps:   178785   Secs:   1940      |  ε:  0.181   β:      0.0
Episode:   200   Avg:   62.737   BestAvg:   69.395   σ:   31.651  |  Steps:   232540   Secs:   3308      |  ε: 0.1637   β:      0.0
Episode:   300   Avg:   65.655   BestAvg:   69.395   σ:   29.590  |  Steps:   273889   Secs:   4621      |  ε: 0.1481   β:      0.0
Episode:   400   Avg:   63.703   BestAvg:   69.498   σ:   31.058  |  Steps:   311513   Secs:   5918      |  ε:  0.134   β:      0.0
```

#### PG
```
Episode:   100   Avg:   66.763   BestAvg:     -inf   σ:   33.226  |  Steps:    57310   Secs:    812      |
Episode:   200   Avg:   72.508   BestAvg:   76.393   σ:   40.901  |  Steps:    84111   Secs:   1499      |
Episode:   300   Avg:   74.811   BestAvg:   78.736   σ:   42.466  |  Steps:   105475   Secs:   2163      |
Episode:   400   Avg:   73.750   BestAvg:   78.736   σ:   41.295  |  Steps:   125659   Secs:   2818      |
Episode:   500   Avg:   76.567   BestAvg:   78.736   σ:   41.331  |  Steps:   145670   Secs:   3474      |
Episode:   600   Avg:   78.894   BestAvg:   80.741   σ:   39.877  |  Steps:   165625   Secs:   4131      |
Episode:   700   Avg:   87.799   BestAvg:   89.861   σ:   52.748  |  Steps:   188732   Secs:   4801      |
Episode:   800   Avg:   92.413   BestAvg:   93.600   σ:   48.748  |  Steps:   212937   Secs:   5474      |
Episode:   900   Avg:   84.696   BestAvg:   94.225   σ:   46.920  |  Steps:   233634   Secs:   6132      |
Episode:  1000   Avg:   87.311   BestAvg:   94.225   σ:   45.327  |  Steps:   254892   Secs:   6793      |
Episode:  1100   Avg:   89.698   BestAvg:   94.225   σ:   46.880  |  Steps:   277241   Secs:   7458      |
Episode:  1200   Avg:   97.582   BestAvg:  100.511   σ:   48.866  |  Steps:   300449   Secs:   8130      |
Episode:  1300   Avg:  104.066   BestAvg:  104.381   σ:   49.167  |  Steps:   325570   Secs:   8809      |
Episode:  1400   Avg:   96.025   BestAvg:  106.327   σ:   44.991  |  Steps:   347414   Secs:   9471      |
Episode:  1500   Avg:   98.833   BestAvg:  106.327   σ:   44.110  |  Steps:   370549   Secs:  10140      |
Episode:  1600   Avg:  105.485   BestAvg:  106.327   σ:   53.419  |  Steps:   394432   Secs:  10815      |
Episode:  1700   Avg:  114.549   BestAvg:  115.933   σ:   54.075  |  Steps:   420381   Secs:  11498      |
Episode:  1800   Avg:  105.741   BestAvg:  123.471   σ:   60.671  |  Steps:   445510   Secs:  12178      |
Episode:  1900   Avg:   55.687   BestAvg:  123.471   σ:   26.342  |  Steps:   474976   Secs:  12876      |
```
