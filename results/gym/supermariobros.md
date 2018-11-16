#### Notes

- Use the meta version (`ppaquette/meta-SuperMarioBros-Tiles-v0`) of the Super Mario environments to avoid environment restarting on death.  Instead just reset environment.
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
Episode:   100   Avg:   76.631   BestAvg:     -inf   σ:   43.971  |  Steps:    22438   Secs:    690      |
Episode:   200   Avg:   81.307   BestAvg:   87.884   σ:   45.704  |  Steps:    42617   Secs:   1371      |
Episode:   300   Avg:   81.837   BestAvg:   87.884   σ:   45.200  |  Steps:    64049   Secs:   2062      |
Episode:   400   Avg:   78.041   BestAvg:   87.884   σ:   45.575  |  Steps:    83733   Secs:   2746      |
Episode:   500   Avg:   77.292   BestAvg:   87.884   σ:   40.995  |  Steps:   103037   Secs:   3426      |
Episode:   600   Avg:   96.604   BestAvg:   97.911   σ:   47.829  |  Steps:   126027   Secs:   4125      |
Episode:   700   Avg:   95.441   BestAvg:   98.294   σ:   50.705  |  Steps:   148457   Secs:   4824      |
Episode:   800   Avg:   60.608   BestAvg:   98.384   σ:   39.045  |  Steps:   174980   Secs:   5541      |
Episode:   900   Avg:   61.228   BestAvg:   98.384   σ:   25.987  |  Steps:   206496   Secs:   6282      |
Episode:  1000   Avg:   64.467   BestAvg:   98.384   σ:   23.401  |  Steps:   234936   Secs:   7008      |
```

#### PG Evaluation
```
Loaded: checkpoints/supermariobros_pg/episode.700.pth
Episode:     1   Avg:   28.028   BestAvg:     -inf   σ:    0.000  |  Steps:       57   Reward:   28.028  |
Episode:     2   Avg:   47.720   BestAvg:     -inf   σ:   19.692  |  Steps:      154   Reward:   67.412  |
Episode:     3   Avg:   54.420   BestAvg:     -inf   σ:   18.663  |  Steps:      151   Reward:   67.821  |
Episode:     4   Avg:   57.464   BestAvg:     -inf   σ:   17.000  |  Steps:      141   Reward:   66.593  |
Episode:     5   Avg:   59.105   BestAvg:     -inf   σ:   15.556  |  Steps:      146   Reward:   65.673  |
Episode:     6   Avg:   90.701   BestAvg:     -inf   σ:   72.062  |  Steps:      582   Reward:  248.676  |
Episode:     7   Avg:   81.572   BestAvg:     -inf   σ:   70.364  |  Steps:       70   Reward:   26.801  |
Episode:     8   Avg:   79.623   BestAvg:     -inf   σ:   66.021  |  Steps:      174   Reward:   65.980  |
Episode:     9   Avg:   73.708   BestAvg:     -inf   σ:   64.454  |  Steps:       65   Reward:   26.392  |
Episode:    10   Avg:   77.477   BestAvg:     -inf   σ:   62.183  |  Steps:      283   Reward:  111.398  |
```
