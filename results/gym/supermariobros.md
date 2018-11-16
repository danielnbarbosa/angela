#### Notes

- Use the meta version (`ppaquette/meta-SuperMarioBros-Tiles-v0`) of the Super Mario environments to avoid environment restarting on death.  Instead just reset environment.
- Set the max timesteps to beyond the level timeout (~4000 steps) to avoid episode terminating without death, otherwise it gets frozen.
- The environment expects 6 different simultaneous inputs, but many combinations are not useful (e.g. left and right) so the action_map defines only the useful combinations.
- Random agent averages score of 40 over 10 rollouts.
- Environment does not accept a seed but is deterministic.
- Increasing frameskip makes the emulator skip rendering frames.  This only affects rendering, not training.
- For some reason a huge reward is given on the first step, distorting the reward function.  This is now removed in the training code.

- Training collapsed after 300 episodes.  When evaluating the model from the 200th episode, agent got to distance 1410 (44%).


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

#### PPO (AWS)
```
Episode:   100   Avg:   63.101   BestAvg:     -inf   σ:   36.236  |  Steps:   178785   Secs:   1940      |  ε:  0.181   β:      0.0
Episode:   200   Avg:   62.737   BestAvg:   69.395   σ:   31.651  |  Steps:   232540   Secs:   3308      |  ε: 0.1637   β:      0.0
Episode:   300   Avg:   65.655   BestAvg:   69.395   σ:   29.590  |  Steps:   273889   Secs:   4621      |  ε: 0.1481   β:      0.0
Episode:   400   Avg:   63.703   BestAvg:   69.498   σ:   31.058  |  Steps:   311513   Secs:   5918      |  ε:  0.134   β:      0.0
```

#### PG (AWS)
```
Episode:   100   Avg:   60.373   BestAvg:     -inf   σ:   38.593  |  Steps:    81288   Secs:    891      |
Episode:   200   Avg:   64.362   BestAvg:   66.800   σ:   45.062  |  Steps:   109424   Secs:   1569      |
Episode:   300   Avg:   67.957   BestAvg:   70.015   σ:   40.559  |  Steps:   132870   Secs:   2227      |
Episode:   400   Avg:   60.115   BestAvg:   70.015   σ:   36.937  |  Steps:   150324   Secs:   2865      |
Episode:   500   Avg:   69.814   BestAvg:   70.957   σ:   38.495  |  Steps:   170104   Secs:   3509      |
Episode:   600   Avg:   66.171   BestAvg:   73.015   σ:   46.310  |  Steps:   195169   Secs:   4173      |
Episode:   700   Avg:   73.212   BestAvg:   73.212   σ:   46.376  |  Steps:   218736   Secs:   4834      |
Episode:   800   Avg:   81.553   BestAvg:   83.595   σ:   40.302  |  Steps:   245588   Secs:   5509      |
Episode:   900   Avg:   75.225   BestAvg:   83.595   σ:   42.320  |  Steps:   265583   Secs:   6157      |
Episode:  1000   Avg:   83.389   BestAvg:   83.595   σ:   42.853  |  Steps:   288035   Secs:   6813      |
Episode:  1050   Avg:   79.612   BestAvg:   84.936   σ:   42.564  |  Steps:      171   Reward:   66.184  |
```
