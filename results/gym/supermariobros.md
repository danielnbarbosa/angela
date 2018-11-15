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
Episode:   100   Avg:   57.426   BestAvg:     -inf   σ:   49.092  |  Steps:   141056   Secs:   1116      |
Episode:   200   Avg:   67.045   BestAvg:   67.587   σ:   45.811  |  Steps:   207250   Secs:   1949      |
Episode:   300   Avg:   61.996   BestAvg:   69.459   σ:   35.855  |  Steps:   244773   Secs:   2668      |
Episode:   400   Avg:   66.301   BestAvg:   69.459   σ:   45.672  |  Steps:   280524   Secs:   3376      |
Episode:   500   Avg:   66.696   BestAvg:   69.459   σ:   39.782  |  Steps:   308935   Secs:   4057      |
Episode:   600   Avg:   70.953   BestAvg:   72.852   σ:   41.188  |  Steps:   336773   Secs:   4731      |
Episode:   700   Avg:   70.206   BestAvg:   75.710   σ:   43.228  |  Steps:   361465   Secs:   5394      |
Episode:   800   Avg:   65.408   BestAvg:   75.710   σ:   36.269  |  Steps:   385260   Secs:   6055      |
Episode:   900   Avg:   68.019   BestAvg:   75.710   σ:   46.118  |  Steps:   407597   Secs:   6711      |
Episode:  1000   Avg:   65.368   BestAvg:   75.710   σ:   42.812  |  Steps:   429653   Secs:   7369      |
```
