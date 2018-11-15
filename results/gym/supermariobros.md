#### Notes

- Use the meta version (`ppaquette/meta-SuperMarioBros-Tiles-v0`) of the Super Mario environments to avoid environment restarting on death.  Instead just reset environment.
- Set the max timesteps to beyond the level timeout (~4000 steps) to avoid episode terminating without death, otherwise it gets frozen.
- The environment expects 6 different simultaneous inputs, but many combinations are not useful (e.g. left and right) so the action_map defines only the useful combinations.
- Random agent averages score of 93 over 10 rollouts.
- Environment does not accept a seed but is deterministic.
- Increasing frameskip makes the emulator skip rendering frames.  This only affects rendering, not training.
- For some reason a huge reward is given on the first step, distorting the reward function.  This is now removed in the training code.

- Training collapsed after 300 episodes.  When evaluating the model from the 200th episode, agent got to distance 1410 (44%).


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
Episode:   100   Avg:  167.091   BestAvg:     -inf   σ:   37.335  |  Steps:   242855   Secs:   3646      |  ε:  0.181   β:      0.0
Episode:   200   Avg:  205.997   BestAvg:  206.377   σ:   50.789  |  Steps:   366310   Secs:   6077      |  ε: 0.1637   β:      0.0
Episode:   300   Avg:  229.278   BestAvg:  235.858   σ:   70.386  |  Steps:   392904   Secs:   7472      |  ε: 0.1481   β:      0.0
Episode:   400   Avg:  180.349   BestAvg:  235.858   σ:   50.222  |  Steps:   410492   Secs:   8775      |  ε:  0.134   β:      0.0
Episode:   500   Avg:  175.703   BestAvg:  235.858   σ:   61.052  |  Steps:   425309   Secs:  10058      |  ε: 0.1213   β:      0.0
Episode:   600   Avg:  149.532   BestAvg:  235.858   σ:   43.417  |  Steps:   440283   Secs:  11322      |  ε: 0.1097   β:      0.0
Episode:   700   Avg:  139.562   BestAvg:  235.858   σ:   43.013  |  Steps:   454745   Secs:  12581      |  ε: 0.09928   β:      0.0
Episode:   800   Avg:  142.688   BestAvg:  235.858   σ:   39.328  |  Steps:   468515   Secs:  13852      |  ε: 0.08983   β:      0.0
Episode:   900   Avg:  134.422   BestAvg:  235.858   σ:   34.771  |  Steps:   481637   Secs:  15110      |  ε: 0.08128   β:      0.0
```
