#### Notes

- Use the meta version (`ppaquette/meta-SuperMarioBros-Tiles-v0`) of the Super Mario environments to avoid environment restarting on death.  Instead just reset environment.
- Set the max timesteps to beyond the level timeout (~4000 steps) to avoid episode terminating without death, otherwise it gets frozen.
- The environment expects 6 different simultaneous inputs, but many combinations are not useful (e.g. left and right) so the action_map defines only the useful combinations.
- Random agent averages score of 93 over 10 rollouts.
- Environment does not accept a seed but seems to be deterministic.
- Increasing frameskip makes the emulator skip rendering frames.  This does not seem to affect the frames visible by the agent.  So it's a pure speed up.

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
Episode:   100   Avg:  179.225   BestAvg:     -inf   σ:   40.327  |  Steps:   278151   Secs:   4842      |  ε:  0.181   β:      0.0
Episode:   200   Avg:  224.243   BestAvg:  224.243   σ:   53.558  |  Steps:   422388   Secs:   8101      |  ε: 0.1637   β:      0.0
Episode:   300   Avg:  236.794   BestAvg:  238.798   σ:   63.570  |  Steps:   452861   Secs:   9684      |  ε: 0.1481   β:      0.0
Episode:   400   Avg:  170.549   BestAvg:  238.798   σ:   56.988  |  Steps:   473663   Secs:  11113      |  ε:  0.134   β:      0.0
Episode:   500   Avg:  161.754   BestAvg:  238.798   σ:   42.487  |  Steps:   494595   Secs:  12543      |  ε: 0.1213   β:      0.0
Episode:   600   Avg:  173.471   BestAvg:  238.798   σ:   45.732  |  Steps:   516474   Secs:  14002      |  ε: 0.1097   β:      0.0
Episode:   700   Avg:  162.218   BestAvg:  238.798   σ:   45.134  |  Steps:   539837   Secs:  15484      |  ε: 0.09928   β:      0.0
```
