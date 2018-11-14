#### Notes

- Use the meta version (`ppaquette/meta-SuperMarioBros-Tiles-v0`) of the Super Mario environments to avoid environment restarting on death.
- Set the max timesteps to beyond the level timeout (4000 steps) to avoid episode terminating without death, otherwise it gets frozen.
- The environment expects 6 different simultaneous inputs, but many combinations are not useful (e.g. left and right) so the action_map defines only the useful combinations.
- Currently Mario is getting stuck at the first tall pipe around distance 720.  Haven't trained for many episodes yet though.

##### Action Indicies
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
6: [0, 0, 0, 0, 0, 1]   run/fire
7: [0, 1, 0, 0, 1, 0]   left jump
8: [0, 0, 0, 1, 1, 0]   right jump
9: [0, 1, 0, 0, 0, 1]   left run
10: [0, 0, 0, 1, 0, 1]  right run
11: [0, 1, 0, 0, 1, 1]  left run jump
12: [0, 0, 0, 1, 1, 1]  right run jump
```

#### PPO
```
Episode:    88   Avg:  174.025   BestAvg:     -inf   σ:   32.066  |  Steps:     4011   Reward:  171.956  |  ε: 0.1833   β:      0.0
```
