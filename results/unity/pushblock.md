#### Notes
- for training max_t is 200, for evaluation max_t is 1000.
- learning still not robust and is prone to collapse.
- had to do a lot of hacking to get multi-agent unity environment to work with PPO.  could be bugs.
- tested fc_units = 32, 64, 128, 64 worked best.
- tested lr = 0.001, 0.0005, 0.0005 had worse training score but better evaluation score.
- agent behavior is not smooth and deliberate, maybe needs to train longer
- should really run evaluation for 100 episodes


#### PPO Training
```
Episode:   100   Avg:     0.44   BestAvg:     -inf   σ:     0.65  |  Steps:    19900   Secs:    417      |  ε: 0.09048   β:      0.0
Episode:   200   Avg:     1.19   BestAvg:     1.25   σ:     0.63  |  Steps:    39800   Secs:    970      |  ε: 0.08186   β:      0.0
```

#### PPO Evaluation
```
Loaded: checkpoints/last_run/episode.100.pth
Episode:    10   Avg:     2.98   BestAvg:     -inf   σ:     0.58  |  Steps:      999   Reward:     3.41  |  ε: 0.0991   β:      0.0
```
