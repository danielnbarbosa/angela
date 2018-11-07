#### Notes
- HalfCheetah-v1 is considered "solved" when the agent obtains an average reward of at least 4800.0 over 100 consecutive episodes.  Assuming the same for HalfCheetah-v2
- with a high max_t (200) the cheetah flips over on it's back, yet still manages and impressive eval score of 1191


#### DDPG Training
```
Episode:   100   Avg:  -25.053   BestAvg:     -inf   σ:   19.299  |  Steps:    20000   Secs:    181      |  ⍺: 0.5000  Buffer:  20000
Episode:   200   Avg:   74.266   BestAvg:   74.266   σ:   43.034  |  Steps:    40000   Secs:    405      |  ⍺: 0.5000  Buffer:  40000
Episode:   300   Avg:  157.333   BestAvg:  157.333   σ:   26.721  |  Steps:    60000   Secs:    651      |  ⍺: 0.5000  Buffer:  60000
Episode:   400   Avg:  172.069   BestAvg:  172.309   σ:   29.055  |  Steps:    80000   Secs:    888      |  ⍺: 0.5000  Buffer:  80000
```

#### DDPG Evaluation
```
Loaded: checkpoints/last_run/episode.400
Episode:    20   Avg: 1191.915   BestAvg:     -inf   σ:   46.429  |  Steps:     1000   Reward: 1143.694  |  ⍺: 0.5000  Buffer:  20000
```
