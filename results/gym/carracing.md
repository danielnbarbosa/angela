#### Notes
- CarRacing-v0 is considered solved when the agent obtains an average reward of at least 900 over 100 consecutive episodes.
- This was my first try on this environment.
- Training was done for ~12 hours on a p2.xlarge with `max_t = 300`, `eps_start = 0.9`, `eps_end = 0.1`
- Evaluation is done with with `max_t = 1000`, `eps_start = 0.0`, `eps_end = 0.0`


#### DQN Training
```
Episode:     1   Avg:  -13.221   BestAvg:     -inf   σ:    0.000  |  Steps:      300   Reward:  -13.221  |  ε: 0.8991  ⍺: 0.5000  Buffer:    300
Episode:   100   Avg:   -8.516   BestAvg:     -inf   σ:    3.904  |  Steps:    30000   Secs:   1252      |  ε: 0.8143  ⍺: 0.5000  Buffer:  30000
Episode:   200   Avg:   -4.694   BestAvg:   -4.694   σ:    3.752  |  Steps:    60000   Secs:   2492      |  ε: 0.7368  ⍺: 0.5000  Buffer:  60000
Episode:   300   Avg:    0.074   BestAvg:    0.074   σ:    5.310  |  Steps:    90000   Secs:   3717      |  ε: 0.6666  ⍺: 0.5000  Buffer:  90000
Episode:   400   Avg:   -1.210   BestAvg:    0.672   σ:    4.840  |  Steps:   120000   Secs:   4916      |  ε: 0.6032  ⍺: 0.5000  Buffer: 100000
Episode:   500   Avg:   -2.420   BestAvg:    0.672   σ:    6.975  |  Steps:   150000   Secs:   6009      |  ε: 0.5457  ⍺: 0.5000  Buffer: 100000
Episode:  1000   Avg:   -5.640   BestAvg:    0.672   σ:   10.201  |  Steps:   300000   Secs:  11987      |  ε: 0.3309  ⍺: 0.5000  Buffer: 100000
Episode:  1500   Avg:   26.582   BestAvg:   26.582   σ:   18.306  |  Steps:   450000   Secs:  18159      |  ε: 0.2007  ⍺: 0.5000  Buffer: 100000
Episode:  2000   Avg:  114.991   BestAvg:  114.991   σ:   40.987  |  Steps:   600000   Secs:  24448      |  ε: 0.1217  ⍺: 0.5000  Buffer: 100000
Episode:  2500   Avg:  170.471   BestAvg:  174.837   σ:   51.593  |  Steps:   749923   Secs:  30749      |  ε: 0.1000  ⍺: 0.5000  Buffer: 100000
Episode:  3000   Avg:  180.007   BestAvg:  184.990   σ:   50.440  |  Steps:   899730   Secs:  37112      |  ε: 0.1000  ⍺: 0.5000  Buffer: 100000
Episode:  3500   Avg:  163.932   BestAvg:  184.990   σ:   51.026  |  Steps:  1049654   Secs:  43565      |  ε: 0.1000  ⍺: 0.5000  Buffer: 100000
```

#### DQN Evaluation
```
Loaded: checkpoints/carracing/episode.3000
Episode:     1   Avg:  678.523   BestAvg:     -inf   σ:    0.000  |  Steps:     1000   Reward:  678.523  |  ε: 0.0000  ⍺: 0.5000  Buffer:   1000
Episode:     2   Avg:  687.838   BestAvg:     -inf   σ:    9.315  |  Steps:     1000   Reward:  697.153  |  ε: 0.0000  ⍺: 0.5000  Buffer:   2000
Episode:     3   Avg:  697.811   BestAvg:     -inf   σ:   16.024  |  Steps:      847   Reward:  717.757  |  ε: 0.0000  ⍺: 0.5000  Buffer:   2847
Episode:     4   Avg:  580.920   BestAvg:     -inf   σ:  202.936  |  Steps:     1000   Reward:  230.247  |  ε: 0.0000  ⍺: 0.5000  Buffer:   3847
Episode:     5   Avg:  510.970   BestAvg:     -inf   σ:  229.169  |  Steps:     1000   Reward:  231.169  |  ε: 0.0000  ⍺: 0.5000  Buffer:   4847
```
