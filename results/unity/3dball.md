#### Notes
- set evaluation_only=True during evaluation
- setting weight_decay=0.0 seemed to be key to seeing learning
- should really evaluate for 100 episodes


#### DDPG Training
```
Episode:   100   Avg:     0.55   BestAvg:     -inf   σ:     0.09  |  Steps:     1883   Secs:     39      |  ⍺: 0.5000  Buffer:  23796
Episode:   200   Avg:     0.53   BestAvg:     0.55   σ:     0.05  |  Steps:     3720   Secs:     77      |  ⍺: 0.5000  Buffer:  47040
Episode:   300   Avg:     0.54   BestAvg:     0.55   σ:     0.04  |  Steps:     5582   Secs:    119      |  ⍺: 0.5000  Buffer:  70584
Episode:   400   Avg:     0.53   BestAvg:     0.55   σ:     0.05  |  Steps:     7437   Secs:    157      |  ⍺: 0.5000  Buffer:  94044
Episode:   500   Avg:     0.57   BestAvg:     0.57   σ:     0.07  |  Steps:     9341   Secs:    198      |  ⍺: 0.5000  Buffer: 100000
Episode:   600   Avg:     0.63   BestAvg:     0.63   σ:     0.06  |  Steps:    11282   Secs:    238      |  ⍺: 0.5000  Buffer: 100000
Episode:   700   Avg:     0.65   BestAvg:     0.66   σ:     0.05  |  Steps:    13249   Secs:    279      |  ⍺: 0.5000  Buffer: 100000
Episode:   800   Avg:     0.68   BestAvg:     0.68   σ:     0.06  |  Steps:    15249   Secs:    321      |  ⍺: 0.5000  Buffer: 100000
Episode:   900   Avg:     0.67   BestAvg:     0.69   σ:     0.07  |  Steps:    17278   Secs:    363      |  ⍺: 0.5000  Buffer: 100000
Episode:  1000   Avg:     0.67   BestAvg:     0.69   σ:     0.07  |  Steps:    19289   Secs:    404      |  ⍺: 0.5000  Buffer: 100000
Episode:  1100   Avg:     0.68   BestAvg:     0.69   σ:     0.07  |  Steps:    21307   Secs:    447      |  ⍺: 0.5000  Buffer: 100000
Episode:  1200   Avg:     0.69   BestAvg:     0.69   σ:     0.07  |  Steps:    23339   Secs:    490      |  ⍺: 0.5000  Buffer: 100000
Episode:  1300   Avg:     0.69   BestAvg:     0.69   σ:     0.06  |  Steps:    25362   Secs:    533      |  ⍺: 0.5000  Buffer: 100000
Episode:  1400   Avg:     0.68   BestAvg:     0.70   σ:     0.10  |  Steps:    27414   Secs:    575      |  ⍺: 0.5000  Buffer: 100000
Episode:  1500   Avg:     0.64   BestAvg:     0.70   σ:     0.06  |  Steps:    29422   Secs:    615      |  ⍺: 0.5000  Buffer: 100000
Episode:  1600   Avg:     0.63   BestAvg:     0.70   σ:     0.07  |  Steps:    31408   Secs:    655      |  ⍺: 0.5000  Buffer: 100000
Episode:  1700   Avg:     2.08   BestAvg:     2.08   σ:     1.37  |  Steps:    37260   Secs:    767      |  ⍺: 0.5000  Buffer: 100000
Episode:  1800   Avg:    43.91   BestAvg:    43.91   σ:    31.92  |  Steps:   109169   Secs:   2270      |  ⍺: 0.5000  Buffer: 100000
Episode:  1900   Avg:    30.38   BestAvg:    48.72   σ:    27.07  |  Steps:   164481   Secs:   3621      |  ⍺: 0.5000  Buffer: 100000
```

#### DDPG Evaluation
```
Loaded: checkpoints/last_run/episode.1900
Episode:    10   Avg:   100.00   BestAvg:     -inf   σ:     0.00  |  Steps:      999   Reward:   100.00  |  ⍺: 0.5000  Buffer: 100000
```
