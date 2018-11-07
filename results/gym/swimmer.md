#### Notes
- back joint quickly learns swimming motion but front joint gets locked up and stays that way
- not sure if it was increased sigma, larger max_t or larger buffer_size but front joint eventually started doing a little better around episode 2300


#### DDPG Training
```
Episode:   100   Avg:   30.159   BestAvg:     -inf   σ:    7.136  |  Steps:    30000   Secs:     83      |  ⍺: 0.5000  Buffer:  30000
Episode:   200   Avg:   32.646   BestAvg:   32.646   σ:    2.784  |  Steps:    60000   Secs:    197      |  ⍺: 0.5000  Buffer:  60000
Episode:   300   Avg:   32.727   BestAvg:   33.455   σ:    5.567  |  Steps:    90000   Secs:    286      |  ⍺: 0.5000  Buffer:  90000
Episode:   400   Avg:   23.325   BestAvg:   33.455   σ:   11.293  |  Steps:   120000   Secs:    381      |  ⍺: 0.5000  Buffer: 120000
Episode:   500   Avg:   26.918   BestAvg:   33.455   σ:    7.683  |  Steps:   150000   Secs:    475      |  ⍺: 0.5000  Buffer: 150000
Episode:   600   Avg:   36.369   BestAvg:   36.369   σ:    4.344  |  Steps:   180000   Secs:    573      |  ⍺: 0.5000  Buffer: 180000
Episode:   700   Avg:   49.781   BestAvg:   49.781   σ:    8.433  |  Steps:   210000   Secs:    668      |  ⍺: 0.5000  Buffer: 200000
Episode:   800   Avg:   38.223   BestAvg:   50.951   σ:   12.348  |  Steps:   240000   Secs:    767      |  ⍺: 0.5000  Buffer: 200000
Episode:   900   Avg:   42.140   BestAvg:   50.951   σ:    8.509  |  Steps:   270000   Secs:    865      |  ⍺: 0.5000  Buffer: 200000
Episode:  1000   Avg:   44.246   BestAvg:   50.951   σ:   10.214  |  Steps:   300000   Secs:    958      |  ⍺: 0.5000  Buffer: 200000
Episode:  1100   Avg:   42.726   BestAvg:   50.951   σ:    6.491  |  Steps:   330000   Secs:   1062      |  ⍺: 0.5000  Buffer: 200000
Episode:  1200   Avg:   49.820   BestAvg:   50.951   σ:    6.943  |  Steps:   360000   Secs:   1160      |  ⍺: 0.5000  Buffer: 200000
Episode:  1300   Avg:   52.605   BestAvg:   52.841   σ:    6.513  |  Steps:   390000   Secs:   1267      |  ⍺: 0.5000  Buffer: 200000
Episode:  1400   Avg:   53.018   BestAvg:   54.419   σ:    5.827  |  Steps:   420000   Secs:   1367      |  ⍺: 0.5000  Buffer: 200000
Episode:  1500   Avg:   49.672   BestAvg:   54.419   σ:    5.547  |  Steps:   450000   Secs:   1464      |  ⍺: 0.5000  Buffer: 200000
Episode:  1600   Avg:   50.089   BestAvg:   54.419   σ:    5.563  |  Steps:   480000   Secs:   1559      |  ⍺: 0.5000  Buffer: 200000
Episode:  1700   Avg:   54.064   BestAvg:   54.436   σ:    4.841  |  Steps:   510000   Secs:   1654      |  ⍺: 0.5000  Buffer: 200000
Episode:  1800   Avg:   55.615   BestAvg:   55.615   σ:    4.505  |  Steps:   540000   Secs:   1750      |  ⍺: 0.5000  Buffer: 200000
Episode:  1900   Avg:   57.885   BestAvg:   58.042   σ:    4.983  |  Steps:   570000   Secs:   1848      |  ⍺: 0.5000  Buffer: 200000
Episode:  2000   Avg:   58.641   BestAvg:   58.641   σ:    5.977  |  Steps:   600000   Secs:   1947      |  ⍺: 0.5000  Buffer: 200000
Episode:  2100   Avg:   56.502   BestAvg:   59.108   σ:    9.312  |  Steps:   630000   Secs:   2045      |  ⍺: 0.5000  Buffer: 200000
Episode:  2200   Avg:   58.002   BestAvg:   59.108   σ:   10.344  |  Steps:   660000   Secs:   2142      |  ⍺: 0.5000  Buffer: 200000
Episode:  2300   Avg:   58.059   BestAvg:   59.108   σ:    9.769  |  Steps:   690000   Secs:   2240      |  ⍺: 0.5000  Buffer: 200000
Episode:  2400   Avg:   62.016   BestAvg:   62.016   σ:    8.548  |  Steps:   720000   Secs:   2351      |  ⍺: 0.5000  Buffer: 200000
Episode:  2500   Avg:   61.921   BestAvg:   63.565   σ:    7.536  |  Steps:   750000   Secs:   2449      |  ⍺: 0.5000  Buffer: 200000
Episode:  2600   Avg:   59.115   BestAvg:   63.565   σ:    8.230  |  Steps:   780000   Secs:   2565      |  ⍺: 0.5000  Buffer: 200000
Episode:  2700   Avg:   60.958   BestAvg:   63.565   σ:    7.935  |  Steps:   810000   Secs:   2697      |  ⍺: 0.5000  Buffer: 200000
Episode:  2800   Avg:   59.933   BestAvg:   63.565   σ:    9.191  |  Steps:   840000   Secs:   2821      |  ⍺: 0.5000  Buffer: 200000
```

#### DDPG Evaluation
```
Loaded: checkpoints/last_run/episode.2300
Episode:    25   Avg:  161.927   BestAvg:     -inf   σ:    7.206  |  Steps:     1000   Reward:  160.191  |  ⍺: 0.5000  Buffer:  25000
```
