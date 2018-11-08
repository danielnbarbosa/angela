#### Notes
- Reacher-v1 is considered "solved" when the agent obtains an average reward of at least -3.75 over 100 consecutive episodes.  Assuming the same for Reacher-v2.
- environment sets max_steps to 50 by default.
- key was to to reduce noise (sigma)
- agent actually does quite well after only 100 episodes of training


#### DDPG Training
```
Episode:   100   Avg:  -22.552   BestAvg:     -inf   σ:   20.903  |  Steps:     5000   Secs:     15      |  ⍺: 0.5000  Buffer:   5000
Episode:   200   Avg:  -13.267   BestAvg:  -13.043   σ:    3.151  |  Steps:    10000   Secs:     29      |  ⍺: 0.5000  Buffer:  10000
Episode:   300   Avg:  -13.268   BestAvg:  -12.777   σ:    2.998  |  Steps:    15000   Secs:     44      |  ⍺: 0.5000  Buffer:  15000
Episode:   400   Avg:  -13.358   BestAvg:  -12.777   σ:    3.173  |  Steps:    20000   Secs:     60      |  ⍺: 0.5000  Buffer:  20000
Episode:   500   Avg:  -13.090   BestAvg:  -12.777   σ:    3.260  |  Steps:    25000   Secs:     80      |  ⍺: 0.5000  Buffer:  25000
Episode:   600   Avg:  -12.735   BestAvg:  -12.671   σ:    3.254  |  Steps:    30000   Secs:     96      |  ⍺: 0.5000  Buffer:  30000
Episode:   700   Avg:  -13.102   BestAvg:  -12.643   σ:    3.218  |  Steps:    35000   Secs:    112      |  ⍺: 0.5000  Buffer:  35000
Episode:   800   Avg:  -12.150   BestAvg:  -12.150   σ:    3.142  |  Steps:    40000   Secs:    127      |  ⍺: 0.5000  Buffer:  40000
Episode:   900   Avg:  -12.142   BestAvg:  -11.878   σ:    3.117  |  Steps:    45000   Secs:    144      |  ⍺: 0.5000  Buffer:  45000
Episode:  1000   Avg:  -12.290   BestAvg:  -11.878   σ:    2.716  |  Steps:    50000   Secs:    167      |  ⍺: 0.5000  Buffer:  50000
Episode:  1100   Avg:  -11.388   BestAvg:  -11.357   σ:    2.990  |  Steps:    55000   Secs:    186      |  ⍺: 0.5000  Buffer:  55000
Episode:  1200   Avg:  -11.939   BestAvg:  -11.173   σ:    3.521  |  Steps:    60000   Secs:    207      |  ⍺: 0.5000  Buffer:  60000
Episode:  1300   Avg:  -11.391   BestAvg:  -11.173   σ:    3.032  |  Steps:    65000   Secs:    224      |  ⍺: 0.5000  Buffer:  65000
Episode:  1400   Avg:  -11.378   BestAvg:  -11.173   σ:    3.453  |  Steps:    70000   Secs:    240      |  ⍺: 0.5000  Buffer:  70000
Episode:  1500   Avg:  -11.072   BestAvg:  -10.792   σ:    3.662  |  Steps:    75000   Secs:    257      |  ⍺: 0.5000  Buffer:  75000
Episode:  1600   Avg:  -10.743   BestAvg:  -10.613   σ:    3.552  |  Steps:    80000   Secs:    273      |  ⍺: 0.5000  Buffer:  80000
Episode:  1700   Avg:  -10.978   BestAvg:  -10.485   σ:    3.601  |  Steps:    85000   Secs:    289      |  ⍺: 0.5000  Buffer:  85000
Episode:  1800   Avg:  -10.325   BestAvg:  -10.250   σ:    3.434  |  Steps:    90000   Secs:    305      |  ⍺: 0.5000  Buffer:  90000
Episode:  1900   Avg:  -10.319   BestAvg:  -10.127   σ:    4.047  |  Steps:    95000   Secs:    328      |  ⍺: 0.5000  Buffer:  95000
Episode:  2000   Avg:  -10.351   BestAvg:   -9.832   σ:    3.229  |  Steps:   100000   Secs:    345      |  ⍺: 0.5000  Buffer: 100000
Episode:  2100   Avg:   -9.950   BestAvg:   -9.669   σ:    3.254  |  Steps:   105000   Secs:    367      |  ⍺: 0.5000  Buffer: 100000
Episode:  2200   Avg:  -10.185   BestAvg:   -9.669   σ:    3.648  |  Steps:   110000   Secs:    385      |  ⍺: 0.5000  Buffer: 100000
Episode:  2300   Avg:  -10.475   BestAvg:   -9.669   σ:    3.322  |  Steps:   115000   Secs:    402      |  ⍺: 0.5000  Buffer: 100000
Episode:  2400   Avg:   -9.915   BestAvg:   -9.669   σ:    2.960  |  Steps:   120000   Secs:    419      |  ⍺: 0.5000  Buffer: 100000
Episode:  2500   Avg:   -9.543   BestAvg:   -9.495   σ:    2.544  |  Steps:   125000   Secs:    435      |  ⍺: 0.5000  Buffer: 100000
Episode:  2600   Avg:   -9.244   BestAvg:   -8.973   σ:    3.035  |  Steps:   130000   Secs:    459      |  ⍺: 0.5000  Buffer: 100000
Episode:  2700   Avg:   -9.506   BestAvg:   -8.973   σ:    2.944  |  Steps:   135000   Secs:    479      |  ⍺: 0.5000  Buffer: 100000
Episode:  2800   Avg:   -9.945   BestAvg:   -8.973   σ:    3.385  |  Steps:   140000   Secs:    502      |  ⍺: 0.5000  Buffer: 100000
Episode:  2900   Avg:   -9.571   BestAvg:   -8.973   σ:    3.271  |  Steps:   145000   Secs:    523      |  ⍺: 0.5000  Buffer: 100000
Episode:  3000   Avg:   -9.491   BestAvg:   -8.973   σ:    2.772  |  Steps:   150000   Secs:    543      |  ⍺: 0.5000  Buffer: 100000
Episode:  3100   Avg:   -9.275   BestAvg:   -8.973   σ:    2.398  |  Steps:   155000   Secs:    559      |  ⍺: 0.5000  Buffer: 100000
Episode:  3200   Avg:   -9.396   BestAvg:   -8.973   σ:    2.790  |  Steps:   160000   Secs:    580      |  ⍺: 0.5000  Buffer: 100000
Episode:  3300   Avg:   -9.262   BestAvg:   -8.973   σ:    2.802  |  Steps:   165000   Secs:    597      |  ⍺: 0.5000  Buffer: 100000
Episode:  3400   Avg:  -10.273   BestAvg:   -8.973   σ:    2.875  |  Steps:   170000   Secs:    617      |  ⍺: 0.5000  Buffer: 100000
Episode:  3500   Avg:   -9.420   BestAvg:   -8.973   σ:    2.765  |  Steps:   175000   Secs:    636      |  ⍺: 0.5000  Buffer: 100000
Episode:  3600   Avg:   -9.726   BestAvg:   -8.973   σ:    2.648  |  Steps:   180000   Secs:    654      |  ⍺: 0.5000  Buffer: 100000
Episode:  3700   Avg:   -9.199   BestAvg:   -8.973   σ:    2.522  |  Steps:   185000   Secs:    675      |  ⍺: 0.5000  Buffer: 100000
Episode:  3800   Avg:   -9.588   BestAvg:   -8.973   σ:    3.264  |  Steps:   190000   Secs:    698      |  ⍺: 0.5000  Buffer: 100000
Episode:  3900   Avg:   -9.465   BestAvg:   -8.973   σ:    3.078  |  Steps:   195000   Secs:    718      |  ⍺: 0.5000  Buffer: 100000
Episode:  4000   Avg:   -9.740   BestAvg:   -8.973   σ:    3.273  |  Steps:   200000   Secs:    733      |  ⍺: 0.5000  Buffer: 100000
Episode:  4100   Avg:   -9.608   BestAvg:   -8.973   σ:    2.527  |  Steps:   205000   Secs:    754      |  ⍺: 0.5000  Buffer: 100000
Episode:  4200   Avg:   -9.464   BestAvg:   -8.973   σ:    3.027  |  Steps:   210000   Secs:    774      |  ⍺: 0.5000  Buffer: 100000
Episode:  4300   Avg:   -9.250   BestAvg:   -8.957   σ:    2.712  |  Steps:   215000   Secs:    796      |  ⍺: 0.5000  Buffer: 100000
Episode:  4400   Avg:   -9.631   BestAvg:   -8.957   σ:    2.702  |  Steps:   220000   Secs:    824      |  ⍺: 0.5000  Buffer: 100000
Episode:  4500   Avg:   -9.778   BestAvg:   -8.957   σ:    2.511  |  Steps:   225000   Secs:    851      |  ⍺: 0.5000  Buffer: 100000
Episode:  4600   Avg:   -9.962   BestAvg:   -8.957   σ:    3.130  |  Steps:   230000   Secs:    876      |  ⍺: 0.5000  Buffer: 100000
Episode:  4700   Avg:   -9.804   BestAvg:   -8.957   σ:    2.416  |  Steps:   235000   Secs:    895      |  ⍺: 0.5000  Buffer: 100000
Episode:  4800   Avg:   -9.155   BestAvg:   -8.957   σ:    2.882  |  Steps:   240000   Secs:    915      |  ⍺: 0.5000  Buffer: 100000
Episode:  4900   Avg:   -9.950   BestAvg:   -8.957   σ:    3.401  |  Steps:   245000   Secs:    934      |  ⍺: 0.5000  Buffer: 100000
Episode:  5000   Avg:   -9.901   BestAvg:   -8.957   σ:    2.653  |  Steps:   250000   Secs:    952      |  ⍺: 0.5000  Buffer: 100000
Episode:  5100   Avg:   -9.460   BestAvg:   -8.957   σ:    2.976  |  Steps:   255000   Secs:    970      |  ⍺: 0.5000  Buffer: 100000
Episode:  5200   Avg:   -9.708   BestAvg:   -8.957   σ:    2.504  |  Steps:   260000   Secs:    994      |  ⍺: 0.5000  Buffer: 100000
```

#### DDPG Evaluation
```
Loaded: checkpoints/last_run/episode.100
Episode:   100   Avg:   -5.362   BestAvg:     -inf   σ:    2.379  |  Steps:     5000   Secs:      2      |  ⍺: 0.5000  Buffer:   5000
Episode:   200   Avg:   -4.842   BestAvg:   -4.817   σ:    2.159  |  Steps:    10000   Secs:      5      |  ⍺: 0.5000  Buffer:  10000
Episode:   300   Avg:   -5.228   BestAvg:   -4.817   σ:    1.943  |  Steps:    15000   Secs:      8      |  ⍺: 0.5000  Buffer:  15000
Episode:   400   Avg:   -5.325   BestAvg:   -4.817   σ:    2.217  |  Steps:    20000   Secs:     11      |  ⍺: 0.5000  Buffer:  20000
Episode:   500   Avg:   -4.752   BestAvg:   -4.708   σ:    2.541  |  Steps:    25000   Secs:     13      |  ⍺: 0.5000  Buffer:  25000

Loaded: checkpoints/last_run/episode.1000
Episode:   100   Avg:   -5.901   BestAvg:     -inf   σ:    2.534  |  Steps:     5000   Secs:      2      |  ⍺: 0.5000  Buffer:   5000
Episode:   200   Avg:   -5.397   BestAvg:   -5.388   σ:    2.264  |  Steps:    10000   Secs:      5      |  ⍺: 0.5000  Buffer:  10000
Episode:   300   Avg:   -5.752   BestAvg:   -5.381   σ:    2.094  |  Steps:    15000   Secs:      8      |  ⍺: 0.5000  Buffer:  15000
Episode:   400   Avg:   -5.659   BestAvg:   -5.381   σ:    2.110  |  Steps:    20000   Secs:     11      |  ⍺: 0.5000  Buffer:  20000
Episode:   500   Avg:   -5.402   BestAvg:   -5.338   σ:    2.819  |  Steps:    25000   Secs:     13      |  ⍺: 0.5000  Buffer:  25000

Loaded: checkpoints/last_run/episode.2000
Episode:   100   Avg:   -5.867   BestAvg:     -inf   σ:    2.197  |  Steps:     5000   Secs:      2      |  ⍺: 0.5000  Buffer:   5000
Episode:   200   Avg:   -5.540   BestAvg:   -5.501   σ:    2.075  |  Steps:    10000   Secs:      5      |  ⍺: 0.5000  Buffer:  10000
Episode:   300   Avg:   -5.765   BestAvg:   -5.416   σ:    1.943  |  Steps:    15000   Secs:      8      |  ⍺: 0.5000  Buffer:  15000
Episode:   400   Avg:   -5.817   BestAvg:   -5.416   σ:    2.029  |  Steps:    20000   Secs:     11      |  ⍺: 0.5000  Buffer:  20000
Episode:   500   Avg:   -5.425   BestAvg:   -5.376   σ:    2.274  |  Steps:    25000   Secs:     13      |  ⍺: 0.5000  Buffer:  25000

Loaded: checkpoints/last_run/episode.3000
Episode:   100   Avg:   -5.818   BestAvg:     -inf   σ:    1.932  |  Steps:     5000   Secs:      2      |  ⍺: 0.5000  Buffer:   5000
Episode:   200   Avg:   -5.413   BestAvg:   -5.400   σ:    1.646  |  Steps:    10000   Secs:      5      |  ⍺: 0.5000  Buffer:  10000
Episode:   300   Avg:   -5.809   BestAvg:   -5.400   σ:    1.608  |  Steps:    15000   Secs:      8      |  ⍺: 0.5000  Buffer:  15000
Episode:   400   Avg:   -5.763   BestAvg:   -5.400   σ:    1.818  |  Steps:    20000   Secs:     11      |  ⍺: 0.5000  Buffer:  20000
Episode:   500   Avg:   -5.198   BestAvg:   -5.176   σ:    1.949  |  Steps:    25000   Secs:     13      |  ⍺: 0.5000  Buffer:  25000

Loaded: checkpoints/last_run/episode.4000
Episode:   100   Avg:   -5.362   BestAvg:     -inf   σ:    2.379  |  Steps:     5000   Secs:      2      |  ⍺: 0.5000  Buffer:   5000
Episode:   200   Avg:   -4.842   BestAvg:   -4.817   σ:    2.159  |  Steps:    10000   Secs:      5      |  ⍺: 0.5000  Buffer:  10000
Episode:   300   Avg:   -5.228   BestAvg:   -4.817   σ:    1.943  |  Steps:    15000   Secs:      8      |  ⍺: 0.5000  Buffer:  15000
Episode:   400   Avg:   -5.325   BestAvg:   -4.817   σ:    2.217  |  Steps:    20000   Secs:     11      |  ⍺: 0.5000  Buffer:  20000
Episode:   500   Avg:   -4.752   BestAvg:   -4.708   σ:    2.541  |  Steps:    25000   Secs:     13      |  ⍺: 0.5000  Buffer:  25000

Loaded: checkpoints/last_run/episode.5000
Episode:   100   Avg:   -5.434   BestAvg:     -inf   σ:    1.936  |  Steps:     5000   Secs:      2      |  ⍺: 0.5000  Buffer:   5000
Episode:   200   Avg:   -5.096   BestAvg:   -5.047   σ:    1.896  |  Steps:    10000   Secs:      5      |  ⍺: 0.5000  Buffer:  10000
Episode:   300   Avg:   -5.264   BestAvg:   -5.026   σ:    1.577  |  Steps:    15000   Secs:      8      |  ⍺: 0.5000  Buffer:  15000
Episode:   400   Avg:   -5.443   BestAvg:   -5.026   σ:    1.893  |  Steps:    20000   Secs:     10      |  ⍺: 0.5000  Buffer:  20000
Episode:   500   Avg:   -4.895   BestAvg:   -4.870   σ:    2.029  |  Steps:    25000   Secs:     13      |  ⍺: 0.5000  Buffer:  25000
```
