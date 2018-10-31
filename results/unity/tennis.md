#### Notes
- removing L2 weight decay helped a lot
- concatenating actions at input layer (vs 1st hidden layer) also helped
- training with same code is much better on AWS vs laptop, probably just due to differences in random seed


#### MADDPGv1 Training (laptop)
```
Episode:   100   Avg:   -0.004   BestAvg:     -inf   σ:    0.005  |  Steps:     1373   Secs:     23      |  ⍺: 0.5000  Buffer:   1473
Episode:   200   Avg:    0.003   BestAvg:    0.003   σ:    0.024  |  Steps:     3039   Secs:     49      |  ⍺: 0.5000  Buffer:   3239
Episode:   300   Avg:   -0.005   BestAvg:    0.003   σ:    0.000  |  Steps:     4362   Secs:     72      |  ⍺: 0.5000  Buffer:   4662
Episode:   400   Avg:   -0.005   BestAvg:    0.003   σ:    0.000  |  Steps:     5687   Secs:     97      |  ⍺: 0.5000  Buffer:   6087
Episode:   500   Avg:   -0.005   BestAvg:    0.003   σ:    0.000  |  Steps:     7008   Secs:    121      |  ⍺: 0.5000  Buffer:   7508
Episode:   600   Avg:    0.007   BestAvg:    0.007   σ:    0.024  |  Steps:     8794   Secs:    152      |  ⍺: 0.5000  Buffer:   9394
Episode:   700   Avg:   -0.001   BestAvg:    0.009   σ:    0.014  |  Steps:    10254   Secs:    180      |  ⍺: 0.5000  Buffer:  10954
Episode:   800   Avg:    0.004   BestAvg:    0.009   σ:    0.020  |  Steps:    11892   Secs:    213      |  ⍺: 0.5000  Buffer:  12692
Episode:   900   Avg:    0.019   BestAvg:    0.019   σ:    0.036  |  Steps:    14048   Secs:    256      |  ⍺: 0.5000  Buffer:  14948
Episode:  1000   Avg:   -0.004   BestAvg:    0.019   σ:    0.007  |  Steps:    15402   Secs:    284      |  ⍺: 0.5000  Buffer:  16402
Episode:  1100   Avg:   -0.005   BestAvg:    0.019   σ:    0.000  |  Steps:    16722   Secs:    314      |  ⍺: 0.5000  Buffer:  17822
Episode:  1200   Avg:   -0.003   BestAvg:    0.019   σ:    0.009  |  Steps:    18091   Secs:    346      |  ⍺: 0.5000  Buffer:  19291
Episode:  1300   Avg:    0.012   BestAvg:    0.019   σ:    0.037  |  Steps:    20028   Secs:    389      |  ⍺: 0.5000  Buffer:  21328
Episode:  1400   Avg:    0.008   BestAvg:    0.022   σ:    0.028  |  Steps:    21789   Secs:    429      |  ⍺: 0.5000  Buffer:  23189
Episode:  1500   Avg:    0.001   BestAvg:    0.022   σ:    0.016  |  Steps:    23343   Secs:    465      |  ⍺: 0.5000  Buffer:  24843
Episode:  1600   Avg:    0.008   BestAvg:    0.022   σ:    0.029  |  Steps:    25152   Secs:    506      |  ⍺: 0.5000  Buffer:  26752
Episode:  1700   Avg:    0.007   BestAvg:    0.022   σ:    0.022  |  Steps:    26914   Secs:    546      |  ⍺: 0.5000  Buffer:  28614
Episode:  1800   Avg:    0.010   BestAvg:    0.022   σ:    0.025  |  Steps:    28789   Secs:    588      |  ⍺: 0.5000  Buffer:  30589
Episode:  1900   Avg:    0.033   BestAvg:    0.033   σ:    0.027  |  Steps:    31409   Secs:    650      |  ⍺: 0.5000  Buffer:  33309
Episode:  2000   Avg:    0.089   BestAvg:    0.089   σ:    0.082  |  Steps:    36179   Secs:    757      |  ⍺: 0.5000  Buffer:  38179
Episode:  2100   Avg:    0.065   BestAvg:    0.091   σ:    0.048  |  Steps:    40047   Secs:    882      |  ⍺: 0.5000  Buffer:  42147
Episode:  2200   Avg:    0.092   BestAvg:    0.094   σ:    0.074  |  Steps:    45449   Secs:    998      |  ⍺: 0.5000  Buffer:  47649
Episode:  2300   Avg:    0.097   BestAvg:    0.101   σ:    0.120  |  Steps:    50696   Secs:   1109      |  ⍺: 0.5000  Buffer:  52996
Episode:  2400   Avg:    0.307   BestAvg:    0.307   σ:    0.465  |  Steps:    63902   Secs:   1387      |  ⍺: 0.5000  Buffer:  66302
Episode:  2450   Avg:    0.501   BestAvg:    0.501   σ:    0.578  |  Steps:    75115   Secs:   1614      |  ⍺: 0.5000  Buffer:  77565

Solved in 2350 episodes!
```

#### MADDPGv1 Evaluation (laptop)
```
Episode:    25   Avg:    2.520   BestAvg:     -inf   σ:    0.496  |  Steps:      999   Reward:    2.650  |  ⍺: 0.5000  Buffer:  24056
```



#### MADDPGv2 Training (AWS)
```
Episode:   100   Avg:    0.000   BestAvg:     -inf   σ:    0.000  |  Steps:     1341   Secs:     12      |  Buffer:   1441   NoiseW: 1.0
Episode:   200   Avg:    0.001   BestAvg:    0.001   σ:    0.010  |  Steps:     2678   Secs:     26      |  Buffer:   2878   NoiseW: 1.0
Episode:   300   Avg:    0.000   BestAvg:    0.001   σ:    0.000  |  Steps:     4001   Secs:     39      |  Buffer:   4301   NoiseW: 1.0
Episode:   400   Avg:    0.000   BestAvg:    0.001   σ:    0.000  |  Steps:     5323   Secs:     52      |  Buffer:   5723   NoiseW: 1.0
Episode:   500   Avg:    0.024   BestAvg:    0.024   σ:    0.043  |  Steps:     7342   Secs:     72      |  Buffer:   7842   NoiseW: 1.0
Episode:   600   Avg:    0.019   BestAvg:    0.034   σ:    0.039  |  Steps:     9128   Secs:     89      |  Buffer:   9728   NoiseW: 1.0
Episode:   700   Avg:    0.006   BestAvg:    0.034   σ:    0.023  |  Steps:    10619   Secs:    104      |  Buffer:  11319   NoiseW: 1.0
Episode:   800   Avg:    0.010   BestAvg:    0.034   σ:    0.030  |  Steps:    12176   Secs:    120      |  Buffer:  12976   NoiseW: 1.0
Episode:   900   Avg:    0.024   BestAvg:    0.034   σ:    0.043  |  Steps:    13901   Secs:    137      |  Buffer:  14801   NoiseW: 1.0
Episode:  1000   Avg:    0.008   BestAvg:    0.034   σ:    0.027  |  Steps:    15365   Secs:    151      |  Buffer:  16365   NoiseW: 1.0
Episode:  1100   Avg:    0.021   BestAvg:    0.034   σ:    0.039  |  Steps:    17073   Secs:    168      |  Buffer:  18173   NoiseW: 1.0
Episode:  1200   Avg:    0.011   BestAvg:    0.034   σ:    0.031  |  Steps:    18658   Secs:    184      |  Buffer:  19858   NoiseW: 1.0
Episode:  1300   Avg:    0.003   BestAvg:    0.034   σ:    0.017  |  Steps:    20047   Secs:    198      |  Buffer:  21347   NoiseW: 1.0
Episode:  1400   Avg:    0.018   BestAvg:    0.034   σ:    0.040  |  Steps:    21735   Secs:    215      |  Buffer:  23135   NoiseW: 1.0
Episode:  1500   Avg:    0.063   BestAvg:    0.063   σ:    0.046  |  Steps:    24302   Secs:    240      |  Buffer:  25802   NoiseW: 1.0
Episode:  1600   Avg:    0.066   BestAvg:    0.070   σ:    0.046  |  Steps:    26996   Secs:    266      |  Buffer:  28596   NoiseW: 1.0
Episode:  1700   Avg:    0.084   BestAvg:    0.084   σ:    0.043  |  Steps:    29847   Secs:    293      |  Buffer:  30000   NoiseW: 1.0
Episode:  1800   Avg:    0.069   BestAvg:    0.086   σ:    0.045  |  Steps:    32550   Secs:    320      |  Buffer:  30000   NoiseW: 1.0
Episode:  1900   Avg:    0.067   BestAvg:    0.086   σ:    0.046  |  Steps:    35083   Secs:    344      |  Buffer:  30000   NoiseW: 1.0
Episode:  2000   Avg:    0.087   BestAvg:    0.087   σ:    0.031  |  Steps:    37937   Secs:    372      |  Buffer:  30000   NoiseW: 1.0
Episode:  2100   Avg:    0.121   BestAvg:    0.121   σ:    0.119  |  Steps:    42193   Secs:    413      |  Buffer:  30000   NoiseW: 1.0
Episode:  2200   Avg:    0.146   BestAvg:    0.149   σ:    0.118  |  Steps:    47863   Secs:    467      |  Buffer:  30000   NoiseW: 1.0
Episode:  2300   Avg:    0.183   BestAvg:    0.185   σ:    0.156  |  Steps:    55161   Secs:    536      |  Buffer:  30000   NoiseW: 1.0
Episode:  2400   Avg:    0.151   BestAvg:    0.192   σ:    0.112  |  Steps:    61107   Secs:    592      |  Buffer:  30000   NoiseW: 1.0
Episode:  2500   Avg:    0.449   BestAvg:    0.449   σ:    0.557  |  Steps:    78769   Secs:    757      |  Buffer:  30000   NoiseW: 1.0
Episode:  2531   Avg:    0.501   BestAvg:    0.501   σ:    0.537  |  Steps:    85312   Secs:    818      |  Buffer:  30000   NoiseW: 1.0

Solved in 2431 episodes!
```
