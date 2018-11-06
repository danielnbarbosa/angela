#### Notes
- agent has learned optimal behavior by episode 7100 (when evaluating without noise) but you wouldn't know this just by looking at training results


#### DDPG Training
```
Episode:   100   Avg:    8.040   BestAvg:     -inf   σ:    4.045  |  Steps:      804   Secs:      1      |  ⍺: 0.5000  Buffer:    804
Episode:   200   Avg:    8.930   BestAvg:    9.050   σ:    3.910  |  Steps:     1697   Secs:      3      |  ⍺: 0.5000  Buffer:   1697
Episode:   300   Avg:    8.750   BestAvg:    9.600   σ:    3.940  |  Steps:     2572   Secs:      5      |  ⍺: 0.5000  Buffer:   2572
Episode:   400   Avg:    8.640   BestAvg:    9.600   σ:    4.713  |  Steps:     3436   Secs:      7      |  ⍺: 0.5000  Buffer:   3436
Episode:   500   Avg:    7.110   BestAvg:    9.600   σ:    3.391  |  Steps:     4147   Secs:      8      |  ⍺: 0.5000  Buffer:   4147
Episode:   600   Avg:    6.910   BestAvg:    9.600   σ:    2.446  |  Steps:     4838   Secs:     10      |  ⍺: 0.5000  Buffer:   4838
Episode:   700   Avg:    5.820   BestAvg:    9.600   σ:    2.211  |  Steps:     5420   Secs:     11      |  ⍺: 0.5000  Buffer:   5420
Episode:   800   Avg:    4.460   BestAvg:    9.600   σ:    1.126  |  Steps:     5866   Secs:     12      |  ⍺: 0.5000  Buffer:   5866
Episode:   900   Avg:    4.330   BestAvg:    9.600   σ:    0.679  |  Steps:     6299   Secs:     13      |  ⍺: 0.5000  Buffer:   6299
Episode:  1000   Avg:    4.490   BestAvg:    9.600   σ:    1.025  |  Steps:     6748   Secs:     14      |  ⍺: 0.5000  Buffer:   6748
Episode:  1100   Avg:    4.280   BestAvg:    9.600   σ:    0.584  |  Steps:     7176   Secs:     15      |  ⍺: 0.5000  Buffer:   7176
Episode:  1200   Avg:    4.340   BestAvg:    9.600   σ:    0.696  |  Steps:     7610   Secs:     16      |  ⍺: 0.5000  Buffer:   7610
Episode:  1300   Avg:    4.360   BestAvg:    9.600   σ:    0.686  |  Steps:     8046   Secs:     17      |  ⍺: 0.5000  Buffer:   8046
Episode:  1400   Avg:    4.590   BestAvg:    9.600   σ:    1.882  |  Steps:     8505   Secs:     18      |  ⍺: 0.5000  Buffer:   8505
Episode:  1500   Avg:    4.340   BestAvg:    9.600   σ:    0.724  |  Steps:     8939   Secs:     19      |  ⍺: 0.5000  Buffer:   8939
Episode:  1600   Avg:    4.440   BestAvg:    9.600   σ:    0.920  |  Steps:     9383   Secs:     20      |  ⍺: 0.5000  Buffer:   9383
Episode:  1700   Avg:    4.280   BestAvg:    9.600   σ:    0.549  |  Steps:     9811   Secs:     21      |  ⍺: 0.5000  Buffer:   9811
Episode:  1800   Avg:    4.170   BestAvg:    9.600   σ:    0.601  |  Steps:    10228   Secs:     22      |  ⍺: 0.5000  Buffer:  10228
Episode:  1900   Avg:    4.240   BestAvg:    9.600   σ:    0.472  |  Steps:    10652   Secs:     23      |  ⍺: 0.5000  Buffer:  10652
Episode:  2000   Avg:    4.180   BestAvg:    9.600   σ:    0.456  |  Steps:    11070   Secs:     24      |  ⍺: 0.5000  Buffer:  11070
Episode:  2100   Avg:    4.500   BestAvg:    9.600   σ:    0.781  |  Steps:    11520   Secs:     25      |  ⍺: 0.5000  Buffer:  11520
Episode:  2200   Avg:    4.600   BestAvg:    9.600   σ:    1.225  |  Steps:    11980   Secs:     26      |  ⍺: 0.5000  Buffer:  11980
Episode:  2300   Avg:    6.660   BestAvg:    9.600   σ:    4.092  |  Steps:    12646   Secs:     28      |  ⍺: 0.5000  Buffer:  12646
Episode:  2400   Avg:    5.130   BestAvg:    9.600   σ:    2.110  |  Steps:    13159   Secs:     30      |  ⍺: 0.5000  Buffer:  13159
Episode:  2500   Avg:    4.230   BestAvg:    9.600   σ:    0.563  |  Steps:    13582   Secs:     31      |  ⍺: 0.5000  Buffer:  13582
Episode:  2600   Avg:    4.430   BestAvg:    9.600   σ:    0.682  |  Steps:    14025   Secs:     32      |  ⍺: 0.5000  Buffer:  14025
Episode:  2700   Avg:    4.520   BestAvg:    9.600   σ:    1.109  |  Steps:    14477   Secs:     33      |  ⍺: 0.5000  Buffer:  14477
Episode:  2800   Avg:    4.270   BestAvg:    9.600   σ:    0.847  |  Steps:    14904   Secs:     34      |  ⍺: 0.5000  Buffer:  14904
Episode:  2900   Avg:    4.460   BestAvg:    9.600   σ:    1.153  |  Steps:    15350   Secs:     35      |  ⍺: 0.5000  Buffer:  15350
Episode:  3000   Avg:    4.230   BestAvg:    9.600   σ:    0.581  |  Steps:    15773   Secs:     36      |  ⍺: 0.5000  Buffer:  15773
Episode:  3100   Avg:    4.440   BestAvg:    9.600   σ:    0.779  |  Steps:    16217   Secs:     37      |  ⍺: 0.5000  Buffer:  16217
Episode:  3200   Avg:    4.550   BestAvg:    9.600   σ:    0.766  |  Steps:    16672   Secs:     38      |  ⍺: 0.5000  Buffer:  16672
Episode:  3300   Avg:    4.210   BestAvg:    9.600   σ:    0.475  |  Steps:    17093   Secs:     39      |  ⍺: 0.5000  Buffer:  17093
Episode:  3400   Avg:    4.090   BestAvg:    9.600   σ:    0.286  |  Steps:    17502   Secs:     40      |  ⍺: 0.5000  Buffer:  17502
Episode:  3500   Avg:    4.340   BestAvg:    9.600   σ:    0.851  |  Steps:    17936   Secs:     41      |  ⍺: 0.5000  Buffer:  17936
Episode:  3600   Avg:    4.250   BestAvg:    9.600   σ:    0.555  |  Steps:    18361   Secs:     42      |  ⍺: 0.5000  Buffer:  18361
Episode:  3700   Avg:    4.220   BestAvg:    9.600   σ:    0.593  |  Steps:    18783   Secs:     42      |  ⍺: 0.5000  Buffer:  18783
Episode:  3800   Avg:    4.400   BestAvg:    9.600   σ:    0.938  |  Steps:    19223   Secs:     43      |  ⍺: 0.5000  Buffer:  19223
Episode:  3900   Avg:    4.550   BestAvg:    9.600   σ:    0.910  |  Steps:    19678   Secs:     44      |  ⍺: 0.5000  Buffer:  19678
Episode:  4000   Avg:    4.630   BestAvg:    9.600   σ:    1.617  |  Steps:    20141   Secs:     45      |  ⍺: 0.5000  Buffer:  20141
Episode:  4100   Avg:    7.650   BestAvg:    9.600   σ:    4.286  |  Steps:    20906   Secs:     47      |  ⍺: 0.5000  Buffer:  20906
Episode:  4200   Avg:    7.450   BestAvg:    9.600   σ:    4.462  |  Steps:    21651   Secs:     48      |  ⍺: 0.5000  Buffer:  21651
Episode:  4300   Avg:    7.390   BestAvg:    9.600   σ:    2.580  |  Steps:    22390   Secs:     50      |  ⍺: 0.5000  Buffer:  22390
Episode:  4400   Avg:   15.340   BestAvg:   15.350   σ:    8.531  |  Steps:    23924   Secs:     53      |  ⍺: 0.5000  Buffer:  23924
Episode:  4500   Avg:   20.970   BestAvg:   20.970   σ:   10.168  |  Steps:    26021   Secs:     57      |  ⍺: 0.5000  Buffer:  26021
Episode:  4600   Avg:   18.000   BestAvg:   21.500   σ:    7.588  |  Steps:    27821   Secs:     60      |  ⍺: 0.5000  Buffer:  27821
Episode:  4700   Avg:   23.510   BestAvg:   23.510   σ:   14.150  |  Steps:    30172   Secs:     65      |  ⍺: 0.5000  Buffer:  30172
Episode:  4800   Avg:   50.470   BestAvg:   50.470   σ:   32.686  |  Steps:    35219   Secs:     76      |  ⍺: 0.5000  Buffer:  35219
Episode:  4900   Avg:   52.360   BestAvg:   61.060   σ:   42.149  |  Steps:    40455   Secs:     85      |  ⍺: 0.5000  Buffer:  40455
Episode:  5000   Avg:   49.960   BestAvg:   61.060   σ:   34.792  |  Steps:    45451   Secs:     94      |  ⍺: 0.5000  Buffer:  45451
Episode:  5100   Avg:   60.280   BestAvg:   61.060   σ:   30.814  |  Steps:    51479   Secs:    105      |  ⍺: 0.5000  Buffer:  51479
Episode:  5200   Avg:   67.320   BestAvg:   68.020   σ:   29.762  |  Steps:    58211   Secs:    117      |  ⍺: 0.5000  Buffer:  58211
Episode:  5300   Avg:  113.100   BestAvg:  113.100   σ:   66.097  |  Steps:    69521   Secs:    135      |  ⍺: 0.5000  Buffer:  69521
Episode:  5400   Avg:  108.600   BestAvg:  120.560   σ:   66.278  |  Steps:    80381   Secs:    158      |  ⍺: 0.5000  Buffer:  80381
Episode:  5500   Avg:   81.270   BestAvg:  120.560   σ:   41.162  |  Steps:    88508   Secs:    176      |  ⍺: 0.5000  Buffer:  88508
Episode:  5600   Avg:   73.250   BestAvg:  120.560   σ:   44.533  |  Steps:    95833   Secs:    187      |  ⍺: 0.5000  Buffer:  95833
Episode:  5700   Avg:   70.630   BestAvg:  120.560   σ:   43.615  |  Steps:   102896   Secs:    198      |  ⍺: 0.5000  Buffer: 100000
Episode:  5800   Avg:   61.100   BestAvg:  120.560   σ:   32.152  |  Steps:   109006   Secs:    208      |  ⍺: 0.5000  Buffer: 100000
Episode:  5900   Avg:   60.040   BestAvg:  120.560   σ:   31.523  |  Steps:   115010   Secs:    218      |  ⍺: 0.5000  Buffer: 100000
Episode:  6000   Avg:   51.260   BestAvg:  120.560   σ:   27.114  |  Steps:   120136   Secs:    226      |  ⍺: 0.5000  Buffer: 100000
Episode:  6100   Avg:   36.250   BestAvg:  120.560   σ:   17.491  |  Steps:   123761   Secs:    233      |  ⍺: 0.5000  Buffer: 100000
Episode:  6200   Avg:   36.000   BestAvg:  120.560   σ:   18.522  |  Steps:   127361   Secs:    240      |  ⍺: 0.5000  Buffer: 100000
Episode:  6300   Avg:   38.130   BestAvg:  120.560   σ:   15.223  |  Steps:   131174   Secs:    246      |  ⍺: 0.5000  Buffer: 100000
Episode:  6400   Avg:   38.100   BestAvg:  120.560   σ:   13.227  |  Steps:   134984   Secs:    253      |  ⍺: 0.5000  Buffer: 100000
Episode:  6500   Avg:   34.520   BestAvg:  120.560   σ:   12.131  |  Steps:   138436   Secs:    259      |  ⍺: 0.5000  Buffer: 100000
Episode:  6600   Avg:   31.090   BestAvg:  120.560   σ:   14.378  |  Steps:   141545   Secs:    265      |  ⍺: 0.5000  Buffer: 100000
Episode:  6700   Avg:   33.030   BestAvg:  120.560   σ:   12.857  |  Steps:   144848   Secs:    271      |  ⍺: 0.5000  Buffer: 100000
Episode:  6800   Avg:   36.530   BestAvg:  120.560   σ:   11.449  |  Steps:   148501   Secs:    277      |  ⍺: 0.5000  Buffer: 100000
Episode:  6900   Avg:   32.460   BestAvg:  120.560   σ:   14.499  |  Steps:   151747   Secs:    286      |  ⍺: 0.5000  Buffer: 100000
Episode:  7000   Avg:   43.650   BestAvg:  120.560   σ:   15.696  |  Steps:   156112   Secs:    296      |  ⍺: 0.5000  Buffer: 100000
Episode:  7100   Avg:  153.030   BestAvg:  153.030   σ:  245.232  |  Steps:   171415   Secs:    329      |  ⍺: 0.5000  Buffer: 100000
Episode:  7200   Avg:  286.350   BestAvg:  355.470   σ:  299.114  |  Steps:   200050   Secs:    374      |  ⍺: 0.5000  Buffer: 100000
Episode:  7300   Avg:  318.810   BestAvg:  355.470   σ:  320.015  |  Steps:   231931   Secs:    423      |  ⍺: 0.5000  Buffer: 100000
Episode:  7400   Avg:  195.570   BestAvg:  355.470   σ:  243.798  |  Steps:   251488   Secs:    453      |  ⍺: 0.5000  Buffer: 100000
Episode:  7500   Avg:  355.330   BestAvg:  359.790   σ:  380.217  |  Steps:   287021   Secs:    519      |  ⍺: 0.5000  Buffer: 100000
Episode:  7600   Avg:  300.780   BestAvg:  391.330   σ:  313.736  |  Steps:   317099   Secs:    565      |  ⍺: 0.5000  Buffer: 100000
Episode:  7700   Avg:   65.520   BestAvg:  391.330   σ:   25.457  |  Steps:   323651   Secs:    580      |  ⍺: 0.5000  Buffer: 100000
Episode:  7800   Avg:  361.260   BestAvg:  391.330   σ:  389.169  |  Steps:   359777   Secs:    643      |  ⍺: 0.5000  Buffer: 100000
Episode:  7900   Avg:  550.880   BestAvg:  569.770   σ:  385.143  |  Steps:   414865   Secs:    736      |  ⍺: 0.5000  Buffer: 100000
```

#### DDPG Evaluation
```
Loaded: checkpoints/last_run/episode.7100
Episode:   100   Avg: 1000.000   BestAvg:     -inf   σ:    0.000  |  Steps:   100000   Secs:     19      |  ⍺: 0.5000  Buffer: 100000
```
