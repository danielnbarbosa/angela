#### Notes
- HalfCheetah-v1 is considered "solved" when the agent obtains an average reward of at least 4800.0 over 100 consecutive episodes.  Assuming the same for HalfCheetah-v2
- with a high max_t (200) the cheetah flips over on it's back, yet still manages and impressive eval score of 1191
- some hyperparamater tuning tips mentioned [here](https://medium.com/@deshpandeshrinath/how-to-train-your-cheetah-with-deep-reinforcement-learning-14855518f916)
- max_t=15 appears to keep cheetah from flipping over

#### DDPG Training
```
Episode:   100   Avg:    1.585   BestAvg:     -inf   σ:    2.096  |  Steps:     1500   Secs:      8      |  ⍺: 0.5000  Buffer:   1500
Episode:   200   Avg:    1.832   BestAvg:    2.081   σ:    2.009  |  Steps:     3000   Secs:     20      |  ⍺: 0.5000  Buffer:   3000
Episode:   300   Avg:    3.060   BestAvg:    3.060   σ:    2.837  |  Steps:     4500   Secs:     34      |  ⍺: 0.5000  Buffer:   4500
Episode:   400   Avg:    7.122   BestAvg:    7.122   σ:    2.174  |  Steps:     6000   Secs:     51      |  ⍺: 0.5000  Buffer:   6000
Episode:   500   Avg:    8.000   BestAvg:    8.026   σ:    2.259  |  Steps:     7500   Secs:     65      |  ⍺: 0.5000  Buffer:   7500
Episode:   600   Avg:    9.068   BestAvg:    9.119   σ:    1.903  |  Steps:     9000   Secs:     77      |  ⍺: 0.5000  Buffer:   9000
Episode:   700   Avg:    8.776   BestAvg:    9.165   σ:    1.767  |  Steps:    10500   Secs:     89      |  ⍺: 0.5000  Buffer:  10500
Episode:   800   Avg:    9.325   BestAvg:    9.325   σ:    1.680  |  Steps:    12000   Secs:    104      |  ⍺: 0.5000  Buffer:  12000
Episode:   900   Avg:    9.030   BestAvg:    9.325   σ:    1.883  |  Steps:    13500   Secs:    116      |  ⍺: 0.5000  Buffer:  13500
Episode:  1000   Avg:    9.387   BestAvg:    9.507   σ:    2.264  |  Steps:    15000   Secs:    128      |  ⍺: 0.5000  Buffer:  15000
Episode:  1100   Avg:    8.724   BestAvg:    9.517   σ:    3.016  |  Steps:    16500   Secs:    137      |  ⍺: 0.5000  Buffer:  16500
Episode:  1200   Avg:    8.533   BestAvg:    9.604   σ:    3.001  |  Steps:    18000   Secs:    151      |  ⍺: 0.5000  Buffer:  18000
Episode:  1300   Avg:    8.115   BestAvg:    9.604   σ:    2.560  |  Steps:    19500   Secs:    162      |  ⍺: 0.5000  Buffer:  19500
Episode:  1400   Avg:    7.852   BestAvg:    9.604   σ:    2.680  |  Steps:    21000   Secs:    172      |  ⍺: 0.5000  Buffer:  21000
Episode:  1500   Avg:    7.192   BestAvg:    9.604   σ:    2.430  |  Steps:    22500   Secs:    183      |  ⍺: 0.5000  Buffer:  22500
Episode:  1600   Avg:    7.567   BestAvg:    9.604   σ:    2.609  |  Steps:    24000   Secs:    195      |  ⍺: 0.5000  Buffer:  24000
Episode:  1700   Avg:    8.250   BestAvg:    9.604   σ:    2.595  |  Steps:    25500   Secs:    207      |  ⍺: 0.5000  Buffer:  25500
Episode:  1800   Avg:    8.339   BestAvg:    9.604   σ:    2.559  |  Steps:    27000   Secs:    219      |  ⍺: 0.5000  Buffer:  27000
Episode:  1900   Avg:    8.555   BestAvg:    9.604   σ:    2.810  |  Steps:    28500   Secs:    231      |  ⍺: 0.5000  Buffer:  28500
Episode:  2000   Avg:    8.965   BestAvg:    9.604   σ:    2.678  |  Steps:    30000   Secs:    242      |  ⍺: 0.5000  Buffer:  30000
Episode:  2100   Avg:    9.687   BestAvg:    9.732   σ:    2.295  |  Steps:    31500   Secs:    252      |  ⍺: 0.5000  Buffer:  31500
Episode:  2200   Avg:    9.300   BestAvg:    9.732   σ:    2.224  |  Steps:    33000   Secs:    263      |  ⍺: 0.5000  Buffer:  33000
Episode:  2300   Avg:   10.252   BestAvg:   10.265   σ:    2.414  |  Steps:    34500   Secs:    274      |  ⍺: 0.5000  Buffer:  34500
Episode:  2400   Avg:   10.393   BestAvg:   10.476   σ:    2.485  |  Steps:    36000   Secs:    286      |  ⍺: 0.5000  Buffer:  36000
Episode:  2500   Avg:   10.558   BestAvg:   10.633   σ:    2.489  |  Steps:    37500   Secs:    297      |  ⍺: 0.5000  Buffer:  37500
Episode:  2600   Avg:   10.572   BestAvg:   10.743   σ:    2.447  |  Steps:    39000   Secs:    308      |  ⍺: 0.5000  Buffer:  39000
Episode:  2700   Avg:   10.870   BestAvg:   11.062   σ:    2.039  |  Steps:    40500   Secs:    318      |  ⍺: 0.5000  Buffer:  40500
Episode:  2800   Avg:   10.369   BestAvg:   11.062   σ:    2.122  |  Steps:    42000   Secs:    329      |  ⍺: 0.5000  Buffer:  42000
Episode:  2900   Avg:   10.407   BestAvg:   11.062   σ:    2.486  |  Steps:    43500   Secs:    341      |  ⍺: 0.5000  Buffer:  43500
Episode:  3000   Avg:   10.868   BestAvg:   11.062   σ:    2.031  |  Steps:    45000   Secs:    352      |  ⍺: 0.5000  Buffer:  45000
Episode:  3100   Avg:   10.623   BestAvg:   11.062   σ:    2.370  |  Steps:    46500   Secs:    363      |  ⍺: 0.5000  Buffer:  46500
Episode:  3200   Avg:   10.734   BestAvg:   11.062   σ:    2.069  |  Steps:    48000   Secs:    374      |  ⍺: 0.5000  Buffer:  48000
Episode:  3300   Avg:   11.021   BestAvg:   11.062   σ:    1.879  |  Steps:    49500   Secs:    385      |  ⍺: 0.5000  Buffer:  49500
Episode:  3400   Avg:   11.250   BestAvg:   11.410   σ:    2.527  |  Steps:    51000   Secs:    396      |  ⍺: 0.5000  Buffer:  51000
Episode:  3500   Avg:   10.671   BestAvg:   11.410   σ:    2.007  |  Steps:    52500   Secs:    408      |  ⍺: 0.5000  Buffer:  52500
Episode:  3600   Avg:   10.259   BestAvg:   11.410   σ:    2.107  |  Steps:    54000   Secs:    419      |  ⍺: 0.5000  Buffer:  54000
Episode:  3700   Avg:   10.580   BestAvg:   11.410   σ:    2.369  |  Steps:    55500   Secs:    433      |  ⍺: 0.5000  Buffer:  55500
Episode:  3800   Avg:   10.530   BestAvg:   11.410   σ:    1.902  |  Steps:    57000   Secs:    444      |  ⍺: 0.5000  Buffer:  57000
Episode:  3900   Avg:   10.701   BestAvg:   11.410   σ:    2.046  |  Steps:    58500   Secs:    455      |  ⍺: 0.5000  Buffer:  58500
Episode:  4000   Avg:   11.209   BestAvg:   11.410   σ:    2.328  |  Steps:    60000   Secs:    472      |  ⍺: 0.5000  Buffer:  60000
Episode:  4100   Avg:   11.161   BestAvg:   11.505   σ:    2.286  |  Steps:    61500   Secs:    484      |  ⍺: 0.5000  Buffer:  61500
Episode:  4200   Avg:   10.868   BestAvg:   11.505   σ:    2.235  |  Steps:    63000   Secs:    495      |  ⍺: 0.5000  Buffer:  63000
Episode:  4300   Avg:   11.020   BestAvg:   11.505   σ:    2.177  |  Steps:    64500   Secs:    506      |  ⍺: 0.5000  Buffer:  64500
Episode:  4400   Avg:   10.806   BestAvg:   11.505   σ:    2.473  |  Steps:    66000   Secs:    517      |  ⍺: 0.5000  Buffer:  66000
Episode:  4500   Avg:   11.033   BestAvg:   11.505   σ:    2.260  |  Steps:    67500   Secs:    527      |  ⍺: 0.5000  Buffer:  67500
Episode:  4600   Avg:   10.844   BestAvg:   11.505   σ:    2.453  |  Steps:    69000   Secs:    538      |  ⍺: 0.5000  Buffer:  69000
Episode:  4700   Avg:   10.861   BestAvg:   11.505   σ:    2.649  |  Steps:    70500   Secs:    551      |  ⍺: 0.5000  Buffer:  70500
Episode:  4800   Avg:   10.687   BestAvg:   11.505   σ:    2.395  |  Steps:    72000   Secs:    562      |  ⍺: 0.5000  Buffer:  72000
Episode:  4900   Avg:   11.195   BestAvg:   11.505   σ:    2.165  |  Steps:    73500   Secs:    572      |  ⍺: 0.5000  Buffer:  73500
Episode:  5000   Avg:   11.017   BestAvg:   11.505   σ:    2.213  |  Steps:    75000   Secs:    581      |  ⍺: 0.5000  Buffer:  75000
Episode:  5100   Avg:   11.286   BestAvg:   11.505   σ:    2.605  |  Steps:    76500   Secs:    591      |  ⍺: 0.5000  Buffer:  76500
Episode:  5200   Avg:   10.661   BestAvg:   11.505   σ:    2.718  |  Steps:    78000   Secs:    602      |  ⍺: 0.5000  Buffer:  78000
Episode:  5300   Avg:   11.126   BestAvg:   11.505   σ:    2.395  |  Steps:    79500   Secs:    614      |  ⍺: 0.5000  Buffer:  79500
Episode:  5400   Avg:   10.744   BestAvg:   11.505   σ:    2.610  |  Steps:    81000   Secs:    624      |  ⍺: 0.5000  Buffer:  81000
Episode:  5500   Avg:   11.156   BestAvg:   11.645   σ:    2.276  |  Steps:    82500   Secs:    635      |  ⍺: 0.5000  Buffer:  82500
Episode:  5600   Avg:   10.915   BestAvg:   11.645   σ:    2.465  |  Steps:    84000   Secs:    647      |  ⍺: 0.5000  Buffer:  84000
Episode:  5700   Avg:   10.810   BestAvg:   11.645   σ:    2.013  |  Steps:    85500   Secs:    657      |  ⍺: 0.5000  Buffer:  85500
Episode:  5800   Avg:   11.010   BestAvg:   11.645   σ:    2.294  |  Steps:    87000   Secs:    667      |  ⍺: 0.5000  Buffer:  87000
Episode:  5900   Avg:   11.137   BestAvg:   11.645   σ:    2.326  |  Steps:    88500   Secs:    679      |  ⍺: 0.5000  Buffer:  88500
Episode:  6000   Avg:   11.099   BestAvg:   11.645   σ:    2.534  |  Steps:    90000   Secs:    691      |  ⍺: 0.5000  Buffer:  90000
Episode:  6100   Avg:   11.036   BestAvg:   11.645   σ:    2.401  |  Steps:    91500   Secs:    702      |  ⍺: 0.5000  Buffer:  91500
Episode:  6200   Avg:   11.075   BestAvg:   11.645   σ:    2.337  |  Steps:    93000   Secs:    714      |  ⍺: 0.5000  Buffer:  93000
Episode:  6300   Avg:   10.857   BestAvg:   11.645   σ:    2.610  |  Steps:    94500   Secs:    726      |  ⍺: 0.5000  Buffer:  94500
Episode:  6400   Avg:   10.950   BestAvg:   11.645   σ:    2.124  |  Steps:    96000   Secs:    739      |  ⍺: 0.5000  Buffer:  96000
Episode:  6500   Avg:   10.931   BestAvg:   11.645   σ:    2.428  |  Steps:    97500   Secs:    751      |  ⍺: 0.5000  Buffer:  97500
Episode:  6600   Avg:   11.302   BestAvg:   11.645   σ:    2.440  |  Steps:    99000   Secs:    763      |  ⍺: 0.5000  Buffer:  99000
Episode:  6700   Avg:   11.258   BestAvg:   11.645   σ:    2.395  |  Steps:   100500   Secs:    774      |  ⍺: 0.5000  Buffer: 100000
Episode:  6800   Avg:   11.503   BestAvg:   11.645   σ:    2.155  |  Steps:   102000   Secs:    786      |  ⍺: 0.5000  Buffer: 100000
Episode:  6900   Avg:   11.025   BestAvg:   11.645   σ:    2.479  |  Steps:   103500   Secs:    798      |  ⍺: 0.5000  Buffer: 100000
Episode:  7000   Avg:   11.565   BestAvg:   11.645   σ:    2.271  |  Steps:   105000   Secs:    810      |  ⍺: 0.5000  Buffer: 100000
Episode:  7100   Avg:   11.008   BestAvg:   11.645   σ:    2.407  |  Steps:   106500   Secs:    822      |  ⍺: 0.5000  Buffer: 100000
Episode:  7200   Avg:   11.725   BestAvg:   11.734   σ:    2.049  |  Steps:   108000   Secs:    833      |  ⍺: 0.5000  Buffer: 100000
Episode:  7300   Avg:   11.199   BestAvg:   11.734   σ:    2.212  |  Steps:   109500   Secs:    845      |  ⍺: 0.5000  Buffer: 100000
Episode:  7400   Avg:   11.170   BestAvg:   11.734   σ:    2.286  |  Steps:   111000   Secs:    856      |  ⍺: 0.5000  Buffer: 100000
Episode:  7500   Avg:   11.148   BestAvg:   11.734   σ:    2.296  |  Steps:   112500   Secs:    868      |  ⍺: 0.5000  Buffer: 100000
Episode:  7600   Avg:   11.127   BestAvg:   11.734   σ:    2.241  |  Steps:   114000   Secs:    883      |  ⍺: 0.5000  Buffer: 100000
Episode:  7700   Avg:   11.484   BestAvg:   11.734   σ:    2.050  |  Steps:   115500   Secs:    894      |  ⍺: 0.5000  Buffer: 100000
Episode:  7800   Avg:   11.599   BestAvg:   11.734   σ:    2.127  |  Steps:   117000   Secs:    905      |  ⍺: 0.5000  Buffer: 100000
Episode:  7900   Avg:   11.656   BestAvg:   11.734   σ:    2.224  |  Steps:   118500   Secs:    917      |  ⍺: 0.5000  Buffer: 100000
Episode:  8000   Avg:   11.412   BestAvg:   11.734   σ:    2.374  |  Steps:   120000   Secs:    929      |  ⍺: 0.5000  Buffer: 100000
Episode:  8100   Avg:   11.215   BestAvg:   11.734   σ:    2.390  |  Steps:   121500   Secs:    942      |  ⍺: 0.5000  Buffer: 100000
Episode:  8200   Avg:   11.493   BestAvg:   11.734   σ:    1.901  |  Steps:   123000   Secs:    954      |  ⍺: 0.5000  Buffer: 100000
Episode:  8300   Avg:   11.606   BestAvg:   11.734   σ:    2.334  |  Steps:   124500   Secs:    966      |  ⍺: 0.5000  Buffer: 100000
Episode:  8400   Avg:   11.936   BestAvg:   12.024   σ:    1.965  |  Steps:   126000   Secs:    979      |  ⍺: 0.5000  Buffer: 100000
Episode:  8500   Avg:   11.706   BestAvg:   12.024   σ:    2.236  |  Steps:   127500   Secs:    992      |  ⍺: 0.5000  Buffer: 100000
```

#### DDPG Evaluation
```
Loaded: checkpoints/last_run/episode.6000
Episode:    20   Avg: 2987.540   BestAvg:     -inf   σ:  130.292  |  Steps:     1000   Reward: 3155.859  |  ⍺: 0.5000  Buffer:  20000
```
