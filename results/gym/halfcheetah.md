#### Notes
- HalfCheetah-v1 is considered "solved" when the agent obtains an average reward of at least 4800.0 over 100 consecutive episodes.  Assuming the same for HalfCheetah-v2
- with a high max_t (200) the cheetah flips over on it's back, yet still manages and impressive eval score of 1191
- some hyperparamater tuning tips mentioned [here](https://medium.com/@deshpandeshrinath/how-to-train-your-cheetah-with-deep-reinforcement-learning-14855518f916)
- max_t between 15 and 30 appears to keep cheetah from flipping over
- still not sure how to have the camera in mujoco track the agent

#### DDPG Training
```
Episode:   100   Avg:    0.340   BestAvg:     -inf   σ:    2.059  |  Steps:     1500   Secs:      8      |  ⍺: 0.5000  Buffer:   1500
Episode:   200   Avg:    2.248   BestAvg:    2.248   σ:    2.238  |  Steps:     3000   Secs:     17      |  ⍺: 0.5000  Buffer:   3000
Episode:   300   Avg:    4.199   BestAvg:    4.199   σ:    1.967  |  Steps:     4500   Secs:     26      |  ⍺: 0.5000  Buffer:   4500
Episode:   400   Avg:    8.074   BestAvg:    8.074   σ:    2.246  |  Steps:     6000   Secs:     37      |  ⍺: 0.5000  Buffer:   6000
Episode:   500   Avg:    8.152   BestAvg:    8.358   σ:    2.325  |  Steps:     7500   Secs:     50      |  ⍺: 0.5000  Buffer:   7500
Episode:   600   Avg:    9.208   BestAvg:    9.247   σ:    2.037  |  Steps:     9000   Secs:     61      |  ⍺: 0.5000  Buffer:   9000
Episode:   700   Avg:    9.297   BestAvg:    9.379   σ:    1.873  |  Steps:    10500   Secs:     74      |  ⍺: 0.5000  Buffer:  10500
Episode:   800   Avg:    9.133   BestAvg:    9.493   σ:    1.585  |  Steps:    12000   Secs:     84      |  ⍺: 0.5000  Buffer:  12000
Episode:   900   Avg:    9.276   BestAvg:    9.493   σ:    2.053  |  Steps:    13500   Secs:     92      |  ⍺: 0.5000  Buffer:  13500
Episode:  1000   Avg:    9.050   BestAvg:    9.495   σ:    1.608  |  Steps:    15000   Secs:    102      |  ⍺: 0.5000  Buffer:  15000
Episode:  1100   Avg:    8.865   BestAvg:    9.495   σ:    1.534  |  Steps:    16500   Secs:    119      |  ⍺: 0.5000  Buffer:  16500
Episode:  1200   Avg:    9.599   BestAvg:    9.700   σ:    2.225  |  Steps:    18000   Secs:    130      |  ⍺: 0.5000  Buffer:  18000
Episode:  1300   Avg:    8.668   BestAvg:    9.700   σ:    2.814  |  Steps:    19500   Secs:    139      |  ⍺: 0.5000  Buffer:  19500
Episode:  1400   Avg:    7.835   BestAvg:    9.700   σ:    2.447  |  Steps:    21000   Secs:    153      |  ⍺: 0.5000  Buffer:  21000
Episode:  1500   Avg:    7.455   BestAvg:    9.700   σ:    2.313  |  Steps:    22500   Secs:    164      |  ⍺: 0.5000  Buffer:  22500
Episode:  1600   Avg:    7.077   BestAvg:    9.700   σ:    2.108  |  Steps:    24000   Secs:    173      |  ⍺: 0.5000  Buffer:  24000
Episode:  1700   Avg:    6.995   BestAvg:    9.700   σ:    3.076  |  Steps:    25500   Secs:    182      |  ⍺: 0.5000  Buffer:  25500
Episode:  1800   Avg:    7.266   BestAvg:    9.700   σ:    3.593  |  Steps:    27000   Secs:    191      |  ⍺: 0.5000  Buffer:  27000
Episode:  1900   Avg:    8.411   BestAvg:    9.700   σ:    3.252  |  Steps:    28500   Secs:    199      |  ⍺: 0.5000  Buffer:  28500
Episode:  2000   Avg:    8.292   BestAvg:    9.700   σ:    3.339  |  Steps:    30000   Secs:    213      |  ⍺: 0.5000  Buffer:  30000
Episode:  2100   Avg:    8.579   BestAvg:    9.700   σ:    3.736  |  Steps:    31500   Secs:    222      |  ⍺: 0.5000  Buffer:  31500
Episode:  2200   Avg:    9.373   BestAvg:    9.700   σ:    3.549  |  Steps:    33000   Secs:    232      |  ⍺: 0.5000  Buffer:  33000
Episode:  2300   Avg:   10.957   BestAvg:   10.957   σ:    2.709  |  Steps:    34500   Secs:    241      |  ⍺: 0.5000  Buffer:  34500
Episode:  2400   Avg:   11.542   BestAvg:   11.598   σ:    2.658  |  Steps:    36000   Secs:    253      |  ⍺: 0.5000  Buffer:  36000
Episode:  2500   Avg:   12.235   BestAvg:   12.329   σ:    2.225  |  Steps:    37500   Secs:    265      |  ⍺: 0.5000  Buffer:  37500
Episode:  2600   Avg:   12.389   BestAvg:   12.415   σ:    1.968  |  Steps:    39000   Secs:    274      |  ⍺: 0.5000  Buffer:  39000
Episode:  2700   Avg:   12.265   BestAvg:   12.630   σ:    1.773  |  Steps:    40500   Secs:    283      |  ⍺: 0.5000  Buffer:  40500
Episode:  2771   Avg:   11.994   BestAvg:   12.630   σ:    1.747  |  Steps:       15   Reward:   13.296  |  ⍺: 0.5000  Buffer:  41565^CTraceback (most recent call last):
  File "./main.py", line 41, in <module>
    training.train(environment, agent, render=args.render, **cfg.train)
  File "/Users/danielnbarbosa/src/ml/danielnbarbosa/angela/libs/algorithms/ddpg/training.py", line 67, in train
    agent.step(state, action, reward, next_state, done)
  File "/Users/danielnbarbosa/src/ml/danielnbarbosa/angela/libs/algorithms/ddpg/agents.py", line 153, in step
    experiences = self.memory.sample()
  File "/Users/danielnbarbosa/src/ml/danielnbarbosa/angela/libs/agent_util.py", line 78, in sample
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
  File "/anaconda3/envs/mujoco/lib/python3.6/site-packages/numpy/core/shape_base.py", line 234, in vstack
    return _nx.concatenate([atleast_2d(_m) for _m in tup], 0)
  File "/anaconda3/envs/mujoco/lib/python3.6/site-packages/numpy/core/shape_base.py", line 234, in <listcomp>
    return _nx.concatenate([atleast_2d(_m) for _m in tup], 0)
  File "/anaconda3/envs/mujoco/lib/python3.6/site-packages/numpy/core/shape_base.py", line 109, in atleast_2d
    res.append(result)
KeyboardInterrupt
(mujoco) danielnbarbosa@cafune~/src/ml/danielnbarbosa/angela$ ./main.py --cfg=cfg/gym/halfcheetah/halfcheetah_ddpg.py
SEED: 0
LowDimActor(
  (fc1): Linear(in_features=17, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=6, bias=True)
)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                  [-1, 128]           2,304
            Linear-2                   [-1, 64]           8,256
            Linear-3                    [-1, 6]             390
================================================================
Total params: 10,950
Trainable params: 10,950
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.04
Estimated Total Size (MB): 0.04
----------------------------------------------------------------
LowDimCritic(
  (fc1): Linear(in_features=17, out_features=128, bias=True)
  (fc2): Linear(in_features=134, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=1, bias=True)
)
Episode:   100   Avg:   -4.445   BestAvg:     -inf   σ:    2.564  |  Steps:     3000   Secs:     20      |  ⍺: 0.5000  Buffer:   3000
Episode:   200   Avg:    3.362   BestAvg:    3.362   σ:    7.038  |  Steps:     6000   Secs:     38      |  ⍺: 0.5000  Buffer:   6000
Episode:   300   Avg:    7.567   BestAvg:    7.574   σ:    5.221  |  Steps:     9000   Secs:     57      |  ⍺: 0.5000  Buffer:   9000
Episode:   400   Avg:   12.708   BestAvg:   12.708   σ:    4.980  |  Steps:    12000   Secs:     74      |  ⍺: 0.5000  Buffer:  12000
Episode:   500   Avg:   16.741   BestAvg:   16.998   σ:    5.598  |  Steps:    15000   Secs:     92      |  ⍺: 0.5000  Buffer:  15000
Episode:   600   Avg:   20.298   BestAvg:   20.298   σ:    4.733  |  Steps:    18000   Secs:    112      |  ⍺: 0.5000  Buffer:  18000
Episode:   700   Avg:   22.189   BestAvg:   22.338   σ:    4.597  |  Steps:    21000   Secs:    155      |  ⍺: 0.5000  Buffer:  21000
Episode:   800   Avg:   26.202   BestAvg:   26.202   σ:    5.463  |  Steps:    24000   Secs:    179      |  ⍺: 0.5000  Buffer:  24000
Episode:   900   Avg:   26.969   BestAvg:   27.417   σ:    6.242  |  Steps:    27000   Secs:    198      |  ⍺: 0.5000  Buffer:  27000
Episode:  1000   Avg:   20.514   BestAvg:   27.417   σ:   10.739  |  Steps:    30000   Secs:    219      |  ⍺: 0.5000  Buffer:  30000
Episode:  1100   Avg:   26.092   BestAvg:   27.417   σ:    8.032  |  Steps:    33000   Secs:    237      |  ⍺: 0.5000  Buffer:  33000
Episode:  1200   Avg:   27.878   BestAvg:   27.943   σ:    6.550  |  Steps:    36000   Secs:    256      |  ⍺: 0.5000  Buffer:  36000
Episode:  1300   Avg:   29.464   BestAvg:   29.479   σ:    8.749  |  Steps:    39000   Secs:    274      |  ⍺: 0.5000  Buffer:  39000
Episode:  1400   Avg:   32.111   BestAvg:   32.357   σ:    7.710  |  Steps:    42000   Secs:    292      |  ⍺: 0.5000  Buffer:  42000
Episode:  1500   Avg:   32.629   BestAvg:   32.716   σ:    6.474  |  Steps:    45000   Secs:    309      |  ⍺: 0.5000  Buffer:  45000
Episode:  1600   Avg:   34.963   BestAvg:   35.332   σ:    6.707  |  Steps:    48000   Secs:    329      |  ⍺: 0.5000  Buffer:  48000
Episode:  1700   Avg:   35.644   BestAvg:   35.803   σ:    5.652  |  Steps:    51000   Secs:    347      |  ⍺: 0.5000  Buffer:  51000
Episode:  1800   Avg:   38.637   BestAvg:   38.699   σ:    5.883  |  Steps:    54000   Secs:    373      |  ⍺: 0.5000  Buffer:  54000
Episode:  1900   Avg:   38.826   BestAvg:   39.324   σ:    5.419  |  Steps:    57000   Secs:    395      |  ⍺: 0.5000  Buffer:  57000
Episode:  2000   Avg:   38.675   BestAvg:   39.701   σ:    6.397  |  Steps:    60000   Secs:    415      |  ⍺: 0.5000  Buffer:  60000
Episode:  2100   Avg:   40.276   BestAvg:   40.317   σ:    5.733  |  Steps:    63000   Secs:    436      |  ⍺: 0.5000  Buffer:  63000
Episode:  2200   Avg:   41.767   BestAvg:   41.890   σ:    6.614  |  Steps:    66000   Secs:    456      |  ⍺: 0.5000  Buffer:  66000
Episode:  2300   Avg:   40.660   BestAvg:   41.890   σ:    7.355  |  Steps:    69000   Secs:    480      |  ⍺: 0.5000  Buffer:  69000
Episode:  2400   Avg:   40.147   BestAvg:   41.890   σ:    7.329  |  Steps:    72000   Secs:    512      |  ⍺: 0.5000  Buffer:  72000
Episode:  2500   Avg:   40.228   BestAvg:   41.890   σ:    6.453  |  Steps:    75000   Secs:    541      |  ⍺: 0.5000  Buffer:  75000
Episode:  2600   Avg:   39.700   BestAvg:   41.890   σ:    6.532  |  Steps:    78000   Secs:    570      |  ⍺: 0.5000  Buffer:  78000
Episode:  2700   Avg:   41.441   BestAvg:   41.890   σ:    7.298  |  Steps:    81000   Secs:    599      |  ⍺: 0.5000  Buffer:  81000
Episode:  2800   Avg:   41.100   BestAvg:   41.896   σ:    7.061  |  Steps:    84000   Secs:    637      |  ⍺: 0.5000  Buffer:  84000
Episode:  2900   Avg:   41.848   BestAvg:   42.149   σ:    6.770  |  Steps:    87000   Secs:    659      |  ⍺: 0.5000  Buffer:  87000
Episode:  3000   Avg:   43.315   BestAvg:   43.599   σ:    6.426  |  Steps:    90000   Secs:    678      |  ⍺: 0.5000  Buffer:  90000
Episode:  3100   Avg:   43.333   BestAvg:   43.865   σ:    6.214  |  Steps:    93000   Secs:    699      |  ⍺: 0.5000  Buffer:  93000
Episode:  3200   Avg:   43.533   BestAvg:   43.865   σ:    6.394  |  Steps:    96000   Secs:    719      |  ⍺: 0.5000  Buffer:  96000
Episode:  3300   Avg:   43.698   BestAvg:   43.961   σ:    5.889  |  Steps:    99000   Secs:    739      |  ⍺: 0.5000  Buffer:  99000
Episode:  3400   Avg:   43.598   BestAvg:   44.206   σ:    7.180  |  Steps:   102000   Secs:    762      |  ⍺: 0.5000  Buffer: 100000
Episode:  3500   Avg:   44.814   BestAvg:   45.268   σ:    5.856  |  Steps:   105000   Secs:    785      |  ⍺: 0.5000  Buffer: 100000
Episode:  3600   Avg:   45.328   BestAvg:   45.360   σ:    4.918  |  Steps:   108000   Secs:    806      |  ⍺: 0.5000  Buffer: 100000
Episode:  3700   Avg:   45.850   BestAvg:   45.975   σ:    4.891  |  Steps:   111000   Secs:    827      |  ⍺: 0.5000  Buffer: 100000
Episode:  3800   Avg:   46.128   BestAvg:   46.845   σ:    5.577  |  Steps:   114000   Secs:    848      |  ⍺: 0.5000  Buffer: 100000
Episode:  3900   Avg:   46.228   BestAvg:   46.845   σ:    5.983  |  Steps:   117000   Secs:    870      |  ⍺: 0.5000  Buffer: 100000
Episode:  4000   Avg:   46.747   BestAvg:   47.059   σ:    5.547  |  Steps:   120000   Secs:    892      |  ⍺: 0.5000  Buffer: 100000
Episode:  4100   Avg:   46.442   BestAvg:   47.378   σ:    6.003  |  Steps:   123000   Secs:    915      |  ⍺: 0.5000  Buffer: 100000
Episode:  4200   Avg:   47.858   BestAvg:   47.867   σ:    5.861  |  Steps:   126000   Secs:    936      |  ⍺: 0.5000  Buffer: 100000
Episode:  4300   Avg:   47.010   BestAvg:   47.867   σ:    7.082  |  Steps:   129000   Secs:    959      |  ⍺: 0.5000  Buffer: 100000
Episode:  4400   Avg:   48.684   BestAvg:   48.802   σ:    6.623  |  Steps:   132000   Secs:    981      |  ⍺: 0.5000  Buffer: 100000
Episode:  4500   Avg:   49.587   BestAvg:   50.627   σ:    6.437  |  Steps:   135000   Secs:   1005      |  ⍺: 0.5000  Buffer: 100000
Episode:  4600   Avg:   48.202   BestAvg:   50.627   σ:    6.713  |  Steps:   138000   Secs:   1026      |  ⍺: 0.5000  Buffer: 100000
Episode:  4700   Avg:   47.723   BestAvg:   50.627   σ:    6.370  |  Steps:   141000   Secs:   1048      |  ⍺: 0.5000  Buffer: 100000
Episode:  4800   Avg:   49.631   BestAvg:   50.627   σ:    6.369  |  Steps:   144000   Secs:   1067      |  ⍺: 0.5000  Buffer: 100000
Episode:  4900   Avg:   50.089   BestAvg:   50.627   σ:    6.371  |  Steps:   147000   Secs:   1086      |  ⍺: 0.5000  Buffer: 100000
Episode:  5000   Avg:   49.706   BestAvg:   50.783   σ:    6.036  |  Steps:   150000   Secs:   1105      |  ⍺: 0.5000  Buffer: 100000
Episode:  5100   Avg:   50.431   BestAvg:   50.783   σ:    6.661  |  Steps:   153000   Secs:   1123      |  ⍺: 0.5000  Buffer: 100000
Episode:  5200   Avg:   49.403   BestAvg:   50.783   σ:    6.176  |  Steps:   156000   Secs:   1142      |  ⍺: 0.5000  Buffer: 100000
Episode:  5300   Avg:   50.256   BestAvg:   50.783   σ:    6.823  |  Steps:   159000   Secs:   1164      |  ⍺: 0.5000  Buffer: 100000
Episode:  5400   Avg:   50.119   BestAvg:   50.972   σ:    6.914  |  Steps:   162000   Secs:   1186      |  ⍺: 0.5000  Buffer: 100000
Episode:  5500   Avg:   50.915   BestAvg:   50.995   σ:    6.726  |  Steps:   165000   Secs:   1210      |  ⍺: 0.5000  Buffer: 100000
Episode:  5600   Avg:   50.725   BestAvg:   51.356   σ:    7.699  |  Steps:   168000   Secs:   1232      |  ⍺: 0.5000  Buffer: 100000
Episode:  5700   Avg:   51.936   BestAvg:   52.214   σ:    6.298  |  Steps:   171000   Secs:   1254      |  ⍺: 0.5000  Buffer: 100000
Episode:  5800   Avg:   50.681   BestAvg:   52.263   σ:    6.990  |  Steps:   174000   Secs:   1277      |  ⍺: 0.5000  Buffer: 100000
Episode:  5900   Avg:   51.257   BestAvg:   52.263   σ:    7.315  |  Steps:   177000   Secs:   1301      |  ⍺: 0.5000  Buffer: 100000
Episode:  6000   Avg:   52.441   BestAvg:   52.557   σ:    7.103  |  Steps:   180000   Secs:   1321      |  ⍺: 0.5000  Buffer: 100000
Episode:  6100   Avg:   51.781   BestAvg:   52.799   σ:    6.941  |  Steps:   183000   Secs:   1368      |  ⍺: 0.5000  Buffer: 100000
Episode:  6200   Avg:   50.612   BestAvg:   52.799   σ:    8.631  |  Steps:   186000   Secs:   1396      |  ⍺: 0.5000  Buffer: 100000
Episode:  6300   Avg:   51.026   BestAvg:   52.799   σ:    8.726  |  Steps:   189000   Secs:   1424      |  ⍺: 0.5000  Buffer: 100000
Episode:  6400   Avg:   51.069   BestAvg:   52.799   σ:    9.746  |  Steps:   192000   Secs:   1450      |  ⍺: 0.5000  Buffer: 100000
Episode:  6500   Avg:   51.918   BestAvg:   52.799   σ:    8.654  |  Steps:   195000   Secs:   1475      |  ⍺: 0.5000  Buffer: 100000
Episode:  6600   Avg:   54.219   BestAvg:   54.860   σ:    6.507  |  Steps:   198000   Secs:   1500      |  ⍺: 0.5000  Buffer: 100000
Episode:  6700   Avg:   52.071   BestAvg:   54.860   σ:    7.974  |  Steps:   201000   Secs:   1530      |  ⍺: 0.5000  Buffer: 100000
Episode:  6800   Avg:   54.048   BestAvg:   54.860   σ:    8.203  |  Steps:   204000   Secs:   1559      |  ⍺: 0.5000  Buffer: 100000
Episode:  6900   Avg:   52.314   BestAvg:   54.860   σ:    7.907  |  Steps:   207000   Secs:   1587      |  ⍺: 0.5000  Buffer: 100000
Episode:  7000   Avg:   53.715   BestAvg:   54.860   σ:    6.919  |  Steps:   210000   Secs:   1615      |  ⍺: 0.5000  Buffer: 100000
Episode:  7100   Avg:   53.868   BestAvg:   54.860   σ:    7.100  |  Steps:   213000   Secs:   1638      |  ⍺: 0.5000  Buffer: 100000
Episode:  7200   Avg:   54.953   BestAvg:   55.248   σ:    7.006  |  Steps:   216000   Secs:   1662      |  ⍺: 0.5000  Buffer: 100000
Episode:  7300   Avg:   54.351   BestAvg:   55.248   σ:    7.257  |  Steps:   219000   Secs:   1685      |  ⍺: 0.5000  Buffer: 100000
Episode:  7400   Avg:   54.639   BestAvg:   55.248   σ:    7.070  |  Steps:   222000   Secs:   1710      |  ⍺: 0.5000  Buffer: 100000
Episode:  7500   Avg:   56.597   BestAvg:   56.931   σ:    5.911  |  Steps:   225000   Secs:   1731      |  ⍺: 0.5000  Buffer: 100000
Episode:  7600   Avg:   55.747   BestAvg:   56.931   σ:    7.360  |  Steps:   228000   Secs:   1756      |  ⍺: 0.5000  Buffer: 100000
Episode:  7700   Avg:   55.906   BestAvg:   56.931   σ:    7.217  |  Steps:   231000   Secs:   1779      |  ⍺: 0.5000  Buffer: 100000
Episode:  7800   Avg:   55.966   BestAvg:   56.931   σ:    7.076  |  Steps:   234000   Secs:   1804      |  ⍺: 0.5000  Buffer: 100000
Episode:  7900   Avg:   56.627   BestAvg:   56.931   σ:    6.309  |  Steps:   237000   Secs:   1843      |  ⍺: 0.5000  Buffer: 100000
Episode:  8000   Avg:   55.709   BestAvg:   56.931   σ:    6.479  |  Steps:   240000   Secs:   1866      |  ⍺: 0.5000  Buffer: 100000
Episode:  8100   Avg:   56.317   BestAvg:   56.966   σ:    7.551  |  Steps:   243000   Secs:   1888      |  ⍺: 0.5000  Buffer: 100000
Episode:  8200   Avg:   56.352   BestAvg:   56.966   σ:    7.600  |  Steps:   246000   Secs:   1908      |  ⍺: 0.5000  Buffer: 100000
Episode:  8300   Avg:   55.536   BestAvg:   56.973   σ:    7.519  |  Steps:   249000   Secs:   1928      |  ⍺: 0.5000  Buffer: 100000
Episode:  8400   Avg:   56.210   BestAvg:   56.973   σ:    7.479  |  Steps:   252000   Secs:   1949      |  ⍺: 0.5000  Buffer: 100000
Episode:  8500   Avg:   55.216   BestAvg:   56.973   σ:    8.368  |  Steps:   255000   Secs:   1969      |  ⍺: 0.5000  Buffer: 100000
Episode:  8600   Avg:   55.438   BestAvg:   56.973   σ:    6.705  |  Steps:   258000   Secs:   1990      |  ⍺: 0.5000  Buffer: 100000
Episode:  8700   Avg:   56.111   BestAvg:   56.973   σ:    7.529  |  Steps:   261000   Secs:   2010      |  ⍺: 0.5000  Buffer: 100000
Episode:  8800   Avg:   57.114   BestAvg:   57.206   σ:    8.199  |  Steps:   264000   Secs:   2031      |  ⍺: 0.5000  Buffer: 100000
Episode:  8900   Avg:   58.346   BestAvg:   58.946   σ:    7.779  |  Steps:   267000   Secs:   2051      |  ⍺: 0.5000  Buffer: 100000
Episode:  9000   Avg:   56.856   BestAvg:   58.946   σ:    6.854  |  Steps:   270000   Secs:   2071      |  ⍺: 0.5000  Buffer: 100000
Episode:  9100   Avg:   56.079   BestAvg:   58.946   σ:    7.426  |  Steps:   273000   Secs:   2091      |  ⍺: 0.5000  Buffer: 100000
Episode:  9200   Avg:   56.420   BestAvg:   58.946   σ:    7.772  |  Steps:   276000   Secs:   2111      |  ⍺: 0.5000  Buffer: 100000
Episode:  9300   Avg:   55.242   BestAvg:   58.946   σ:    7.546  |  Steps:   279000   Secs:   2131      |  ⍺: 0.5000  Buffer: 100000
Episode:  9400   Avg:   53.353   BestAvg:   58.946   σ:    8.288  |  Steps:   282000   Secs:   2152      |  ⍺: 0.5000  Buffer: 100000
Episode:  9500   Avg:   56.642   BestAvg:   58.946   σ:    7.808  |  Steps:   285000   Secs:   2172      |  ⍺: 0.5000  Buffer: 100000
Episode:  9600   Avg:   57.681   BestAvg:   58.946   σ:    7.702  |  Steps:   288000   Secs:   2192      |  ⍺: 0.5000  Buffer: 100000
Episode:  9700   Avg:   56.512   BestAvg:   58.946   σ:    7.215  |  Steps:   291000   Secs:   2213      |  ⍺: 0.5000  Buffer: 100000
Episode:  9800   Avg:   59.380   BestAvg:   59.758   σ:    8.206  |  Steps:   294000   Secs:   2233      |  ⍺: 0.5000  Buffer: 100000
Episode:  9900   Avg:   58.957   BestAvg:   60.751   σ:    8.170  |  Steps:   297000   Secs:   2253      |  ⍺: 0.5000  Buffer: 100000
Episode: 10000   Avg:   58.720   BestAvg:   60.751   σ:    8.528  |  Steps:   300000   Secs:   2273      |  ⍺: 0.5000  Buffer: 100000
...
Episode: 17000   Avg:   57.565   BestAvg:   60.751   σ:    9.272  |  Steps:   510000   Secs:   3731      |  ⍺: 0.5000  Buffer: 100000
```

#### DDPG Evaluation
```
Loaded: checkpoints/last_run/episode.7500
Episode:   100   Avg: 4630.887   BestAvg:     -inf   σ:  478.430  |  Steps:   100000   Secs:     53      |  ⍺: 0.5000  Buffer: 100000
```
