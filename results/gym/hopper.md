#### Notes
- Hopper-v1 is considered "solved" when the agent obtains an average reward of at least 3800.0 over 100 consecutive episodes.  Assuming the same for Hopper-v2
- hopper sets done flag if something wrong happens, so step count is automatically controlled


#### DDPG Training
```
Episode:   100   Avg:   32.880   BestAvg:     -inf   σ:   18.432  |  Steps:     2188   Secs:      7      |  ⍺: 0.5000  Buffer:   2188
Episode:   200   Avg:   40.314   BestAvg:   41.738   σ:    2.185  |  Steps:     4508   Secs:     15      |  ⍺: 0.5000  Buffer:   4508
Episode:   300   Avg:   40.004   BestAvg:   41.738   σ:    1.684  |  Steps:     6803   Secs:     23      |  ⍺: 0.5000  Buffer:   6803
Episode:   400   Avg:   40.861   BestAvg:   41.738   σ:    2.718  |  Steps:     9161   Secs:     31      |  ⍺: 0.5000  Buffer:   9161
Episode:   500   Avg:   89.228   BestAvg:   89.228   σ:   44.934  |  Steps:    14643   Secs:     48      |  ⍺: 0.5000  Buffer:  14643
Episode:   600   Avg:  129.299   BestAvg:  134.710   σ:   30.599  |  Steps:    22329   Secs:     72      |  ⍺: 0.5000  Buffer:  22329
Episode:   700   Avg:  171.442   BestAvg:  171.442   σ:   10.785  |  Steps:    30941   Secs:    100      |  ⍺: 0.5000  Buffer:  30941
Episode:   800   Avg:  174.166   BestAvg:  176.403   σ:   11.594  |  Steps:    39397   Secs:    127      |  ⍺: 0.5000  Buffer:  39397
Episode:   900   Avg:  176.078   BestAvg:  176.403   σ:   15.299  |  Steps:    47913   Secs:    158      |  ⍺: 0.5000  Buffer:  47913
Episode:  1000   Avg:  188.688   BestAvg:  188.688   σ:   10.747  |  Steps:    56966   Secs:    187      |  ⍺: 0.5000  Buffer:  56966
Episode:  1100   Avg:  205.213   BestAvg:  205.213   σ:    8.365  |  Steps:    66580   Secs:    223      |  ⍺: 0.5000  Buffer:  66580
Episode:  1200   Avg:  203.614   BestAvg:  215.932   σ:   75.876  |  Steps:    76680   Secs:    257      |  ⍺: 0.5000  Buffer:  76680
Episode:  1300   Avg:  246.196   BestAvg:  254.170   σ:   41.454  |  Steps:    88357   Secs:    297      |  ⍺: 0.5000  Buffer:  88357
Episode:  1400   Avg:  197.976   BestAvg:  254.170   σ:   37.357  |  Steps:    98148   Secs:    330      |  ⍺: 0.5000  Buffer:  98148
Episode:  1500   Avg:  241.064   BestAvg:  254.170   σ:   42.973  |  Steps:   109534   Secs:    372      |  ⍺: 0.5000  Buffer: 100000
Episode:  1600   Avg:  230.230   BestAvg:  254.170   σ:   86.791  |  Steps:   120116   Secs:    406      |  ⍺: 0.5000  Buffer: 100000
Episode:  1700   Avg:  255.774   BestAvg:  255.909   σ:   81.008  |  Steps:   131685   Secs:    446      |  ⍺: 0.5000  Buffer: 100000
Episode:  1800   Avg:  292.722   BestAvg:  293.184   σ:   36.538  |  Steps:   145437   Secs:    494      |  ⍺: 0.5000  Buffer: 100000
Episode:  1900   Avg:  284.275   BestAvg:  294.882   σ:   32.719  |  Steps:   158097   Secs:    537      |  ⍺: 0.5000  Buffer: 100000
Episode:  2000   Avg:  270.622   BestAvg:  294.882   σ:   57.310  |  Steps:   169762   Secs:    577      |  ⍺: 0.5000  Buffer: 100000
Episode:  2100   Avg:  312.123   BestAvg:  312.123   σ:   23.227  |  Steps:   182344   Secs:    618      |  ⍺: 0.5000  Buffer: 100000
Episode:  2200   Avg:  330.087   BestAvg:  332.586   σ:   31.995  |  Steps:   195390   Secs:    659      |  ⍺: 0.5000  Buffer: 100000
Episode:  2300   Avg:  292.415   BestAvg:  332.586   σ:   68.609  |  Steps:   207722   Secs:    699      |  ⍺: 0.5000  Buffer: 100000
Episode:  2400   Avg:  297.221   BestAvg:  332.586   σ:   32.726  |  Steps:   220101   Secs:    739      |  ⍺: 0.5000  Buffer: 100000
Episode:  2500   Avg:  303.289   BestAvg:  332.586   σ:   66.899  |  Steps:   232972   Secs:    780      |  ⍺: 0.5000  Buffer: 100000
Episode:  2600   Avg:  315.182   BestAvg:  336.040   σ:  103.681  |  Steps:   245241   Secs:    820      |  ⍺: 0.5000  Buffer: 100000
Episode:  2700   Avg:  317.228   BestAvg:  336.040   σ:   98.459  |  Steps:   256884   Secs:    861      |  ⍺: 0.5000  Buffer: 100000
Episode:  2800   Avg:  350.727   BestAvg:  399.221   σ:  180.004  |  Steps:   270313   Secs:    908      |  ⍺: 0.5000  Buffer: 100000
Episode:  2900   Avg:  347.151   BestAvg:  399.221   σ:  212.276  |  Steps:   285648   Secs:    959      |  ⍺: 0.5000  Buffer: 100000
Episode:  3000   Avg:  618.138   BestAvg:  618.138   σ:  242.881  |  Steps:   307022   Secs:   1029      |  ⍺: 0.5000  Buffer: 100000
Episode:  3100   Avg:  400.380   BestAvg:  647.977   σ:  334.775  |  Steps:   330219   Secs:   1104      |  ⍺: 0.5000  Buffer: 100000
Episode:  3200   Avg:  637.605   BestAvg:  666.216   σ:  520.943  |  Steps:   354701   Secs:   1183      |  ⍺: 0.5000  Buffer: 100000
Episode:  3300   Avg:  410.399   BestAvg:  666.216   σ:  183.190  |  Steps:   369766   Secs:   1231      |  ⍺: 0.5000  Buffer: 100000
Episode:  3400   Avg:  322.867   BestAvg:  666.216   σ:  172.299  |  Steps:   382161   Secs:   1271      |  ⍺: 0.5000  Buffer: 100000
Episode:  3500   Avg:  360.719   BestAvg:  666.216   σ:  175.933  |  Steps:   395456   Secs:   1313      |  ⍺: 0.5000  Buffer: 100000
Episode:  3600   Avg:  379.353   BestAvg:  666.216   σ:  167.542  |  Steps:   409697   Secs:   1359      |  ⍺: 0.5000  Buffer: 100000
Episode:  3700   Avg:  604.428   BestAvg:  666.216   σ:  224.938  |  Steps:   431449   Secs:   1431      |  ⍺: 0.5000  Buffer: 100000
Episode:  3800   Avg:  585.776   BestAvg:  666.216   σ:  385.132  |  Steps:   452752   Secs:   1501      |  ⍺: 0.5000  Buffer: 100000
Episode:  3900   Avg:  448.132   BestAvg:  701.296   σ:  364.162  |  Steps:   470695   Secs:   1565      |  ⍺: 0.5000  Buffer: 100000
Episode:  4000   Avg:  473.555   BestAvg:  701.296   σ:  371.537  |  Steps:   488663   Secs:   1633      |  ⍺: 0.5000  Buffer: 100000
Episode:  4100   Avg:  911.663   BestAvg:  917.016   σ:  454.886  |  Steps:   520045   Secs:   1776      |  ⍺: 0.5000  Buffer: 100000
Episode:  4200   Avg:  804.592   BestAvg:  949.221   σ:  274.306  |  Steps:   546422   Secs:   1878      |  ⍺: 0.5000  Buffer: 100000
Episode:  4300   Avg: 1150.222   BestAvg: 1188.679   σ:  619.924  |  Steps:   585598   Secs:   2010      |  ⍺: 0.5000  Buffer: 100000
Episode:  4400   Avg: 1057.282   BestAvg: 1188.679   σ:  697.619  |  Steps:   620740   Secs:   2123      |  ⍺: 0.5000  Buffer: 100000
Episode:  4500   Avg:  798.750   BestAvg: 1188.679   σ:  460.404  |  Steps:   647110   Secs:   2209      |  ⍺: 0.5000  Buffer: 100000
Episode:  4600   Avg:  669.513   BestAvg: 1188.679   σ:  546.021  |  Steps:   673154   Secs:   2334      |  ⍺: 0.5000  Buffer: 100000
Episode:  4700   Avg: 1126.431   BestAvg: 1188.679   σ:  849.078  |  Steps:   712135   Secs:   2466      |  ⍺: 0.5000  Buffer: 100000
Episode:  4800   Avg: 1213.789   BestAvg: 1290.116   σ:  816.850  |  Steps:   750725   Secs:   2592      |  ⍺: 0.5000  Buffer: 100000
Episode:  4900   Avg:  811.883   BestAvg: 1290.116   σ:  431.415  |  Steps:   776136   Secs:   2677      |  ⍺: 0.5000  Buffer: 100000
Episode:  5000   Avg:  767.793   BestAvg: 1290.116   σ:  394.317  |  Steps:   801207   Secs:   2760      |  ⍺: 0.5000  Buffer: 100000
Episode:  5100   Avg:  843.050   BestAvg: 1290.116   σ:  593.476  |  Steps:   828604   Secs:   2849      |  ⍺: 0.5000  Buffer: 100000
Episode:  5200   Avg:  905.920   BestAvg: 1290.116   σ:  635.483  |  Steps:   858337   Secs:   2948      |  ⍺: 0.5000  Buffer: 100000
Episode:  5300   Avg:  788.695   BestAvg: 1290.116   σ:  390.313  |  Steps:   884763   Secs:   3049      |  ⍺: 0.5000  Buffer: 100000
Episode:  5400   Avg:  763.667   BestAvg: 1290.116   σ:  399.238  |  Steps:   909435   Secs:   3129      |  ⍺: 0.5000  Buffer: 100000
Episode:  5500   Avg:  866.257   BestAvg: 1290.116   σ:  601.552  |  Steps:   936879   Secs:   3218      |  ⍺: 0.5000  Buffer: 100000
Episode:  5600   Avg:  905.122   BestAvg: 1290.116   σ:  720.502  |  Steps:   966620   Secs:   3316      |  ⍺: 0.5000  Buffer: 100000
Episode:  5700   Avg:  798.135   BestAvg: 1290.116   σ:  460.974  |  Steps:   990998   Secs:   3396      |  ⍺: 0.5000  Buffer: 100000
Episode:  5800   Avg:  942.687   BestAvg: 1290.116   σ:  534.575  |  Steps:  1019212   Secs:   3489      |  ⍺: 0.5000  Buffer: 100000
Episode:  5900   Avg: 1085.581   BestAvg: 1290.116   σ:  717.517  |  Steps:  1051742   Secs:   3597      |  ⍺: 0.5000  Buffer: 100000
Episode:  6000   Avg: 1244.672   BestAvg: 1349.209   σ:  784.898  |  Steps:  1090991   Secs:   3728      |  ⍺: 0.5000  Buffer: 100000
Episode:  6100   Avg: 1067.496   BestAvg: 1349.209   σ:  738.391  |  Steps:  1123868   Secs:   3839      |  ⍺: 0.5000  Buffer: 100000
Episode:  6200   Avg:  883.891   BestAvg: 1349.209   σ:  585.302  |  Steps:  1151315   Secs:   3928      |  ⍺: 0.5000  Buffer: 100000
Episode:  6300   Avg:  573.562   BestAvg: 1349.209   σ:  288.115  |  Steps:  1170504   Secs:   3991      |  ⍺: 0.5000  Buffer: 100000
Episode:  6400   Avg:  901.654   BestAvg: 1349.209   σ:  315.187  |  Steps:  1197723   Secs:   4082      |  ⍺: 0.5000  Buffer: 100000
Episode:  6500   Avg:  671.278   BestAvg: 1349.209   σ:  457.223  |  Steps:  1218686   Secs:   4153      |  ⍺: 0.5000  Buffer: 100000
Episode:  6600   Avg:  941.456   BestAvg: 1349.209   σ:  529.887  |  Steps:  1247657   Secs:   4248      |  ⍺: 0.5000  Buffer: 100000
Episode:  6700   Avg:  991.667   BestAvg: 1349.209   σ:  602.428  |  Steps:  1278754   Secs:   4349      |  ⍺: 0.5000  Buffer: 100000
Episode:  6800   Avg: 1007.428   BestAvg: 1349.209   σ:  692.146  |  Steps:  1311900   Secs:   4459      |  ⍺: 0.5000  Buffer: 100000
Episode:  6900   Avg:  786.216   BestAvg: 1349.209   σ:  451.793  |  Steps:  1336340   Secs:   4539      |  ⍺: 0.5000  Buffer: 100000
Episode:  7000   Avg:  821.682   BestAvg: 1349.209   σ:  280.118  |  Steps:  1362098   Secs:   4639      |  ⍺: 0.5000  Buffer: 100000
Episode:  7100   Avg:  834.133   BestAvg: 1349.209   σ:  259.233  |  Steps:  1388016   Secs:   4726      |  ⍺: 0.5000  Buffer: 100000
Episode:  7200   Avg: 1114.554   BestAvg: 1349.209   σ:  600.834  |  Steps:  1423080   Secs:   4843      |  ⍺: 0.5000  Buffer: 100000
Episode:  7300   Avg: 1066.175   BestAvg: 1349.209   σ:  702.283  |  Steps:  1456085   Secs:   4950      |  ⍺: 0.5000  Buffer: 100000
Episode:  7400   Avg:  759.361   BestAvg: 1349.209   σ:  400.842  |  Steps:  1479404   Secs:   5026      |  ⍺: 0.5000  Buffer: 100000
Episode:  7500   Avg:  809.096   BestAvg: 1349.209   σ:  612.593  |  Steps:  1508063   Secs:   5123      |  ⍺: 0.5000  Buffer: 100000
Episode:  7600   Avg:  803.616   BestAvg: 1349.209   σ:  585.140  |  Steps:  1533596   Secs:   5222      |  ⍺: 0.5000  Buffer: 100000
Episode:  7700   Avg:  476.350   BestAvg: 1349.209   σ:  398.829  |  Steps:  1548447   Secs:   5281      |  ⍺: 0.5000  Buffer: 100000
Episode:  7800   Avg:  657.433   BestAvg: 1349.209   σ:  323.071  |  Steps:  1569228   Secs:   5367      |  ⍺: 0.5000  Buffer: 100000
Episode:  7900   Avg:  902.887   BestAvg: 1349.209   σ:  474.174  |  Steps:  1597509   Secs:   5461      |  ⍺: 0.5000  Buffer: 100000
Episode:  8000   Avg:  951.418   BestAvg: 1349.209   σ:  496.702  |  Steps:  1627941   Secs:   5563      |  ⍺: 0.5000  Buffer: 100000
Episode:  8100   Avg:  900.620   BestAvg: 1349.209   σ:  674.361  |  Steps:  1656989   Secs:   5659      |  ⍺: 0.5000  Buffer: 100000
Episode:  8200   Avg:  854.068   BestAvg: 1349.209   σ:  577.825  |  Steps:  1684151   Secs:   5749      |  ⍺: 0.5000  Buffer: 100000
Episode:  8300   Avg: 1108.433   BestAvg: 1349.209   σ:  699.437  |  Steps:  1718791   Secs:   5864      |  ⍺: 0.5000  Buffer: 100000
Episode:  8400   Avg:  806.895   BestAvg: 1349.209   σ:  596.620  |  Steps:  1744988   Secs:   5951      |  ⍺: 0.5000  Buffer: 100000
Episode:  8500   Avg:  807.478   BestAvg: 1349.209   σ:  620.882  |  Steps:  1771918   Secs:   6037      |  ⍺: 0.5000  Buffer: 100000
Episode:  8600   Avg:  899.211   BestAvg: 1349.209   σ:  599.331  |  Steps:  1800629   Secs:   6130      |  ⍺: 0.5000  Buffer: 100000
Episode:  8700   Avg: 1017.593   BestAvg: 1349.209   σ:  591.528  |  Steps:  1833264   Secs:   6236      |  ⍺: 0.5000  Buffer: 100000
Episode:  8800   Avg:  913.638   BestAvg: 1349.209   σ:  508.859  |  Steps:  1862159   Secs:   6330      |  ⍺: 0.5000  Buffer: 100000
Episode:  8900   Avg:  987.499   BestAvg: 1349.209   σ:  695.964  |  Steps:  1894911   Secs:   6435      |  ⍺: 0.5000  Buffer: 100000
Episode:  9000   Avg:  818.957   BestAvg: 1349.209   σ:  404.901  |  Steps:  1920897   Secs:   6519      |  ⍺: 0.5000  Buffer: 100000
```

#### DDPG Evaluation
```
Loaded: checkpoints/last_run/episode.6100
Episode:    50   Avg: 3272.664   BestAvg:     -inf   σ:  814.936  |  Steps:     1000   Reward: 3471.522  |  ⍺: 0.5000  Buffer:  45625
```
