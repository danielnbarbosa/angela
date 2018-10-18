#### Notes
- PG results are from pong_pg (I think).
- PPO results are from pong_ppo_gpu config (I think).
- ran pong_pg_03 for 50000 episodes got to 0.31 during train, -6.09 on eval (over 11 episodes)


#### PG
```
Episode   100   Avg:  -20.24   BestAvg:    -inf   σ:    0.91   |   Steps:   126138  Secs:    345
Episode   200   Avg:  -19.88   BestAvg:  -19.80   σ:    1.24   |   Steps:   268926  Secs:    713
Episode   300   Avg:  -19.23   BestAvg:  -19.19   σ:    1.36   |   Steps:   437749  Secs:   1147
Episode   400   Avg:  -18.81   BestAvg:  -18.77   σ:    1.46   |   Steps:   642895  Secs:   1692
Episode   500   Avg:  -18.52   BestAvg:  -18.49   σ:    1.72   |   Steps:   875870  Secs:   2327
Episode   600   Avg:  -18.57   BestAvg:  -18.30   σ:    1.57   |   Steps:  1129086  Secs:   2989
Episode   700   Avg:  -18.09   BestAvg:  -18.09   σ:    1.87   |   Steps:  1400105  Secs:   3696
Episode   800   Avg:  -18.04   BestAvg:  -17.76   σ:    1.76   |   Steps:  1692795  Secs:   4462
Episode   900   Avg:  -17.44   BestAvg:  -17.37   σ:    1.82   |   Steps:  1982316  Secs:   5221
Episode  1000   Avg:  -17.23   BestAvg:  -17.20   σ:    1.98   |   Steps:  2283225  Secs:   6048
Episode  1100   Avg:  -16.46   BestAvg:  -16.45   σ:    2.37   |   Steps:  2599824  Secs:   6873
Episode  1200   Avg:  -16.04   BestAvg:  -15.95   σ:    2.40   |   Steps:  2927577  Secs:   7723
Episode  1300   Avg:  -15.87   BestAvg:  -15.76   σ:    2.40   |   Steps:  3256860  Secs:   8579
Episode  1400   Avg:  -15.05   BestAvg:  -15.05   σ:    2.40   |   Steps:  3597507  Secs:   9509
Episode  1500   Avg:  -14.96   BestAvg:  -14.68   σ:    2.52   |   Steps:  3945640  Secs:  10413
Episode  1600   Avg:  -13.50   BestAvg:  -13.50   σ:    3.10   |   Steps:  4309188  Secs:  11432
Episode  1700   Avg:  -12.93   BestAvg:  -12.84   σ:    2.82   |   Steps:  4685104  Secs:  12575
Episode  1800   Avg:  -11.21   BestAvg:  -11.21   σ:    3.55   |   Steps:  5075375  Secs:  13912
Episode  1900   Avg:  -11.67   BestAvg:  -10.86   σ:    3.53   |   Steps:  5471946  Secs:  15155
Episode  2000   Avg:  -10.10   BestAvg:  -10.01   σ:    3.19   |   Steps:  5873829  Secs:  16291
Episode  2100   Avg:   -9.31   BestAvg:   -9.20   σ:    3.48   |   Steps:  6287365  Secs:  17365
Episode  2200   Avg:   -8.64   BestAvg:   -8.64   σ:    3.43   |   Steps:  6712483  Secs:  18467
Episode  2300   Avg:   -8.56   BestAvg:   -8.44   σ:    3.40   |   Steps:  7149874  Secs:  19600
Episode  2400   Avg:   -8.08   BestAvg:   -7.51   σ:    4.00   |   Steps:  7587002  Secs:  20780
Episode  2500   Avg:   -6.92   BestAvg:   -6.76   σ:    4.42   |   Steps:  8025336  Secs:  21915
Episode  2600   Avg:   -6.35   BestAvg:   -6.32   σ:    4.15   |   Steps:  8484818  Secs:  23101
Episode  2700   Avg:   -5.54   BestAvg:   -5.35   σ:    4.22   |   Steps:  8955355  Secs:  24371
Episode  2800   Avg:   -6.89   BestAvg:   -5.35   σ:    4.13   |   Steps:  9450537  Secs:  25703
Episode  2900   Avg:   -5.76   BestAvg:   -5.35   σ:    4.52   |   Steps:  9942068  Secs:  27146
Episode  3000   Avg:   -4.79   BestAvg:   -4.78   σ:    4.60   |   Steps: 10432969  Secs:  28546
Episode  3100   Avg:   -4.64   BestAvg:   -3.96   σ:    4.32   |   Steps: 10934019  Secs:  29931
Episode  3200   Avg:   -5.57   BestAvg:   -3.96   σ:    4.35   |   Steps: 11431836  Secs:  31267
Episode  3300   Avg:   -4.97   BestAvg:   -3.96   σ:    4.25   |   Steps: 11936077  Secs:  32575
Episode  3400   Avg:   -4.62   BestAvg:   -3.96   σ:    4.74   |   Steps: 12447091  Secs:  33898
Episode  3500   Avg:   -3.85   BestAvg:   -3.77   σ:    4.33   |   Steps: 12971218  Secs:  35325
Episode  3600   Avg:   -3.53   BestAvg:   -3.53   σ:    5.09   |   Steps: 13511663  Secs:  36721
Episode  3700   Avg:   -2.89   BestAvg:   -2.57   σ:    4.77   |   Steps: 14049017  Secs:  38154
Episode  3800   Avg:   -2.67   BestAvg:   -2.12   σ:    4.86   |   Steps: 14594892  Secs:  39569
Episode  3900   Avg:   -3.58   BestAvg:   -2.12   σ:    4.75   |   Steps: 15147099  Secs:  40997
Episode  4000   Avg:   -3.27   BestAvg:   -2.12   σ:    4.87   |   Steps: 15706010  Secs:  42488
Episode  4100   Avg:   -3.19   BestAvg:   -2.12   σ:    4.64   |   Steps: 16264224  Secs:  43930
Episode  4200   Avg:   -2.67   BestAvg:   -2.12   σ:    4.73   |   Steps: 16789437  Secs:  45325
Episode  4300   Avg:   -2.15   BestAvg:   -1.71   σ:    4.76   |   Steps: 17315071  Secs:  46688
Episode  4400   Avg:   -2.90   BestAvg:   -1.71   σ:    4.93   |   Steps: 17840259  Secs:  48050
Episode  4500   Avg:   -0.33   BestAvg:   -0.33   σ:    4.58   |   Steps: 18382697  Secs:  49499
Episode  4600   Avg:   -1.91   BestAvg:   -0.31   σ:    4.68   |   Steps: 18905931  Secs:  50852
Episode  4700   Avg:   -0.92   BestAvg:   -0.31   σ:    4.77   |   Steps: 19439968  Secs:  52223
Episode  4800   Avg:   -0.54   BestAvg:   -0.31   σ:    4.96   |   Steps: 19982280  Secs:  53735
Episode  4900   Avg:   -0.42   BestAvg:   -0.31   σ:    5.38   |   Steps: 20520505  Secs:  55284
Episode  5000   Avg:    0.00   BestAvg:    0.68   σ:    5.23   |   Steps: 21036535  Secs:  56747
Episode  5100   Avg:   -0.87   BestAvg:    0.68   σ:    4.63   |   Steps: 21588941  Secs:  58415
Episode  5200   Avg:   -0.96   BestAvg:    0.68   σ:    4.53   |   Steps: 22130751  Secs:  60010
Episode  5300   Avg:   -0.88   BestAvg:    0.68   σ:    5.02   |   Steps: 22673427  Secs:  61537
Episode  5400   Avg:    0.33   BestAvg:    0.68   σ:    4.86   |   Steps: 23227248  Secs:  63206
Episode  5500   Avg:    0.15   BestAvg:    0.68   σ:    5.30   |   Steps: 23754741  Secs:  64767
Episode  5600   Avg:    0.43   BestAvg:    0.78   σ:    5.30   |   Steps: 24295917  Secs:  66341
Episode  5700   Avg:    0.76   BestAvg:    0.94   σ:    4.34   |   Steps: 24830136  Secs:  68154
Episode  5800   Avg:    0.04   BestAvg:    0.94   σ:    4.88   |   Steps: 25373625  Secs:  69718
Episode  5900   Avg:    0.42   BestAvg:    0.94   σ:    4.92   |   Steps: 25928068  Secs:  71213
Episode  6000   Avg:    0.51   BestAvg:    0.94   σ:    4.75   |   Steps: 26506925  Secs:  72752
Episode  6100   Avg:    1.07   BestAvg:    1.27   σ:    4.67   |   Steps: 27077945  Secs:  74310
Episode  6200   Avg:    0.99   BestAvg:    1.39   σ:    5.46   |   Steps: 27664086  Secs:  75916
Episode  6300   Avg:    1.17   BestAvg:    1.39   σ:    5.02   |   Steps: 28242604  Secs:  77606
Episode  6400   Avg:    1.41   BestAvg:    1.67   σ:    4.74   |   Steps: 28791872  Secs:  79184
Episode  6500   Avg:    0.99   BestAvg:    1.67   σ:    4.44   |   Steps: 29345070  Secs:  80626
Episode  6600   Avg:    0.70   BestAvg:    1.67   σ:    5.32   |   Steps: 29912134  Secs:  82189
Episode  6700   Avg:    1.69   BestAvg:    1.87   σ:    5.06   |   Steps: 30478557  Secs:  83789
Episode  6800   Avg:    1.29   BestAvg:    2.12   σ:    4.48   |   Steps: 31041912  Secs:  85424
Episode  6900   Avg:    1.52   BestAvg:    2.12   σ:    5.08   |   Steps: 31614525  Secs:  87088
Episode  7000   Avg:    0.92   BestAvg:    2.12   σ:    5.07   |   Steps: 32187734  Secs:  88674
Episode  7100   Avg:    1.21   BestAvg:    2.12   σ:    5.12   |   Steps: 32740294  Secs:  90106
Episode  7200   Avg:    2.12   BestAvg:    2.71   σ:    5.00   |   Steps: 33302296  Secs:  91562
Episode  7300   Avg:    1.69   BestAvg:    2.71   σ:    4.99   |   Steps: 33900517  Secs:  93343
Episode  7400   Avg:    0.85   BestAvg:    2.71   σ:    4.44   |   Steps: 34512268  Secs:  95101
Episode  7500   Avg:    1.29   BestAvg:    2.71   σ:    5.43   |   Steps: 35098666  Secs:  96807
Episode  7600   Avg:    1.01   BestAvg:    2.71   σ:    5.16   |   Steps: 35673558  Secs:  98379
Episode  7700   Avg:    1.98   BestAvg:    2.71   σ:    4.86   |   Steps: 36256709  Secs: 100215
Episode  7800   Avg:    1.35   BestAvg:    2.97   σ:    4.39   |   Steps: 36813940  Secs: 101909
Episode  7900   Avg:    2.14   BestAvg:    2.97   σ:    5.11   |   Steps: 37385256  Secs: 103558
Episode  8000   Avg:    2.28   BestAvg:    3.48   σ:    6.02   |   Steps: 37932477  Secs: 105031
Episode  8100   Avg:    1.73   BestAvg:    3.48   σ:    5.23   |   Steps: 38529342  Secs: 106608
Episode  8200   Avg:    1.25   BestAvg:    3.48   σ:    4.85   |   Steps: 39111070  Secs: 108381
Episode  8300   Avg:    2.13   BestAvg:    3.48   σ:    4.97   |   Steps: 39692956  Secs: 110160
Episode  8400   Avg:    2.04   BestAvg:    3.48   σ:    5.08   |   Steps: 40272475  Secs: 112066
Saved to checkpoints/best/pong_conv2dsmall_fc200.pth

Loaded: checkpoints/best/pong_conv2dsmall_fc200.pth
Episode   100   Avg:     1.07   BestAvg:     -inf   σ:     4.74   |   Steps:   561786  Secs:   1686
Episode   200   Avg:     2.19   BestAvg:     2.57   σ:     5.03   |   Steps:  1150253  Secs:   3374
Episode   300   Avg:     2.01   BestAvg:     2.57   σ:     5.18   |   Steps:  1717129  Secs:   5040
Episode   400   Avg:     1.86   BestAvg:     2.64   σ:     4.60   |   Steps:  2270654  Secs:   6626
Episode   500   Avg:     2.75   BestAvg:     2.77   σ:     4.60   |   Steps:  2837772  Secs:   8295
Episode   600   Avg:     2.60   BestAvg:     3.36   σ:     4.63   |   Steps:  3404221  Secs:   9919
Episode   700   Avg:     2.32   BestAvg:     3.36   σ:     4.48   |   Steps:  3988634  Secs:  11626
Episode   800   Avg:     2.67   BestAvg:     3.36   σ:     5.05   |   Steps:  4557696  Secs:  13256
Episode   900   Avg:     2.64   BestAvg:     3.36   σ:     4.96   |   Steps:  5099538  Secs:  14804
Episode  1000   Avg:     2.94   BestAvg:     3.82   σ:     4.93   |   Steps:  5657200  Secs:  16439
Episode  1100   Avg:     2.41   BestAvg:     3.82   σ:     4.81   |   Steps:  6209780  Secs:  18022
Episode  1200   Avg:     2.26   BestAvg:     3.82   σ:     5.00   |   Steps:  6765091  Secs:  19651
Episode  1300   Avg:     2.66   BestAvg:     3.82   σ:     4.78   |   Steps:  7314662  Secs:  21222
Episode  1400   Avg:     3.69   BestAvg:     4.25   σ:     5.06   |   Steps:  7877658  Secs:  22873
Episode  1500   Avg:     3.21   BestAvg:     4.25   σ:     4.60   |   Steps:  8440556  Secs:  24478
Episode  1600   Avg:     3.37   BestAvg:     4.25   σ:     4.99   |   Steps:  9004292  Secs:  26136
Episode  1700   Avg:     3.12   BestAvg:     4.25   σ:     4.96   |   Steps:  9560902  Secs:  27906
Episode  1800   Avg:     2.87   BestAvg:     4.25   σ:     4.76   |   Steps: 10116194  Secs:  29518
Episode  1900   Avg:     2.25   BestAvg:     4.25   σ:     4.96   |   Steps: 10663455  Secs:  31127
Episode  2000   Avg:     1.62   BestAvg:     4.25   σ:     5.28   |   Steps: 11218080  Secs:  32729
Episode  2100   Avg:     2.83   BestAvg:     4.25   σ:     5.14   |   Steps: 11766714  Secs:  34345
Episode  2200   Avg:     3.71   BestAvg:     4.25   σ:     4.69   |   Steps: 12313118  Secs:  35915
Saved to checkpoints/best/pong_conv2dsmall_fc200_2.pth
```

#### PPO
```
Episode   100   Avg:   -14.44   BestAvg:     -inf   σ:     2.68   |   ε:  0.181   β:   0.01212   Steps:   203865  Secs:   3181
Episode   200   Avg:   -13.46   BestAvg:   -12.58   σ:     1.17   |   ε: 0.1637   β:   0.007339   Steps:   421166  Secs:   6462
Loaded: checkpoints/last_run/episode.200.pth
Episode   100   Avg:    -8.95   BestAvg:     -inf   σ:     1.18   |   ε:  0.181   β:   0.01212   Steps:   199511  Secs:   3063
Episode   200   Avg:    -7.26   BestAvg:    -7.24   σ:     1.02   |   ε: 0.1637   β:   0.007339   Steps:   399411  Secs:   6122
Episode   300   Avg:    -9.58   BestAvg:    -7.23   σ:     2.03   |   ε: 0.1481   β:   0.004446   Steps:   597972  Secs:   9158
Episode   300   Avg:   -12.86   BestAvg:    -8.37   σ:     2.69   |   ε: 0.1481   β:   0.004446   Steps:   586557  Secs:   8974
Episode   400   Avg:   -10.53   BestAvg:    -8.37   σ:     2.02   |   ε:  0.134   β:   0.002693   Steps:   783070  Secs:  11974
Episode   500   Avg:    -9.15   BestAvg:    -8.37   σ:     1.24   |   ε: 0.1213   β:   0.001631   Steps:   982721  Secs:  15015
```
