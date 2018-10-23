#### Notes
- can't set max_t too high or else it triggers nans due to bug in environment reward function
- done triggered by agent hitting head on floor
- agents continue to gather reward after entering done state unless handled properly, currently setting rewards to 0 after done
- helpful to separate training from evaluation
- ideally agent would be able to learn a stable gait with a low max_t while training, then this gait will be repeated evaluating
- however during evaluation agent frequently flips itself upside down and gets stuck, a state it doesn't see during training with low max_t
- sometimes does better during evaluation with noise on
- solve criteria is a score of 2000 when max_t is 1000
- you get ~ +50 just for sitting still, perhaps facing the target?
- helpful too see what agent is doing during training to get a feel for how it's getting stuck
- max_t of 200 triggers nan.  max_t of 70 does not.
- current hyperparamaters somehow result in nice walking gait instead of jumping spider

#### Ideas
- add -100 reward on done.  current reward on done is -1.  easily drowned out by large positive rewards.
- adjust learning rate to handle high rewards better.
- tweak standard deviation of noise.
- decay noise

#### DDPG Training
```
Episode:   100   Avg:   -0.646   BestAvg:     -inf   σ:    0.788  |  Steps:     7568   Secs:     76      |  ⍺: 0.5000  Buffer:  92016
Episode:   200   Avg:    1.013   BestAvg:    1.013   σ:    0.827  |  Steps:    17344   Secs:    176      |  ⍺: 0.5000  Buffer: 210528
Episode:   300   Avg:    0.574   BestAvg:    1.036   σ:    0.673  |  Steps:    24570   Secs:    253      |  ⍺: 0.5000  Buffer: 298440
Episode:   400   Avg:    0.970   BestAvg:    1.036   σ:    0.682  |  Steps:    33097   Secs:    339      |  ⍺: 0.5000  Buffer: 300000
Episode:   500   Avg:    1.242   BestAvg:    1.303   σ:    0.850  |  Steps:    42927   Secs:    436      |  ⍺: 0.5000  Buffer: 300000
Episode:   600   Avg:    1.764   BestAvg:    1.786   σ:    0.953  |  Steps:    52827   Secs:    539      |  ⍺: 0.5000  Buffer: 300000
Episode:   700   Avg:    2.493   BestAvg:    2.515   σ:    1.064  |  Steps:    62706   Secs:    644      |  ⍺: 0.5000  Buffer: 300000
Episode:   800   Avg:    2.681   BestAvg:    2.973   σ:    1.407  |  Steps:    72592   Secs:    747      |  ⍺: 0.5000  Buffer: 300000
Episode:   900   Avg:    4.236   BestAvg:    4.236   σ:    1.669  |  Steps:    82466   Secs:    844      |  ⍺: 0.5000  Buffer: 300000
Episode:  1000   Avg:    8.101   BestAvg:    8.101   σ:    1.735  |  Steps:    92366   Secs:    945      |  ⍺: 0.5000  Buffer: 300000
Episode:  1100   Avg:    9.896   BestAvg:    9.896   σ:    2.243  |  Steps:   102266   Secs:   1047      |  ⍺: 0.5000  Buffer: 300000
Episode:  1200   Avg:    9.864   BestAvg:   11.496   σ:    3.711  |  Steps:   112166   Secs:   1150      |  ⍺: 0.5000  Buffer: 300000
Episode:  1300   Avg:   11.815   BestAvg:   11.906   σ:    2.307  |  Steps:   122066   Secs:   1248      |  ⍺: 0.5000  Buffer: 300000
Episode:  1400   Avg:   14.512   BestAvg:   14.512   σ:    2.315  |  Steps:   131966   Secs:   1344      |  ⍺: 0.5000  Buffer: 300000
Episode:  1500   Avg:   17.431   BestAvg:   17.431   σ:    2.726  |  Steps:   141866   Secs:   1446      |  ⍺: 0.5000  Buffer: 300000
Episode:  1600   Avg:   20.271   BestAvg:   20.271   σ:    3.073  |  Steps:   151766   Secs:   1549      |  ⍺: 0.5000  Buffer: 300000
Episode:  1700   Avg:   21.043   BestAvg:   21.428   σ:    3.572  |  Steps:   161666   Secs:   1652      |  ⍺: 0.5000  Buffer: 300000
Episode:  1800   Avg:   24.719   BestAvg:   24.868   σ:    3.401  |  Steps:   171566   Secs:   1747      |  ⍺: 0.5000  Buffer: 300000
Episode:  1900   Avg:   25.569   BestAvg:   25.793   σ:    3.357  |  Steps:   181466   Secs:   1848      |  ⍺: 0.5000  Buffer: 300000
Episode:  2000   Avg:   24.369   BestAvg:   25.793   σ:    3.699  |  Steps:   191366   Secs:   1950      |  ⍺: 0.5000  Buffer: 300000
Episode:  2100   Avg:   27.017   BestAvg:   27.112   σ:    3.732  |  Steps:   201266   Secs:   2053      |  ⍺: 0.5000  Buffer: 300000
Episode:  2200   Avg:   24.052   BestAvg:   27.179   σ:    5.347  |  Steps:   211166   Secs:   2151      |  ⍺: 0.5000  Buffer: 300000
Episode:  2300   Avg:   27.241   BestAvg:   27.241   σ:    4.164  |  Steps:   221066   Secs:   2247      |  ⍺: 0.5000  Buffer: 300000
Episode:  2400   Avg:   27.415   BestAvg:   28.249   σ:    5.059  |  Steps:   230966   Secs:   2349      |  ⍺: 0.5000  Buffer: 300000
Episode:  2500   Avg:   28.775   BestAvg:   28.775   σ:    4.528  |  Steps:   240866   Secs:   2451      |  ⍺: 0.5000  Buffer: 300000
Episode:  2600   Avg:   29.189   BestAvg:   30.220   σ:    4.430  |  Steps:   250766   Secs:   2553      |  ⍺: 0.5000  Buffer: 300000
Episode:  2700   Avg:   28.586   BestAvg:   30.220   σ:    5.446  |  Steps:   260666   Secs:   2649      |  ⍺: 0.5000  Buffer: 300000
Episode:  2800   Avg:   31.763   BestAvg:   31.763   σ:    6.314  |  Steps:   270566   Secs:   2749      |  ⍺: 0.5000  Buffer: 300000
Episode:  2900   Avg:   33.980   BestAvg:   34.205   σ:    5.815  |  Steps:   280466   Secs:   2852      |  ⍺: 0.5000  Buffer: 300000
Episode:  3000   Avg:   33.120   BestAvg:   34.458   σ:    5.614  |  Steps:   290366   Secs:   2955      |  ⍺: 0.5000  Buffer: 300000
Episode:  3100   Avg:   35.458   BestAvg:   36.066   σ:    5.385  |  Steps:   300266   Secs:   3052      |  ⍺: 0.5000  Buffer: 300000
Episode:  3200   Avg:   37.046   BestAvg:   37.046   σ:    5.460  |  Steps:   310166   Secs:   3148      |  ⍺: 0.5000  Buffer: 300000
Episode:  3300   Avg:   37.685   BestAvg:   37.956   σ:    5.266  |  Steps:   320066   Secs:   3251      |  ⍺: 0.5000  Buffer: 300000
Episode:  3400   Avg:   37.783   BestAvg:   37.956   σ:    6.810  |  Steps:   329966   Secs:   3354      |  ⍺: 0.5000  Buffer: 300000
Episode:  3500   Avg:   40.921   BestAvg:   41.251   σ:    5.215  |  Steps:   339866   Secs:   3455      |  ⍺: 0.5000  Buffer: 300000
Episode:  3600   Avg:   39.092   BestAvg:   41.251   σ:    5.745  |  Steps:   349766   Secs:   3551      |  ⍺: 0.5000  Buffer: 300000
Episode:  3700   Avg:   37.789   BestAvg:   41.251   σ:    5.502  |  Steps:   359666   Secs:   3654      |  ⍺: 0.5000  Buffer: 300000
Episode:  3800   Avg:   39.570   BestAvg:   41.251   σ:    5.854  |  Steps:   369566   Secs:   3756      |  ⍺: 0.5000  Buffer: 300000
Episode:  3900   Avg:   39.868   BestAvg:   41.251   σ:    5.249  |  Steps:   379466   Secs:   3858      |  ⍺: 0.5000  Buffer: 300000
Episode:  4000   Avg:   38.683   BestAvg:   41.251   σ:    5.188  |  Steps:   389366   Secs:   3954      |  ⍺: 0.5000  Buffer: 300000
Episode:  4100   Avg:   36.710   BestAvg:   41.251   σ:    5.238  |  Steps:   399266   Secs:   4052      |  ⍺: 0.5000  Buffer: 300000
Episode:  4200   Avg:   38.724   BestAvg:   41.251   σ:    5.150  |  Steps:   409166   Secs:   4154      |  ⍺: 0.5000  Buffer: 300000
Episode:  4300   Avg:   41.019   BestAvg:   41.251   σ:    4.716  |  Steps:   419066   Secs:   4256      |  ⍺: 0.5000  Buffer: 300000
Episode:  4400   Avg:   43.351   BestAvg:   43.507   σ:    5.466  |  Steps:   428966   Secs:   4356      |  ⍺: 0.5000  Buffer: 300000
Episode:  4500   Avg:   40.201   BestAvg:   43.507   σ:    5.523  |  Steps:   438866   Secs:   4452      |  ⍺: 0.5000  Buffer: 300000
Episode:  4600   Avg:   40.085   BestAvg:   43.507   σ:    6.104  |  Steps:   448766   Secs:   4555      |  ⍺: 0.5000  Buffer: 300000
Episode:  4700   Avg:   39.128   BestAvg:   43.507   σ:    6.571  |  Steps:   458666   Secs:   4658      |  ⍺: 0.5000  Buffer: 300000
Episode:  4800   Avg:   38.999   BestAvg:   43.507   σ:    5.286  |  Steps:   468566   Secs:   4758      |  ⍺: 0.5000  Buffer: 300000
Episode:  4900   Avg:   40.149   BestAvg:   43.507   σ:    5.501  |  Steps:   478466   Secs:   4853      |  ⍺: 0.5000  Buffer: 300000
Episode:  5000   Avg:   37.914   BestAvg:   43.507   σ:    5.306  |  Steps:   488366   Secs:   4954      |  ⍺: 0.5000  Buffer: 300000
Episode:  5100   Avg:   39.771   BestAvg:   43.507   σ:    5.299  |  Steps:   498266   Secs:   5057      |  ⍺: 0.5000  Buffer: 300000
Episode:  5200   Avg:   40.825   BestAvg:   43.507   σ:    5.497  |  Steps:   508166   Secs:   5159      |  ⍺: 0.5000  Buffer: 300000
Episode:  5300   Avg:   40.984   BestAvg:   43.507   σ:    5.485  |  Steps:   518066   Secs:   5256      |  ⍺: 0.5000  Buffer: 300000
Episode:  5400   Avg:   41.068   BestAvg:   43.507   σ:    5.172  |  Steps:   527966   Secs:   5355      |  ⍺: 0.5000  Buffer: 300000
Episode:  5500   Avg:   39.779   BestAvg:   43.507   σ:    6.371  |  Steps:   537866   Secs:   5457      |  ⍺: 0.5000  Buffer: 300000
Episode:  5600   Avg:   38.048   BestAvg:   43.507   σ:    5.597  |  Steps:   547766   Secs:   5560      |  ⍺: 0.5000  Buffer: 300000
Episode:  5700   Avg:   37.695   BestAvg:   43.507   σ:    5.044  |  Steps:   557666   Secs:   5658      |  ⍺: 0.5000  Buffer: 300000
Episode:  5800   Avg:   39.695   BestAvg:   43.507   σ:    4.858  |  Steps:   567566   Secs:   5754      |  ⍺: 0.5000  Buffer: 300000
Episode:  5900   Avg:   37.291   BestAvg:   43.507   σ:    5.192  |  Steps:   577466   Secs:   5856      |  ⍺: 0.5000  Buffer: 300000
Episode:  6000   Avg:   40.135   BestAvg:   43.507   σ:    5.814  |  Steps:   587366   Secs:   5959      |  ⍺: 0.5000  Buffer: 300000
Episode:  6100   Avg:   36.854   BestAvg:   43.507   σ:    5.185  |  Steps:   597266   Secs:   6061      |  ⍺: 0.5000  Buffer: 300000
Episode:  6200   Avg:   37.760   BestAvg:   43.507   σ:    5.978  |  Steps:   607166   Secs:   6156      |  ⍺: 0.5000  Buffer: 300000
Episode:  6300   Avg:   35.495   BestAvg:   43.507   σ:    8.290  |  Steps:   617066   Secs:   6257      |  ⍺: 0.5000  Buffer: 300000
Episode:  6400   Avg:   38.921   BestAvg:   43.507   σ:    5.306  |  Steps:   626966   Secs:   6359      |  ⍺: 0.5000  Buffer: 300000
Episode:  6500   Avg:   40.171   BestAvg:   43.507   σ:    5.084  |  Steps:   636866   Secs:   6462      |  ⍺: 0.5000  Buffer: 300000
Episode:  6600   Avg:   39.527   BestAvg:   43.507   σ:    5.081  |  Steps:   646766   Secs:   6559      |  ⍺: 0.5000  Buffer: 300000
Episode:  6700   Avg:   40.209   BestAvg:   43.507   σ:    5.048  |  Steps:   656666   Secs:   6655      |  ⍺: 0.5000  Buffer: 300000
Episode:  6800   Avg:   39.723   BestAvg:   43.507   σ:    4.650  |  Steps:   666566   Secs:   6757      |  ⍺: 0.5000  Buffer: 300000
Episode:  6900   Avg:   40.219   BestAvg:   43.507   σ:    5.055  |  Steps:   676466   Secs:   6859      |  ⍺: 0.5000  Buffer: 300000
Episode:  7000   Avg:   39.121   BestAvg:   43.507   σ:    4.967  |  Steps:   686366   Secs:   6961      |  ⍺: 0.5000  Buffer: 300000
```

#### DDPG Evaluation
```
Loaded: checkpoints/crawler_ddpg_aws_02/episode.4700
Episode:    10   Avg:  862.084   BestAvg:     -inf   σ:   57.347  |  Steps:      999   Reward:  820.314  |  ⍺: 0.5000  Buffer: 100000
```
