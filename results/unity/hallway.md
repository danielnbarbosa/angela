#### Notes
- this environment is not reproducing consistent results with a fixed seed.  maybe something broken in env?
- setting noise non-zero (beta=.01) helped a lot
- agent is not taking box color into account and always moving to same goal.
- this may be due to unbalanced rewards: +1 For moving to correct goal. -0.1 For moving to incorrect goal.
- still aways off from solve score of 0.7
- changing beta_decay from 0.995 to 0.996 killed performance

#### PPO
```
Episode:   100   Avg:    -0.99   BestAvg:     -inf   σ:     0.03  |  Steps:    50000   Secs:    297      |  ε: 0.09048   β:   0.006058
Episode:   200   Avg:    -0.97   BestAvg:    -0.97   σ:     0.05  |  Steps:   100000   Secs:    596      |  ε: 0.08186   β:   0.003678
Episode:   300   Avg:    -0.29   BestAvg:    -0.29   σ:     0.37  |  Steps:   148030   Secs:    887      |  ε: 0.07407   β:   0.002223
Episode:   400   Avg:     0.00   BestAvg:     0.04   σ:     0.19  |  Steps:   195229   Secs:   1174      |  ε: 0.06702   β:   0.001347
Episode:   500   Avg:     0.05   BestAvg:     0.08   σ:     0.19  |  Steps:   235552   Secs:   1410      |  ε: 0.06064   β:   0.0008157
Episode:   600   Avg:    -0.15   BestAvg:     0.08   σ:     0.17  |  Steps:   276433   Secs:   1635      |  ε: 0.05486   β:   0.0004941
Episode:   700   Avg:    -0.46   BestAvg:     0.08   σ:     0.17  |  Steps:   325877   Secs:   1909      |  ε: 0.04964   β:   0.0002993
Episode:   800   Avg:    -0.30   BestAvg:     0.08   σ:     0.13  |  Steps:   372277   Secs:   2159      |  ε: 0.04491   β:   0.0001813
Episode:   900   Avg:    -0.32   BestAvg:     0.08   σ:     0.15  |  Steps:   418638   Secs:   2399      |  ε: 0.04064   β:   0.0001098
```
