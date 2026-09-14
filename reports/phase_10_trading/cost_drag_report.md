# Cost drag report

                          Model  gross   net  cost_drag_pp  exposure  turnover  sharpe_net  maxdd
                always-positive  25.32 24.82          0.50    100.00    1.0000       0.687  23.77
                always-negative   7.53  7.53          0.00      0.00    0.0000       0.000   0.00
                         random  21.36 12.47          8.89     38.67    0.3867       0.511  10.50
                   previous-day  23.19 19.31          3.88     40.00    0.4000       1.174   5.23
                    momentum-5d  29.62 28.07          1.55     54.67    0.5467       1.308   5.82
                   momentum-10d   7.53  7.53          0.00      0.00    0.0000       0.000   0.00
                   ma-crossover  21.61 19.68          1.93     21.33    0.2133       1.318   1.35
                         logreg  33.11 22.38         10.74     48.00    0.4800       1.050   6.39
                  random-forest  30.57 27.47          3.10     92.00    0.9200       0.858  18.66
                  grad-boosting   8.46  5.89          2.57     18.67    0.1867       0.020   8.71
     Pure LSTM (Technical only)  25.32 24.82          0.50    100.00    1.0000       0.687  23.77
                   LSTM + Macro  25.32 24.82          0.50    100.00    1.0000       0.687  23.77
                    LSTM + News  13.50 10.37          3.14     26.67    0.2667       0.286  13.32
           LSTM + Static Fusion  25.32 24.82          0.50    100.00    1.0000       0.687  23.77
Proposed (Adaptive Soft Gating)   9.03  3.50          5.53     53.33    0.5333       0.022  25.05

- Highest cost drag: logreg (10.74 pp) — high turnover + low edge ⇒ costs dominate.
- Proposed net (3.50) vs gross (9.03): 5.53 pp drag wipes out >60% of gross edge.
- momentum-5d: 1.55 pp drag, Sharpe 1.308 net — cheapest edge per trade.
- Any model with exposure ≈100% (PureLSTM/LSTM+Macro/Static here) is buy-and-hold with extra costs; its 'return' is benchmark return minus drag.
