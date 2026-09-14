# Statistical tests (fold-level AUC, n=6 per model)

- PureLSTM vs Proposed-stabilized: Wilcoxon p=0.4375, paired-t p=0.9212032060664057 (mean 0.5575 vs 0.5560)
- LSTM+Macro vs Proposed-stabilized: Wilcoxon p=0.09375, paired-t p=0.061500453992343516 (mean 0.5878 vs 0.5560)
- LSTM+News vs Proposed-stabilized: Wilcoxon p=0.15625, paired-t p=0.07753041791195825 (mean 0.5154 vs 0.5560)
- Bootstrap 95% CI for (LSTM+Macro − Proposed) net return: [-3.18, 9.89]
