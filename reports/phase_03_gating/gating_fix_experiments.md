# Gating fix experiments (test-side reporting for diagnosis; selection must use VAL)

- baseline: AUC=0.4921 MCC=-0.0152 p_std=0.00205 ent=0.8634 gates=[0.139, 0.666, 0.195]
- temperature-2.0: AUC=0.51 MCC=0.0 p_std=0.00195 ent=1.0854 gates=[0.355, 0.385, 0.26]
- min-weight-0.05: AUC=0.5122 MCC=0.0 p_std=0.00299 ent=1.0291 gates=[0.367, 0.46, 0.174]
- entropy-0.01: AUC=0.5206 MCC=0.0 p_std=0.00226 ent=1.0938 gates=[0.345, 0.366, 0.289]
- entropy-0.05: AUC=0.5149 MCC=0.0 p_std=0.00244 ent=1.0984 gates=[0.338, 0.337, 0.325]
- modality-dropout-0.1: AUC=0.5265 MCC=0.0 p_std=0.00147 ent=1.0845 gates=[0.261, 0.345, 0.395]
- aux-0.1: AUC=0.3794 MCC=-0.0908 p_std=0.00276 ent=0.6454 gates=[0.145, 0.789, 0.066]
- combined(temp2+minw05+ent01): AUC=0.5275 MCC=0.0941 p_std=0.0031 ent=1.0949 gates=[0.332, 0.369, 0.299]
