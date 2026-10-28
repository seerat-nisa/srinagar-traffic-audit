# Spatio-Temporal Diagnostic Audit: Srinagar Smart City Traffic Logs

Post-publication diagnostic audit of the CDAC signal-timing logs and ANPR camera event logs from Presentation Convent, Srinagar, analyzed in:
> Y. Paul, S. Ul Nisa, and R. Singh, "AI-Powered Traffic Optimization for Srinagar Smart City," *IETE Journal of Research*, 2025.

## Key Audit Findings

| Task & Dataset | Original Random Split (80/20, `seed=42`) | Root Cause Identified in Audit | Strict Temporal / Corridor Split |
| :--- | :--- | :--- | :--- |
| **Signal Green-Time Regression**<br>(`STAGE_TIMING_Convent`, $N=3,908$) | **SVR:** $R^2 = 0.9957$, $\text{MSE} = 0.0100$, $\text{MAE} = 0.1000$<br>**Linear Reg:** $R^2 = -9.0086$<br>**Random Forest:** $R^2 = -21.4525$ | Controller log is stationary (`Utilized Green = 12s`) across 3,905 cycles (`08:00:55–20:59:55`), with 3 regime transitions (`07:00:30 FLASH = 3571s`, `08:00:07 = 48s`, `21:00:07 = 1s`). Random split placed the `3571s` outlier in train: OLS/RF overfit the outlier, while RBF-SVR ($\epsilon=0.1$) predicted a constant $11.9\text{s}$ ($\text{MAE} \approx \epsilon$, $\text{MSE} \approx \epsilon^2$). | Evaluating specifically on regime-transition windows (`07:00–08:01` and `20:55–21:00`) exposes large boundary residuals ($\text{MAE} > 11.0\text{s}$), showing static feature mappings cannot anticipate phase changes. |
| **ANPR Volume Classification**<br>(`p_convent.xlsx`, $N=2,393$ events, 4 channels) | **Random Forest Classifier:** Accuracy $= 1.00$, Macro-$F_1 = 1.00$ ($n_{\text{test}} = 479$) | Pre-split `groupby(['hour', 'channel_id']).size()` merged 10 discrete hourly count buckets (`51` to `501`) back onto individual vehicle rows before random splitting, allowing tree ensembles to memorize a 10-row `(hour, channel_id)` lookup table. | **Corridor-Held-Out (`ZeroBridge_L1`):** $0.00$ exact-bucket accuracy (unseen channel distribution).<br>**Chronological Split (`18:00–19:00` held out):** Requires reformulating target from static count labels to causal rolling-window regression. |

## Running the Audit
```bash
pip install pandas numpy scikit-learn openpyxl
python srinagar_spatiotemporal_audit.py --signal_path "STAGE_TIMING Convent.xlsx" --volume_path "p convent.xlsx"