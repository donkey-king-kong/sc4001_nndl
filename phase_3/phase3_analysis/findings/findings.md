## Baseline
- Lambda 0 baseline: CIDEr 104.73 ± 0.39, CHAIRi 6.80 ± 0.32%, CHAIRs 10.22 ± 0.49% over 3 seeds.

## Conditions vs baseline
- structured lam 3: CIDEr -0.21. CHAIRi +0.05 points, 95% CI [-0.32, +0.41], p = 0.789, not significant. CHAIRs +0.01 points, 95% CI [-0.54, +0.57], p = 0.936, not significant.
- random lam 3: CIDEr +0.36. CHAIRi -0.19 points, 95% CI [-0.52, +0.14], p = 0.296, not significant. CHAIRs -0.21 points, 95% CI [-0.71, +0.31], p = 0.432, not significant.
- UL structured lam 3: CIDEr +0.29. CHAIRi +0.11 points, 95% CI [-0.24, +0.46], p = 0.540, not significant. CHAIRs +0.20 points, 95% CI [-0.34, +0.73], p = 0.484, not significant.

## Lambda selection
- Lambda 3 was chosen on val; test only reports it.

## Ablation
- Random vs co-occurrence negatives at lambda 3: CHAIRi -0.23 points, 95% CI [-0.60, +0.12], not significant.

## Loss type
- Unlikelihood vs margin loss at lambda 3: CHAIRi +0.07 points, 95% CI [-0.31, +0.44], not significant.

## Breakdowns (baseline)
- By object frequency: rarest quartile 9.78%, most common 6.66%, highest in Q1 179-2123.
- By co-occurrence strength, person excluded: 4.26% in the weakest bin vs 14.83% at the peak (0.2-0.4); the rate rises across the lower bins.
