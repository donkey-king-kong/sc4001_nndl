## Baseline
- Lambda 0 baseline: CIDEr 104.28 ± 0.15, CHAIRi 7.52 ± 0.33%, CHAIRs 11.16 ± 0.55% over 3 seeds.
- Phase 1 baseline: CIDEr 104.43 ± 0.58, CHAIRi 7.21 ± 0.19%. The lambda 0 run reproduces it (CIDEr -0.15, CHAIRi +0.31 points, compared with the larger seed std of the two).

## Conditions vs baseline
- structured lam 0.1: CIDEr -0.54. CHAIRi -0.06 points, 95% CI [-0.41, +0.28], p = 0.721, not significant. CHAIRs -0.05 points, 95% CI [-0.59, +0.47], p = 0.844, not significant.
- structured lam 0.3: CIDEr -0.17. CHAIRi -0.21 points, 95% CI [-0.55, +0.13], p = 0.230, not significant. CHAIRs -0.17 points, 95% CI [-0.71, +0.32], p = 0.506, not significant.
- structured lam 1: CIDEr -0.56. CHAIRi -0.27 points, 95% CI [-0.62, +0.06], p = 0.106, not significant. CHAIRs -0.30 points, 95% CI [-0.82, +0.21], p = 0.260, not significant.
- structured lam 3: CIDEr -0.06. CHAIRi -0.31 points, 95% CI [-0.65, +0.03], p = 0.067, not significant. CHAIRs -0.36 points, 95% CI [-0.87, +0.13], p = 0.159, not significant.
- random lam 3: CIDEr -0.51. CHAIRi -0.14 points, 95% CI [-0.51, +0.22], p = 0.433, not significant. CHAIRs -0.03 points, 95% CI [-0.63, +0.49], p = 0.885, not significant.
- UL structured lam 3: CIDEr -0.01. CHAIRi -0.43 points, 95% CI [-0.76, -0.08], p = 0.018, significant. CHAIRs -0.48 points, 95% CI [-1.01, +0.06], p = 0.077, not significant.

## Lambda selection
- Best lambda by the rule: 3 (CHAIRi -0.31 points vs baseline on matched seeds).
- Its CHAIRi change is not significant, so the choice is within noise.

## Ablation
- Random vs co-occurrence negatives at lambda 3: CHAIRi +0.17 points, 95% CI [-0.21, +0.52], not significant.

## Loss type
- Unlikelihood vs margin loss at lambda 3: CHAIRi -0.12 points, 95% CI [-0.45, +0.23], not significant.

## Swap accuracy
- At lambda 0 the model already prefers the true caption in 98.94% of co-occurrence pairs, with a mean NLL gap of 1.197.
- The mean gap is well above the 0.2 margin, so the hinge loss is zero for most pairs and the auxiliary loss gives little gradient.
- At lambda 3 accuracy is 99.09% and the gap 1.228.

## Breakdowns (baseline)
- By object frequency: rarest quartile 10.02%, most common 7.88%, highest in Q1 179-2123.
- By co-occurrence strength, person excluded: 5.09% in the weakest bin vs 15.53% at the peak (0.2-0.4); the rate rises across the lower bins.
