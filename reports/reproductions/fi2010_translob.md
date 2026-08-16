# FI-2010 TransLOB Reproduction Snapshot

## Protocol Provenance
- Execution commit: `0e2209cc2190fac8f3370ac8a88019131fe3dfba`
- Framework commit: `d3db43310cdd6b9a48a445a97bde9b424fe8d194`
- Protocol SHA-256: `ea83a3732b1fe43e6eabc04a596bf76b41de22f6c863488dcc6adc41d4032e50`
- Parameter count: 101895

## Coverage
- Planned: 250
- Completed: 250
- Missing: 0
- anchored_forward: 225
- first_seven_final_three: 25
- Folds: [1, 2, 3, 4, 5, 6, 7, 8, 9]
- Horizons: [10, 20, 30, 50, 100]
- Seeds: [1337, 2027, 31415, 424242, 8675309]

## CUDA
- CUDA runs: 250
- Nonzero GPU memory: 250
- Peak GPU memory range: 44131328 – 44371968 bytes

## Verification
- total_verified: 250
- metric_mismatches: 0
- prediction_mismatches: 0
- checkpoint_mismatches: 0
- provenance_mismatches: 0

## Aggregate Results
### anchored_forward | h10
- Runs: 45
- macro_f1: 0.366471 +/- 0.135103
- mcc: 0.133926 +/- 0.180258
- accuracy: 0.640811 +/- 0.083671
- balanced_accuracy: 0.398888 +/- 0.092915
- nll: 0.974596 +/- 0.369858
- brier: 0.516169 +/- 0.119293
- ece: 0.072302 +/- 0.065249
- Training time: 2240s

### anchored_forward | h100
- Runs: 45
- macro_f1: 0.479409 +/- 0.096203
- mcc: 0.204413 +/- 0.143587
- accuracy: 0.496110 +/- 0.076497
- balanced_accuracy: 0.478196 +/- 0.094618
- nll: 1.089206 +/- 0.202153
- brier: 0.622454 +/- 0.089655
- ece: 0.117164 +/- 0.062484
- Training time: 1956s

### anchored_forward | h20
- Runs: 45
- macro_f1: 0.346212 +/- 0.081658
- mcc: 0.089908 +/- 0.102348
- accuracy: 0.506313 +/- 0.087449
- balanced_accuracy: 0.374264 +/- 0.052046
- nll: 1.252245 +/- 0.378044
- brier: 0.656139 +/- 0.117946
- ece: 0.115720 +/- 0.093722
- Training time: 2546s

### anchored_forward | h30
- Runs: 45
- macro_f1: 0.410308 +/- 0.106583
- mcc: 0.150856 +/- 0.149473
- accuracy: 0.471740 +/- 0.111186
- balanced_accuracy: 0.421068 +/- 0.092804
- nll: 1.179247 +/- 0.272445
- brier: 0.655149 +/- 0.121857
- ece: 0.108991 +/- 0.076204
- Training time: 2385s

### anchored_forward | h50
- Runs: 45
- macro_f1: 0.438774 +/- 0.101140
- mcc: 0.168859 +/- 0.158004
- accuracy: 0.451717 +/- 0.109550
- balanced_accuracy: 0.441599 +/- 0.100159
- nll: 1.186053 +/- 0.290421
- brier: 0.661153 +/- 0.120340
- ece: 0.115212 +/- 0.080784
- Training time: 2379s

### first_seven_final_three | h10
- Runs: 5
- macro_f1: 0.407430 +/- 0.117534
- mcc: 0.183810 +/- 0.165854
- accuracy: 0.718647 +/- 0.024944
- balanced_accuracy: 0.418813 +/- 0.080250
- nll: 0.837704 +/- 0.182560
- brier: 0.432656 +/- 0.054548
- ece: 0.075256 +/- 0.035912
- Training time: 2878s

### first_seven_final_three | h100
- Runs: 5
- macro_f1: 0.559812 +/- 0.037452
- mcc: 0.337500 +/- 0.058617
- accuracy: 0.555665 +/- 0.039190
- balanced_accuracy: 0.555534 +/- 0.039892
- nll: 0.948822 +/- 0.049399
- brier: 0.552734 +/- 0.030589
- ece: 0.075452 +/- 0.018478
- Training time: 2311s

### first_seven_final_three | h20
- Runs: 5
- macro_f1: 0.349681 +/- 0.029385
- mcc: 0.098780 +/- 0.028854
- accuracy: 0.596669 +/- 0.014506
- balanced_accuracy: 0.369748 +/- 0.016700
- nll: 0.997586 +/- 0.043776
- brier: 0.567227 +/- 0.014204
- ece: 0.083865 +/- 0.014734
- Training time: 2504s

### first_seven_final_three | h30
- Runs: 5
- macro_f1: 0.462061 +/- 0.060513
- mcc: 0.221718 +/- 0.088948
- accuracy: 0.556181 +/- 0.051774
- balanced_accuracy: 0.461210 +/- 0.055505
- nll: 1.032671 +/- 0.113382
- brier: 0.576182 +/- 0.056680
- ece: 0.068234 +/- 0.024474
- Training time: 2938s

### first_seven_final_three | h50
- Runs: 5
- macro_f1: 0.513254 +/- 0.047252
- mcc: 0.295865 +/- 0.070841
- accuracy: 0.544322 +/- 0.046106
- balanced_accuracy: 0.514668 +/- 0.047300
- nll: 0.981250 +/- 0.058216
- brier: 0.562050 +/- 0.035702
- ece: 0.061681 +/- 0.015009
- Training time: 2627s

## Fold Summaries
- Fold 1: 25 runs, mean macro-F1 = 0.3300
- Fold 2: 25 runs, mean macro-F1 = 0.3353
- Fold 3: 25 runs, mean macro-F1 = 0.3390
- Fold 4: 25 runs, mean macro-F1 = 0.3384
- Fold 5: 25 runs, mean macro-F1 = 0.3642
- Fold 6: 25 runs, mean macro-F1 = 0.4054
- Fold 7: 25 runs, mean macro-F1 = 0.4436
- Fold 8: 25 runs, mean macro-F1 = 0.5359
- Fold 9: 25 runs, mean macro-F1 = 0.5823

## Seed Summaries
- Seed 1337: 50 runs, mean macro-F1 = 0.4001
- Seed 2027: 50 runs, mean macro-F1 = 0.4036
- Seed 31415: 50 runs, mean macro-F1 = 0.4081
- Seed 424242: 50 runs, mean macro-F1 = 0.4389
- Seed 8675309: 50 runs, mean macro-F1 = 0.4155

## Training Time
- Mean: 2336s, min: 100s, max: 9051s

## Collapse Audit
- Total runs: 250
- Single-class runs: 23

## Disclosures
- W_O: Paper equation includes W^O; official repository module omits it. DeepBook retains W^O per the paper equation. Classified AMBIGUOUS_SOURCE_CONFLICT.
- L2: Paper specifies L2 regularization on dense-64 classifier. Coefficient not recoverable from authoritative sources. Omission frozen as source-ambiguity adaptation.
- Labels: Official FI-2010 supplied labels only. No alternative labels used.
- Horizon 30: Horizon 30 is part of the controlled DeepBook comparison but has no direct paper match (source TransLOB horizon set: 10,20,50,100). Classified as unmatched; no post-hoc tolerance applied.
- TLOB: TLOB remains unexecuted. 250 cells planned for future execution.

## Hashes
- report_json_sha256: `34d9354088a91824b920031539740b1ba49577e6a8f6c4bbbf201a90ddf51c0a`
- report_md_sha256: `859e8309df84bf507bd0b4bf8a28719afc2d60f61e16eeacba959ce614bdfe5f`
- run_index_sha256: `eca7f01ca9896e29d66717991609f0c915acfb9f43fce6be79a93707d27d13da`

## Limitations
- No significance testing or hypothesis claims.
- TransLOB results are descriptive; no numeric performance threshold was preregistered.
- L2 regularization on dense-64 classifier is omitted (coefficient not recoverable).
- Horizon 30 has no direct literature comparison.
- TLOB has not been executed.
- GPU memory and timing measurements are hardware-dependent.
