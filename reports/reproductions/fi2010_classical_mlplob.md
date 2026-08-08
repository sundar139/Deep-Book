# FI-2010 Classical and MLP-LOB Reproduction Snapshot

## Provenance

- Execution commit: `dd0446e743b35c2dbe7cae3c17e46562850b9772`
- Protocol commit: `f254599eb215558588aed0647a3e3317dab36da3`
- Protocol SHA-256: `c3d5eac2dc722c90cb9b704496ee8b181919c9dbc9f1a76306436a6aa25aac37`
- Archive SHA-256: `cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664`

### Configuration Hashes
- `classical`: `6aca58ba66b72b9c3c465bbc90a28b5694d10d8436d7564eef693cb7972402d0`
- `mlplob`: `7754064e6ff686d9210cc272b1415e1b11996e19f912ccf6b92660f5b78b2dfa`

## Coverage

- Planned total: 900
- Selected confirmatory total: 650

### By Model

| Model | Completed |
|---|---|
| majority | 50 |
| causal_persistence | 50 |
| logistic_current_event | 50 |
| random_forest | 250 |
| mlplob | 250 |
| deeplob | 0 |

### By Setup
- anchored_forward: 585
- first_seven_final_three: 65

### Seed Completeness
- 1337: 250/300 (incomplete)
- 2027: 100/150 (incomplete)
- 31415: 100/150 (incomplete)
- 424242: 100/150 (incomplete)
- 8675309: 100/150 (incomplete)

## Run Status

- Missing: 250
- Failed: 0
- Interrupted: 0
- Running: 0
- Duplicate: 0
- Ineligible: 0
- Orphan predictions: 0
- Orphan checkpoints: 0

## Verification

- Total verified: 650
- Metric mismatches: 0
- Prediction mismatches: 0
- Checkpoint mismatches: 0
- Provenance mismatches: 0

## Majority-Class Collapse

All 50 majority-class runs intentionally predict exactly one class: the modal class in the corresponding training partition. 33 runs predict only stationary. 17 runs predict only up. 0 runs predict only down. This single-class behavior is definitional for the majority-class baseline. It is not a training failure, implementation bug, or unexplained model collapse.

- Total majority runs: 50
- Single-class runs: 50
- Stationary-only: 33
- Up-only: 17
- Down-only: 0
- All runs single-class: True

## Causal-Persistence Sample Counts

persisted samples = declared test observations - segments x horizon. The first 'horizon' observations of each independent segment have no causal predecessor at the required lag and are removed. Setup 2 applies this separately to days 8, 9, and 10.

- Total causal-persistence runs: 50
- Runs with verified sample-count invariant: 50
- Setup 1 segments: 1
- Setup 2 segments: 3

## Reconciliation Events

A running manifest without a valid recoverable last-state checkpoint was quarantined and the logical cell was re-executed cleanly. The quarantined evidence was retained and inventoried.

- Total events: 6
- Reconciliation digest: `62e45e3a60f831bee4ee6a2d426350f3de7b09901d91f25803144bfd4dd386cb`

| Timestamp | Run ID | Reason | Disposition |
|---|---|---|---|
| 2026-08-03T010159Z | random_forest-anchored-forward-f1-h20-s31415 | running manifest has no valid recoverable last checkpoint | quarantined and re-executed cleanly |
| 2026-08-03T011210Z | random_forest-anchored-forward-f1-h50-s8675309 | running manifest has no valid recoverable last checkpoint | quarantined and re-executed cleanly |
| 2026-08-03T012230Z | random_forest-anchored-forward-f2-h10-s8675309 | running manifest has no valid recoverable last checkpoint | quarantined and re-executed cleanly |
| 2026-08-03T013254Z | random_forest-anchored-forward-f2-h100-s8675309 | running manifest has no valid recoverable last checkpoint | quarantined and re-executed cleanly |
| 2026-08-03T045312Z | mlplob-anchored-forward-f1-h10-s1337 | running manifest has no valid recoverable last checkpoint | quarantined and re-executed cleanly |
| 2026-08-03T045445Z | random_forest-anchored-forward-f5-h100-s2027 | running manifest has no valid recoverable last checkpoint | quarantined and re-executed cleanly |

## Aggregate Results

### By Model / Setup / Horizon

#### causal_persistence | anchored_forward | h10

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.386086 +/- 0.010197
- mcc: 0.095875 +/- 0.021815
- accuracy: 0.523565 +/- 0.053492
- balanced_accuracy: 0.386088 +/- 0.010198
- nll: 16.455493 +/- 1.847543
- brier: 0.952871 +/- 0.106984
- ece: 0.476435 +/- 0.053492

#### causal_persistence | anchored_forward | h100

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.360659 +/- 0.016270
- mcc: 0.033829 +/- 0.028737
- accuracy: 0.378286 +/- 0.014046
- balanced_accuracy: 0.360648 +/- 0.016263
- nll: 21.473238 +/- 0.485134
- brier: 1.243428 +/- 0.028092
- ece: 0.621714 +/- 0.014046

#### causal_persistence | anchored_forward | h20

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.380768 +/- 0.009178
- mcc: 0.083239 +/- 0.021086
- accuracy: 0.447921 +/- 0.049016
- balanced_accuracy: 0.380769 +/- 0.009179
- nll: 19.068143 +/- 1.692951
- brier: 1.104159 +/- 0.098032
- ece: 0.552079 +/- 0.049016

#### causal_persistence | anchored_forward | h30

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.376906 +/- 0.012434
- mcc: 0.074414 +/- 0.025752
- accuracy: 0.412008 +/- 0.045471
- balanced_accuracy: 0.376907 +/- 0.012434
- nll: 20.308509 +/- 1.570504
- brier: 1.175983 +/- 0.090941
- ece: 0.587992 +/- 0.045471

#### causal_persistence | anchored_forward | h50

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.366051 +/- 0.018989
- mcc: 0.052225 +/- 0.033992
- accuracy: 0.377986 +/- 0.035308
- balanced_accuracy: 0.366048 +/- 0.018991
- nll: 21.483603 +/- 1.219487
- brier: 1.244028 +/- 0.070616
- ece: 0.622014 +/- 0.035308

#### causal_persistence | first_seven_final_three | h10

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.396491
- mcc: 0.121616
- accuracy: 0.598093
- balanced_accuracy: 0.396496
- nll: 13.881392
- brier: 0.803815
- ece: 0.401907

#### causal_persistence | first_seven_final_three | h100

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.384198
- mcc: 0.077405
- accuracy: 0.385564
- balanced_accuracy: 0.384173
- nll: 21.221881
- brier: 1.228873
- ece: 0.614436

#### causal_persistence | first_seven_final_three | h20

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.390578
- mcc: 0.109775
- accuracy: 0.516789
- balanced_accuracy: 0.390583
- nll: 16.689521
- brier: 0.966422
- ece: 0.483211

#### causal_persistence | first_seven_final_three | h30

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.394139
- mcc: 0.110646
- accuracy: 0.476498
- balanced_accuracy: 0.394144
- nll: 18.081129
- brier: 1.047005
- ece: 0.523502

#### causal_persistence | first_seven_final_three | h50

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.386900
- mcc: 0.093658
- accuracy: 0.422556
- balanced_accuracy: 0.386899
- nll: 19.944195
- brier: 1.154887
- ece: 0.577444

#### logistic_current_event | anchored_forward | h10

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.318158 +/- 0.013893
- mcc: 0.116224 +/- 0.016917
- accuracy: 0.640763 +/- 0.049758
- balanced_accuracy: 0.359384 +/- 0.007168
- nll: 0.865874 +/- 0.063090
- brier: 0.502069 +/- 0.044757
- ece: 0.033680 +/- 0.028584

#### logistic_current_event | anchored_forward | h100

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.394696 +/- 0.031685
- mcc: 0.158060 +/- 0.033528
- accuracy: 0.467133 +/- 0.047196
- balanced_accuracy: 0.420549 +/- 0.016619
- nll: 1.035325 +/- 0.050577
- brier: 0.624219 +/- 0.030840
- ece: 0.037003 +/- 0.025728

#### logistic_current_event | anchored_forward | h20

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.405447 +/- 0.014762
- mcc: 0.204646 +/- 0.017486
- accuracy: 0.568948 +/- 0.054010
- balanced_accuracy: 0.420106 +/- 0.008668
- nll: 0.959986 +/- 0.051978
- brier: 0.567427 +/- 0.037256
- ece: 0.041225 +/- 0.036535

#### logistic_current_event | anchored_forward | h30

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.411342 +/- 0.019665
- mcc: 0.178182 +/- 0.020659
- accuracy: 0.504508 +/- 0.052632
- balanced_accuracy: 0.425789 +/- 0.011913
- nll: 1.014707 +/- 0.038196
- brier: 0.607943 +/- 0.027103
- ece: 0.037418 +/- 0.037113

#### logistic_current_event | anchored_forward | h50

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.438365 +/- 0.015787
- mcc: 0.168312 +/- 0.017313
- accuracy: 0.451087 +/- 0.017690
- balanced_accuracy: 0.443303 +/- 0.013030
- nll: 1.052377 +/- 0.013469
- brier: 0.635055 +/- 0.008679
- ece: 0.019374 +/- 0.010472

#### logistic_current_event | first_seven_final_three | h10

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.328239
- mcc: 0.121351
- accuracy: 0.709600
- balanced_accuracy: 0.357267
- nll: 0.782648
- brier: 0.442956
- ece: 0.080210

#### logistic_current_event | first_seven_final_three | h100

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.354386
- mcc: 0.121869
- accuracy: 0.401613
- balanced_accuracy: 0.408111
- nll: 1.114097
- brier: 0.671172
- ece: 0.075084

#### logistic_current_event | first_seven_final_three | h20

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.421175
- mcc: 0.227339
- accuracy: 0.643828
- balanced_accuracy: 0.422127
- nll: 0.894255
- brier: 0.520216
- ece: 0.100298

#### logistic_current_event | first_seven_final_three | h30

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.433846
- mcc: 0.204309
- accuracy: 0.575777
- balanced_accuracy: 0.434261
- nll: 0.969870
- brier: 0.576298
- ece: 0.096115

#### logistic_current_event | first_seven_final_three | h50

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.433220
- mcc: 0.160664
- accuracy: 0.445342
- balanced_accuracy: 0.440869
- nll: 1.053544
- brier: 0.636292
- ece: 0.007307

#### majority | anchored_forward | h10

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.258526 +/- 0.012739
- mcc: 0.000000 +/- 0.000000
- accuracy: 0.635049 +/- 0.052067
- balanced_accuracy: 0.333333 +/- 0.000000
- nll: 0.909906 +/- 0.060620
- brier: 0.530627 +/- 0.044081
- ece: 0.043035 +/- 0.034497

#### majority | anchored_forward | h100

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.185665 +/- 0.011779
- mcc: 0.000000 +/- 0.000000
- accuracy: 0.386817 +/- 0.033478
- balanced_accuracy: 0.333333 +/- 0.000000
- nll: 1.078657 +/- 0.050556
- brier: 0.653845 +/- 0.029227
- ece: 0.027487 +/- 0.023615

#### majority | anchored_forward | h20

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.230601 +/- 0.017891
- mcc: 0.000000 +/- 0.000000
- accuracy: 0.531470 +/- 0.064584
- balanced_accuracy: 0.333333 +/- 0.000000
- nll: 1.016912 +/- 0.046464
- brier: 0.608512 +/- 0.033559
- ece: 0.052397 +/- 0.043604

#### majority | anchored_forward | h30

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.209074 +/- 0.021982
- mcc: 0.000000 +/- 0.000000
- accuracy: 0.460378 +/- 0.072456
- balanced_accuracy: 0.333333 +/- 0.000000
- nll: 1.065231 +/- 0.030701
- brier: 0.643321 +/- 0.021684
- ece: 0.058265 +/- 0.049134

#### majority | anchored_forward | h50

- Runs: 9
- Seeds: [1337]
- macro_f1: 0.172074 +/- 0.020476
- mcc: 0.000000 +/- 0.000000
- accuracy: 0.350279 +/- 0.057278
- balanced_accuracy: 0.333333 +/- 0.000000
- nll: 1.098229 +/- 0.006684
- brier: 0.666378 +/- 0.004455
- ece: 0.041172 +/- 0.035256

#### majority | first_seven_final_three | h10

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.276036
- mcc: 0.000000
- accuracy: 0.706642
- balanced_accuracy: 0.333333
- nll: 0.830802
- brier: 0.473026
- ece: 0.101315

#### majority | first_seven_final_three | h100

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.170524
- mcc: 0.000000
- accuracy: 0.343700
- balanced_accuracy: 0.333333
- nll: 1.160242
- brier: 0.700163
- ece: 0.062452

#### majority | first_seven_final_three | h20

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.255279
- mcc: 0.000000
- accuracy: 0.620531
- balanced_accuracy: 0.333333
- nll: 0.959075
- brier: 0.566918
- ece: 0.126470

#### majority | first_seven_final_three | h30

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.239609
- mcc: 0.000000
- accuracy: 0.561069
- balanced_accuracy: 0.333333
- nll: 1.031118
- brier: 0.619505
- ece: 0.143040

#### majority | first_seven_final_three | h50

- Runs: 1
- Seeds: [1337]
- macro_f1: 0.144027
- mcc: 0.000000
- accuracy: 0.275577
- balanced_accuracy: 0.333333
- nll: 1.109697
- brier: 0.673957
- ece: 0.068309

#### mlplob | anchored_forward | h10

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.407776 +/- 0.084297
- mcc: 0.146848 +/- 0.082942
- accuracy: 0.607829 +/- 0.054980
- balanced_accuracy: 0.419731 +/- 0.050844
- nll: 1.389651 +/- 0.422681
- brier: 0.588405 +/- 0.073900
- ece: 0.134429 +/- 0.062426

#### mlplob | anchored_forward | h100

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.473358 +/- 0.041323
- mcc: 0.242119 +/- 0.051697
- accuracy: 0.516415 +/- 0.030075
- balanced_accuracy: 0.479170 +/- 0.039928
- nll: 1.792789 +/- 0.546070
- brier: 0.689806 +/- 0.056327
- ece: 0.208426 +/- 0.058379

#### mlplob | anchored_forward | h20

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.395709 +/- 0.043362
- mcc: 0.106054 +/- 0.046128
- accuracy: 0.476346 +/- 0.060306
- balanced_accuracy: 0.401989 +/- 0.025009
- nll: 1.625923 +/- 0.465360
- brier: 0.727403 +/- 0.079618
- ece: 0.192783 +/- 0.066048

#### mlplob | anchored_forward | h30

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.428930 +/- 0.030569
- mcc: 0.146545 +/- 0.049322
- accuracy: 0.456266 +/- 0.056499
- balanced_accuracy: 0.431525 +/- 0.029097
- nll: 1.722391 +/- 0.466317
- brier: 0.749190 +/- 0.081858
- ece: 0.212209 +/- 0.064444

#### mlplob | anchored_forward | h50

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.450050 +/- 0.036360
- mcc: 0.183478 +/- 0.048392
- accuracy: 0.460014 +/- 0.036982
- balanced_accuracy: 0.455392 +/- 0.034180
- nll: 1.650494 +/- 0.491043
- brier: 0.730720 +/- 0.063659
- ece: 0.198124 +/- 0.064572

#### mlplob | first_seven_final_three | h10

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.395705 +/- 0.085930
- mcc: 0.143752 +/- 0.066491
- accuracy: 0.674620 +/- 0.028285
- balanced_accuracy: 0.411442 +/- 0.058917
- nll: 1.191589 +/- 0.346630
- brier: 0.513451 +/- 0.052084
- ece: 0.113176 +/- 0.041855

#### mlplob | first_seven_final_three | h100

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.486943 +/- 0.005297
- mcc: 0.258484 +/- 0.006011
- accuracy: 0.498222 +/- 0.003401
- balanced_accuracy: 0.502373 +/- 0.003546
- nll: 2.273386 +/- 0.286891
- brier: 0.750663 +/- 0.014757
- ece: 0.265555 +/- 0.014818

#### mlplob | first_seven_final_three | h20

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.416905 +/- 0.007839
- mcc: 0.127704 +/- 0.016144
- accuracy: 0.544619 +/- 0.019296
- balanced_accuracy: 0.414116 +/- 0.006642
- nll: 1.550820 +/- 0.342679
- brier: 0.685531 +/- 0.052084
- ece: 0.198518 +/- 0.044467

#### mlplob | first_seven_final_three | h30

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.443978 +/- 0.011236
- mcc: 0.170800 +/- 0.013699
- accuracy: 0.498678 +/- 0.033724
- balanced_accuracy: 0.450816 +/- 0.004588
- nll: 1.479549 +/- 0.421443
- brier: 0.693912 +/- 0.065638
- ece: 0.171273 +/- 0.051762

#### mlplob | first_seven_final_three | h50

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.464687 +/- 0.006241
- mcc: 0.205203 +/- 0.004962
- accuracy: 0.470406 +/- 0.009231
- balanced_accuracy: 0.478142 +/- 0.003395
- nll: 1.879286 +/- 0.517506
- brier: 0.746695 +/- 0.059795
- ece: 0.211331 +/- 0.064320

#### random_forest | anchored_forward | h10

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.456024 +/- 0.016948
- mcc: 0.244889 +/- 0.019159
- accuracy: 0.635698 +/- 0.029870
- balanced_accuracy: 0.452442 +/- 0.015578
- nll: 0.858474 +/- 0.038658
- brier: 0.500525 +/- 0.027316
- ece: 0.079409 +/- 0.035390

#### random_forest | anchored_forward | h100

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.350462 +/- 0.032618
- mcc: 0.153950 +/- 0.043509
- accuracy: 0.466936 +/- 0.057721
- balanced_accuracy: 0.406582 +/- 0.015875
- nll: 1.082257 +/- 0.124256
- brier: 0.641581 +/- 0.062376
- ece: 0.069247 +/- 0.050486

#### random_forest | anchored_forward | h20

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.606655 +/- 0.006758
- mcc: 0.461612 +/- 0.021747
- accuracy: 0.687352 +/- 0.025886
- balanced_accuracy: 0.590480 +/- 0.006178
- nll: 0.811358 +/- 0.024038
- brier: 0.471009 +/- 0.017802
- ece: 0.105746 +/- 0.042571

#### random_forest | anchored_forward | h30

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.550448 +/- 0.017095
- mcc: 0.352413 +/- 0.036846
- accuracy: 0.588370 +/- 0.018760
- balanced_accuracy: 0.552372 +/- 0.016469
- nll: 0.918666 +/- 0.018861
- brier: 0.547950 +/- 0.013013
- ece: 0.085786 +/- 0.021995

#### random_forest | anchored_forward | h50

- Runs: 45
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.452787 +/- 0.050536
- mcc: 0.220326 +/- 0.044299
- accuracy: 0.463852 +/- 0.050720
- balanced_accuracy: 0.477421 +/- 0.028242
- nll: 1.020551 +/- 0.045511
- brier: 0.616369 +/- 0.027817
- ece: 0.047210 +/- 0.027500

#### random_forest | first_seven_final_three | h10

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.393002 +/- 0.010122
- mcc: 0.154041 +/- 0.009306
- accuracy: 0.466741 +/- 0.019688
- balanced_accuracy: 0.444523 +/- 0.005096
- nll: 1.008657 +/- 0.005128
- brier: 0.601500 +/- 0.003446
- ece: 0.061501 +/- 0.015239

#### random_forest | first_seven_final_three | h100

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.307172 +/- 0.000988
- mcc: 0.103879 +/- 0.001758
- accuracy: 0.389359 +/- 0.000866
- balanced_accuracy: 0.390620 +/- 0.000836
- nll: 1.320352 +/- 0.003274
- brier: 0.745755 +/- 0.000949
- ece: 0.146043 +/- 0.002156

#### random_forest | first_seven_final_three | h20

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.428764 +/- 0.001821
- mcc: 0.214285 +/- 0.001938
- accuracy: 0.443640 +/- 0.001045
- balanced_accuracy: 0.493647 +/- 0.001986
- nll: 1.097256 +/- 0.019369
- brier: 0.648619 +/- 0.009205
- ece: 0.146431 +/- 0.002826

#### random_forest | first_seven_final_three | h30

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.398247 +/- 0.002314
- mcc: 0.180113 +/- 0.002788
- accuracy: 0.400023 +/- 0.002079
- balanced_accuracy: 0.458264 +/- 0.002719
- nll: 1.149024 +/- 0.013159
- brier: 0.685622 +/- 0.006594
- ece: 0.103783 +/- 0.002607

#### random_forest | first_seven_final_three | h50

- Runs: 5
- Seeds: [1337, 2027, 31415, 424242, 8675309]
- macro_f1: 0.306249 +/- 0.000870
- mcc: 0.105920 +/- 0.001836
- accuracy: 0.334020 +/- 0.000864
- balanced_accuracy: 0.403764 +/- 0.001078
- nll: 1.281980 +/- 0.013651
- brier: 0.748281 +/- 0.003953
- ece: 0.152292 +/- 0.002216

## Environment

- gpu: NVIDIA GeForce RTX 4070 Laptop GPU
- mlplob_cuda_runs: 250
- mlplob_inference_ms_per_sample_mean: 0.018380736670143973
- mlplob_nonzero_gpu_memory_runs: 250
- mlplob_parameter_count: 520579
- mlplob_peak_gpu_memory_bytes: 29540352
- mlplob_training_seconds_max: 1196.1785970999626
- mlplob_training_seconds_mean: 272.83832446999827
- mlplob_training_seconds_min: 15.90248689998407
- numpy: 2.4.6
- platform: Windows-10-10.0.26200-SP0
- python: 3.11.15
- sklearn: unknown
- torch: 2.13.0+cu132
- torch_cuda: 13.2

## Reconciliation Digest

- Reconciliation digest: `62e45e3a60f831bee4ee6a2d426350f3de7b09901d91f25803144bfd4dd386cb`

## Disclosures

- Pushed-commit disclosure: Commits de61432 and dd0446e were pushed to origin/main during earlier work despite no-push instructions. Public history was not rewritten. This result-packaging commit remains local and is not pushed.
- Result commit is local only: True
- DeepLOB pending: DeepLOB confirmatory runs: 0. DeepLOB planned and pending: 250. No TransLOB, TLOB, Hawkes, modern-data, execution simulation, reinforcement learning, or paid-data work has been performed.

## Limitations

- No significance testing or hypothesis claims.
- Majority baseline is definitionally single-class.
- MLP-LOB is a simple architecture; not DeepLOB.
- Setup 1 and Setup 2 results are reported separately.
- FI-2010 data-audit reports are not included in this snapshot; they are generated by a separate audit command.
- The scikit-learn version is reported as 'unknown' because run manifests do not record it; the random forest results are therefore not pinned to a specific scikit-learn version in the provenance record.
