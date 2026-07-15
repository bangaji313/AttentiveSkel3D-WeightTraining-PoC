# Five Scenarios Model Audit Report

Generated: 2026-07-14T19:42:47.253870

## Summary

- Total scenarios: 5
- Successfully loaded: 5
- Bugs detected: 0

## Checkpoints

```
                     Scenario                                                                                                                        Absolute_Path  File_Exists  File_Size_MB           SHA256  State_Dict_Keys  Total_Parameters Error
                   Full Model G:\data-aji\KULIAH\Semester 8\IFB500-TUGAS_AKHIR-AA\AttentiveSkel3D-WeightTraining-PoC\models\saved_models\AttentiveSkel3D_Final.pth         True      1.292329 5440164315d6d275               29            110823  None
              Baseline 3D-CNN  G:\data-aji\KULIAH\Semester 8\IFB500-TUGAS_AKHIR-AA\AttentiveSkel3D-WeightTraining-PoC\models\saved_models\baseline_3dcnn_model.pth         True      1.186819 3224393f85793286               23            102342  None
          Ablasi A - No Prior     G:\data-aji\KULIAH\Semester 8\IFB500-TUGAS_AKHIR-AA\AttentiveSkel3D-WeightTraining-PoC\models\saved_models\ablasi_a_no_prior.pth         True      1.290421 d13e665670e3ad61               28            110790  None
Ablasi B - No Learned Spatial   G:\data-aji\KULIAH\Semester 8\IFB500-TUGAS_AKHIR-AA\AttentiveSkel3D-WeightTraining-PoC\models\saved_models\ablasi_b_no_learned.pth         True      1.191672 805634011271f1bd               25            102471  None
       Ablasi C - No Temporal  G:\data-aji\KULIAH\Semester 8\IFB500-TUGAS_AKHIR-AA\AttentiveSkel3D-WeightTraining-PoC\models\saved_models\ablasi_c_no_temporal.pth         True      1.288319 a739bd17cf24d8bc               27            110694  None
```

## Predictions

```
                     Scenario  Logits_Class0  Logits_Class1  Softmax_Class0  Softmax_Class1  Pred_Class  Confidence                                                                                                        Top5_Joints
                   Full Model      -0.744340       0.479351        0.227288    7.727123e-01           1    0.772712  [(22, 1.0), (31, 0.9951843619346619), (16, 0.9805877804756165), (29, 0.957427442073822), (14, 0.918994128704071)]
              Baseline 3D-CNN       8.563449      -7.918457        1.000000    6.950223e-08           0    1.000000                                                               [(32, 0.5), (15, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]
          Ablasi A - No Prior       1.161450      -0.996617        0.896420    1.035798e-01           0    0.896420                                                               [(32, 0.5), (15, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]
Ablasi B - No Learned Spatial      -0.496371       0.445055        0.280613    7.193875e-01           1    0.719387 [(20, 1.0), (13, 0.977603554725647), (22, 0.9590165615081787), (21, 0.9394317865371704), (16, 0.9253363609313965)]
       Ablasi C - No Temporal       2.373545      -3.876117        0.998073    1.927384e-03           0    0.998073 [(13, 1.0), (14, 0.9823464751243591), (22, 0.9214217662811279), (15, 0.8540952205657959), (12, 0.791946530342102)]
```

## Similarity Matrix

```
                   Scenario_1                    Scenario_2  L1_Distance_Logits  L2_Distance_Logits  Cosine_Similarity_Softmax  Pearson_Corr_Attention  Logits_Exact_Match  Softmax_Exact_Match  Attention_Exact_Match
                   Full Model               Baseline 3D-CNN            8.852798            8.864483                   0.282188                     NaN               False                False                  False
                   Full Model           Ablasi A - No Prior            1.690879            1.704482                   0.390443                     NaN               False                False                  False
                   Full Model Ablasi B - No Learned Spatial            0.141133            0.177010                   0.996318                0.677413               False                False                  False
                   Full Model        Ablasi C - No Temporal            3.736676            3.787566                   0.284041                0.710203               False                False                  False
              Baseline 3D-CNN           Ablasi A - No Prior            7.161920            7.165942                   0.993390                     NaN               False                False                   True
              Baseline 3D-CNN Ablasi B - No Learned Spatial            8.711666            8.718619                   0.363403                     NaN               False                False                  False
              Baseline 3D-CNN        Ablasi C - No Temporal            5.116122            5.227592                   0.999998                     NaN               False                False                  False
          Ablasi A - No Prior Ablasi B - No Learned Spatial            1.549746            1.553510                   0.467938                     NaN               False                False                  False
          Ablasi A - No Prior        Ablasi C - No Temporal            2.045798            2.209151                   0.993610                     NaN               False                False                  False
Ablasi B - No Learned Spatial        Ablasi C - No Temporal            3.595544            3.668034                   0.365202                0.619367               False                False                  False
```

## No Bugs Detected

All scenarios loaded successfully with distinct configurations.
