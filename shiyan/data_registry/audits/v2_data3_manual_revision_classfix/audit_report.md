# 数据审计报告：v2_data3_manual_revision_classfix

> 历史快照：本报告和指纹对应 EXP005 训练前的数据状态。当前磁盘标签已有 8 个文件的目标数与该 manifest 不一致，因此现有权重可据此追溯，但后续重训前必须重新生成 audit、manifest 和 fingerprint。

生成时间：2026-09-02T15:42:53.112641+00:00

数据根目录：`shiyan/data3`

数据指纹：`9e330f03b56c3728ed1c485404d92306ed94edb22c7bcd8a0f09ac0a91438fa8`

## 结论摘要

- 图片数：4481
- 标注文件数：4481
- 忽略的辅助标签文件数：1
- 标注目标数：21008
- 类别数：25
- 精确重复图片记录数：0
- 推断场景组数：4481
- 跨 split 场景组数：0
- 问题记录数：47，其中 error 0，warn 47

## split 摘要

| split | images | label_files | objects | ship_objects | aircraft_objects | vehicle_objects |
| --- | --- | --- | --- | --- | --- | --- |
| train | 4481 | 4481 | 21008 | 2760 | 17849 | 399 |

## 类别分布

| class_id | class_name | group | train_objects | val_objects | total_objects | val_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | HM | ship | 17 | 0 | 17 | 0.0 |
| 1 | LQS | ship | 29 | 0 | 29 | 0.0 |
| 2 | QHS | ship | 637 | 0 | 637 | 0.0 |
| 3 | MS | ship | 2077 | 0 | 2077 | 0.0 |
| 4 | A1_SU-35 | aircraft | 1317 | 0 | 1317 | 0.0 |
| 5 | A2_C-130 | aircraft | 1297 | 0 | 1297 | 0.0 |
| 6 | A3_C-17 | aircraft | 998 | 0 | 998 | 0.0 |
| 7 | A4_C-5 | aircraft | 500 | 0 | 500 | 0.0 |
| 8 | A5_F-16 | aircraft | 1017 | 0 | 1017 | 0.0 |
| 9 | A6_TU-160 | aircraft | 361 | 0 | 361 | 0.0 |
| 10 | A7_E-3 | aircraft | 547 | 0 | 547 | 0.0 |
| 11 | A8_B-52 | aircraft | 750 | 0 | 750 | 0.0 |
| 12 | A9_P-3C | aircraft | 895 | 0 | 895 | 0.0 |
| 13 | A10_B-1B | aircraft | 762 | 0 | 762 | 0.0 |
| 14 | A11_E-8 | aircraft | 432 | 0 | 432 | 0.0 |
| 15 | A12_TU-22 | aircraft | 583 | 0 | 583 | 0.0 |
| 16 | A13_F-15 | aircraft | 1265 | 0 | 1265 | 0.0 |
| 17 | A14_KC-135 | aircraft | 1424 | 0 | 1424 | 0.0 |
| 18 | A15_F-22 | aircraft | 493 | 0 | 493 | 0.0 |
| 19 | A16_FA-18 | aircraft | 2147 | 0 | 2147 | 0.0 |
| 20 | A17_TU-95 | aircraft | 1114 | 0 | 1114 | 0.0 |
| 21 | A18_KC-10 | aircraft | 262 | 0 | 262 | 0.0 |
| 22 | A19_SU-34 | aircraft | 933 | 0 | 933 | 0.0 |
| 23 | A20_SU-24 | aircraft | 752 | 0 | 752 | 0.0 |
| 24 | FSC | vehicle | 399 | 0 | 399 | 0.0 |

## 数量最少的类别

| class_id | class_name | group | total_objects | train_objects | val_objects |
| --- | --- | --- | --- | --- | --- |
| 0 | HM | ship | 17 | 17 | 0 |
| 1 | LQS | ship | 29 | 29 | 0 |
| 21 | A18_KC-10 | aircraft | 262 | 262 | 0 |
| 9 | A6_TU-160 | aircraft | 361 | 361 | 0 |
| 24 | FSC | vehicle | 399 | 399 | 0 |
| 14 | A11_E-8 | aircraft | 432 | 432 | 0 |
| 18 | A15_F-22 | aircraft | 493 | 493 | 0 |
| 7 | A4_C-5 | aircraft | 500 | 500 | 0 |
| 10 | A7_E-3 | aircraft | 547 | 547 | 0 |
| 15 | A12_TU-22 | aircraft | 583 | 583 | 0 |

## 数量最多的类别

| class_id | class_name | group | total_objects | train_objects | val_objects |
| --- | --- | --- | --- | --- | --- |
| 19 | A16_FA-18 | aircraft | 2147 | 2147 | 0 |
| 3 | MS | ship | 2077 | 2077 | 0 |
| 17 | A14_KC-135 | aircraft | 1424 | 1424 | 0 |
| 4 | A1_SU-35 | aircraft | 1317 | 1317 | 0 |
| 5 | A2_C-130 | aircraft | 1297 | 1297 | 0 |
| 16 | A13_F-15 | aircraft | 1265 | 1265 | 0 |
| 20 | A17_TU-95 | aircraft | 1114 | 1114 | 0 |
| 8 | A5_F-16 | aircraft | 1017 | 1017 | 0 |
| 6 | A3_C-17 | aircraft | 998 | 998 | 0 |
| 22 | A19_SU-34 | aircraft | 933 | 933 | 0 |

## 图片尺寸分布

| split | long_side_bucket | image_count |
| --- | --- | --- |
| train | 1025-1536 | 30 |
| train | 641-1024 | 4448 |
| train | <=640 | 3 |

## 问题类型汇总

| severity | issue_type | count |
| --- | --- | --- |
| warn | box_extends_outside_image | 43 |
| warn | empty_label_file | 4 |

## 审计输出文件

- `shiyan/data_registry/manifests/v2_data3_manual_revision_classfix/image_manifest.csv`
- `shiyan/data_registry/manifests/v2_data3_manual_revision_classfix/label_manifest.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/objects.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/class_distribution.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/split_summary.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/image_size_buckets.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/duplicate_images.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/scene_groups.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/scene_split_overlap.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/label_issues.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/issue_summary.csv`
- `shiyan/data_registry/audits/v2_data3_manual_revision_classfix/summary.json`
- `shiyan/data_registry/fingerprints/v2_data3_manual_revision_classfix_dataset_fingerprint.txt`

## 后续处理原则

本次审计只读取原始数据，不修改 `shiyan/data/`。若后续人工修标，应先在问题表中登记，再生成新的标签版本，例如 `labels_v1_cleaned`，并对新版本重新审计。
