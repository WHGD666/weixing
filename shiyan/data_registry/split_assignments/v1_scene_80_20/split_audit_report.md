# 划分审计报告：v1_scene_80_20

生成时间：2026-08-28T11:20:27.035680+00:00

输入审计版本：`v0_original`

划分指纹：`126b297c2706c71b8be8df4e3736b9b30871f18673147df1c3e31e3997f8ec77`

## 划分原则

- 原始数据不移动、不复制、不重命名、不修改。
- 划分单元为 `scene_id`，即文件名中 `_cropN` 之前的部分。
- 同一个 `scene_id` 下的切片全部进入同一个 split。
- 目标比例为 train:val = 8:2。
- 在候选随机划分中选择类别比例更接近 8:2、且稀有类别验证集不为空的方案。

## split 摘要

| split | images | image_ratio | scenes | objects | object_ratio |
| --- | --- | --- | --- | --- | --- |
| train | 3584 | 0.799821 | 2900 | 16690 | 0.797306 |
| val | 897 | 0.200179 | 725 | 4243 | 0.202694 |

## 大类分布

| group | train_objects | val_objects | total_objects | val_ratio |
| --- | --- | --- | --- | --- |
| ship | 2156 | 526 | 2682 | 0.196122 |
| aircraft | 14213 | 3636 | 17849 | 0.203709 |
| vehicle | 321 | 81 | 402 | 0.201493 |

## 25 类分布

| class_id | class_name | group | train_objects | val_objects | total_objects | val_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | HM | ship | 14 | 3 | 17 | 0.176471 |
| 1 | LQS | ship | 24 | 6 | 30 | 0.2 |
| 2 | QHS | ship | 499 | 142 | 641 | 0.221529 |
| 3 | MS | ship | 1619 | 375 | 1994 | 0.188064 |
| 4 | A1_SU-35 | aircraft | 1022 | 295 | 1317 | 0.223994 |
| 5 | A2_C-130 | aircraft | 1083 | 214 | 1297 | 0.164996 |
| 6 | A3_C-17 | aircraft | 778 | 220 | 998 | 0.220441 |
| 7 | A4_C-5 | aircraft | 399 | 101 | 500 | 0.202 |
| 8 | A5_F-16 | aircraft | 852 | 165 | 1017 | 0.162242 |
| 9 | A6_TU-160 | aircraft | 296 | 65 | 361 | 0.180055 |
| 10 | A7_E-3 | aircraft | 418 | 129 | 547 | 0.235832 |
| 11 | A8_B-52 | aircraft | 598 | 152 | 750 | 0.202667 |
| 12 | A9_P-3C | aircraft | 729 | 166 | 895 | 0.185475 |
| 13 | A10_B-1B | aircraft | 614 | 148 | 762 | 0.194226 |
| 14 | A11_E-8 | aircraft | 335 | 97 | 432 | 0.224537 |
| 15 | A12_TU-22 | aircraft | 477 | 106 | 583 | 0.181818 |
| 16 | A13_F-15 | aircraft | 1008 | 257 | 1265 | 0.203162 |
| 17 | A14_KC-135 | aircraft | 1156 | 268 | 1424 | 0.188202 |
| 18 | A15_F-22 | aircraft | 379 | 114 | 493 | 0.231237 |
| 19 | A16_FA-18 | aircraft | 1675 | 472 | 2147 | 0.219842 |
| 20 | A17_TU-95 | aircraft | 884 | 230 | 1114 | 0.206463 |
| 21 | A18_KC-10 | aircraft | 208 | 54 | 262 | 0.206107 |
| 22 | A19_SU-34 | aircraft | 707 | 226 | 933 | 0.242229 |
| 23 | A20_SU-24 | aircraft | 595 | 157 | 752 | 0.208777 |
| 24 | FSC | vehicle | 321 | 81 | 402 | 0.201493 |

## 稀有类别覆盖

| class_id | class_name | group | train_objects | val_objects | total_objects | val_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | HM | ship | 14 | 3 | 17 | 0.176471 |
| 1 | LQS | ship | 24 | 6 | 30 | 0.2 |
| 21 | A18_KC-10 | aircraft | 208 | 54 | 262 | 0.206107 |
| 9 | A6_TU-160 | aircraft | 296 | 65 | 361 | 0.180055 |
| 24 | FSC | vehicle | 321 | 81 | 402 | 0.201493 |
| 14 | A11_E-8 | aircraft | 335 | 97 | 432 | 0.224537 |
| 18 | A15_F-22 | aircraft | 379 | 114 | 493 | 0.231237 |
| 7 | A4_C-5 | aircraft | 399 | 101 | 500 | 0.202 |

## 自动检查

| check | status | detail |
| --- | --- | --- |
| scene_disjointness | pass | 0 overlapping scenes |
| all_classes_present_in_train | pass |  |
| all_classes_present_in_val | pass |  |
| target_val_image_ratio | pass | 0.200179 |

## 输出文件

- `train.txt`：训练图片清单。
- `val.txt`：验证图片清单。
- `image_assignments.csv`：每张图片的 split 分配。
- `scene_assignments.csv`：每个场景组的 split 分配。
- `class_distribution_by_split.csv`：25 类 train/val 目标数量。
- `group_distribution_by_split.csv`：舰船、飞机、发射车三大类目标数量。
- `split_summary.csv`：图片、场景、目标数量摘要。
- `audit_checks.csv`：划分质量自动检查。
- `metadata.json`：划分参数、随机种子、搜索诊断和指纹。
- `split_fingerprint.txt`：划分指纹。
- `shiyan/configs/dataset/weixing_v1_scene_80_20.yaml`：YOLO 数据配置。

## 后续使用

训练时应优先使用 `shiyan/configs/dataset/weixing_v1_scene_80_20.yaml`，不要直接使用官方 `shiyan/data/dataset.yaml`。官方原始数据目录保持不变，后续所有正式实验都需要记录本 split 版本和划分指纹。
