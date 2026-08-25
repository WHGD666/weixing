# Label Contract v0

状态：draft，等待数据审计后冻结。

## 类别范围

本项目按 25 个细分类训练和评估。

| ID | Name | Group |
| ---: | --- | --- |
| 0 | HM | ship |
| 1 | LQS | ship |
| 2 | QHS | ship |
| 3 | MS | ship |
| 4 | A1_SU-35 | aircraft |
| 5 | A2_C-130 | aircraft |
| 6 | A3_C-17 | aircraft |
| 7 | A4_C-5 | aircraft |
| 8 | A5_F-16 | aircraft |
| 9 | A6_TU-160 | aircraft |
| 10 | A7_E-3 | aircraft |
| 11 | A8_B-52 | aircraft |
| 12 | A9_P-3C | aircraft |
| 13 | A10_B-1B | aircraft |
| 14 | A11_E-8 | aircraft |
| 15 | A12_TU-22 | aircraft |
| 16 | A13_F-15 | aircraft |
| 17 | A14_KC-135 | aircraft |
| 18 | A15_F-22 | aircraft |
| 19 | A16_FA-18 | aircraft |
| 20 | A17_TU-95 | aircraft |
| 21 | A18_KC-10 | aircraft |
| 22 | A19_SU-34 | aircraft |
| 23 | A20_SU-24 | aircraft |
| 24 | FSC | vehicle |

## 判定原则

- 子类分类正确才可能计为 TP。
- 舰船大类包含 ID 0-3。
- 飞机大类包含 ID 4-23。
- 车辆大类当前只包含 ID 24，即发射车。
