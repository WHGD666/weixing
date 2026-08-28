# Docker 封装与评测系统说明汇总

本文汇总 `shiyan/docker/` 目录下比赛方提供的 Docker 封装说明、评测系统使用手册、答疑记录和示例 Dockerfile。本文是项目内部执行参考，不替代官方原文；若后续官方说明更新，以最新官方文档和评测系统页面为准。

## 一、来源文件

| 文件 | 作用 |
| --- | --- |
| `目标检测算法Docker封装说明.docx` | 说明最终算法镜像的目录结构、运行入口、输入输出格式、本地 Docker 验证方式。 |
| `赛事评测管理系统参赛队伍使用手册.docx` | 说明登录系统、生成 ACR 命令、镜像打 tag、push、提交评测、查看记录和上传源码报告的流程。 |
| `答疑.md` | 记录线上测评时间、正式提交次数、报告提交要求、CUDA 版本、坐标格式等关键补充信息。 |
| `Dockerfile` | 比赛方提供的镜像构建模板，基于 CUDA 12.1 和 micromamba 创建运行环境。 |

## 二、总体结论

我们当前路线没有走歪，但需要明确区分两个阶段：

1. **Windows 本机研发阶段**
   - 可以继续使用当前 Windows + conda + RTX 4060 Laptop 环境做训练、调参、数据审计、验证集评估和可视化。
   - 当前安装的 `torch 2.5.1+cu121` 与比赛方 CUDA 12.1 方向一致。
   - 不能把 Windows conda 环境直接导出为最终 Docker 的 `environment.yml`。

2. **Linux/WSL/Docker 交付阶段**
   - 最终镜像必须面向 Linux x86_64 / amd64。
   - `environment.yml` 必须在 Linux x86_64 环境中根据实际推理依赖生成。
   - Docker 镜像需要在支持 GPU 的 Linux、WSL2 或云 GPU 环境中完成 `docker run --gpus` 验证。

因此，当前阶段继续在 Windows 上完成实验研发是合理的；但代码从一开始就要避免写死 Windows 路径，并提前按官方 `/input -> /output/result.json` 接口设计推理入口。

## 三、线上测评时间与提交规则

### 预测评

- 时间：`8.27 12:00 - 8.29 17:00`
- 每支队伍可不限次数提交 Docker。
- 预测评只用于测试 Docker 封装和提交流程是否可行。
- 预测评数据不是最终计分数据，预测评分数不具备参考价值。

### 正式测评

- 时间：`8.30 12:00 - 9.5 17:00`
- 每支队伍最多提交 5 次正式测评。
- 取最高测评分数作为线上测评阶段最终得分。
- 首次提交会被系统优先测评。
- 截止后关闭提交通道；已提交但未完成运行的任务会继续跑完。
- 系统中的实时综合排名仅代表提交时刻排名，不一定是最终排名。

### 报告与源码

- 网站提供源代码和报告上传接口。
- 截止时间同为 `9.5 17:00`。
- 报告和源码打成 `.zip` 上传。
- 不要把过大的 `.pth`、训练权重、训练图片、数据集放进报告源码 zip。
- 报告只收截止前最后一次成功上传版本，不要求每次 Docker 提交都重新交报告。

## 四、官方 Docker 目录结构

比赛方推荐的交付目录结构如下：

```text
detector-docker-delivery/
├── Dockerfile
├── environment.yml
├── app/
│   ├── main.py
│   ├── detector.py
│   ├── preprocess.py
│   ├── postprocess.py
│   └── 其他代码文件
└── models/
    └── 模型文件
```

各部分职责：

| 位置 | 要求 |
| --- | --- |
| `Dockerfile` | 直接使用比赛方提供模板，一般不需要修改。 |
| `environment.yml` | 在 Linux x86_64 环境中由实际运行环境自动导出。 |
| `app/` | 放入口文件、推理代码、预处理、后处理、类别表、配置文件等。 |
| `models/` | 放最终推理使用的模型文件，如 `.pt`、`.pth`、`.onnx`、`.engine` 等。 |

注意事项：

- Dockerfile 会把 `app/` 复制到镜像内 `/app/`。
- Dockerfile 会把 `models/` 复制到镜像内 `/app/models/`。
- 推理代码中不要写 `C:\`、`D:\` 等 Windows 绝对路径。
- 运行时统一使用 Linux 容器内路径，例如 `/app/models/xxx.pt`、`/input`、`/output`。

## 五、官方 Dockerfile 关键信息

当前官方 Dockerfile 的核心设计：

```text
FROM nvidia/cuda:12.1.1-base-ubuntu22.04
WORKDIR /app
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n detector -f /tmp/environment.yml
ENV PATH=/opt/micromamba/envs/detector/bin:/opt/micromamba/bin:$PATH
COPY app /app
COPY models /app/models
RUN mkdir -p /input /output
ENTRYPOINT ["python", "/app/main.py"]
```

含义：

- 基础镜像是 Ubuntu 22.04 + CUDA 12.1.1。
- 使用 micromamba 创建名为 `detector` 的环境。
- 环境来源是项目根目录下的 `environment.yml`。
- 容器启动时执行 `/app/main.py`。
- 平台会通过命令行参数传入 `--input` 和 `--output`。

答疑中进一步说明：测评服务器 CUDA 方向已经由 Dockerfile 固定为 CUDA 12.1，因此 PyTorch、ONNX Runtime、TensorRT 等依赖都应选择 CUDA 12.1 兼容版本。

## 六、environment.yml 生成要求

官方重点要求：`environment.yml` 必须对应 Linux x86_64 环境，不能直接使用 Windows conda 环境导出结果。

推荐流程：

```bash
conda activate 你的环境名
conda env export --no-builds \
| sed '/^prefix:/d' \
> environment.yml
```

需要注意：

- 这一步应在 Linux x86_64、WSL2 Ubuntu、Linux GPU 服务器或云 GPU 主机上执行。
- 不要手写最终 `environment.yml`。
- 不要把 Windows 环境里的 `win-64`、`pywin32`、`C:\路径` 写进去。
- 每次修改依赖后都要重新导出。
- 最终 Docker 是否可用，必须通过 GPU 环境下的 `docker run --gpus` 验证。

对我们项目的影响：

- 当前 `shiyan/environment/requirements.txt` 仍然有价值，用于本机训练和队友复现。
- 最终 Docker 阶段要在 Linux/WSL 中按同类依赖重建环境，再导出 `environment.yml`。
- `requirements.txt` 是研发依赖基准；`environment.yml` 是最终交付镜像依赖快照。

## 七、容器运行接口

官方容器入口必须支持：

```bash
python /app/main.py --input /input --output /output
```

其中：

- `/input`：评测系统挂载的测试图片目录。
- `/output`：算法需要写入结果文件的目录。
- 程序只读取 `/input` 第一层图片，不递归读取子目录。
- 支持图片格式：`.jpg`、`.jpeg`、`.png`、`.bmp`。
- 必须生成 `/output/result.json`。

本地 Docker 验证命令示例：

```bash
docker run --rm \
  --gpus '"device=0"' \
  --network none \
  -v "$PWD/test-input:/input:ro" \
  -v "$PWD/test-output:/output" \
  detector-team001:1.0 \
  --input /input \
  --output /output
```

关键点：

- `--network none` 表示容器运行时不能联网。
- 模型权重、类别表、配置文件、推理代码都必须提前打进镜像。
- 不能在运行时下载模型或依赖。
- 必须确认真实使用 GPU，不只是安装了 CUDA 版本 PyTorch。

## 八、result.json 输出格式

官方要求容器输出 `/output/result.json`，结构如下：

```json
{
  "status": "success",
  "images": [
    {
      "image_id": "000001",
      "file_name": "000001.jpg",
      "width": 1920,
      "height": 1080,
      "run_end_timestamp": 1723968000123,
      "objects": [
        {
          "category_id": 1,
          "category_name": "person",
          "score": 0.9321,
          "bbox": [100.5, 200.0, 500.0, 700.0]
        }
      ]
    }
  ]
}
```

字段要求：

| 字段 | 说明 |
| --- | --- |
| `status` | 成功时应为 `success`。 |
| `images` | 所有输入图片的结果列表。 |
| `image_id` | 通常使用图片文件名 stem。 |
| `file_name` | 输入图片文件名。 |
| `width` / `height` | 原图宽高。 |
| `run_end_timestamp` | 每张图片推理结束后立刻记录的 Unix 毫秒时间戳。 |
| `objects` | 当前图片的检测目标列表；没有目标时返回空列表。 |
| `category_id` | 类别 ID，整数。 |
| `category_name` | 类别名称，字符串。 |
| `score` | 置信度，0 到 1。 |
| `bbox` | 原图像素坐标 `[x1, y1, x2, y2]`。 |

特别注意：

- `bbox` 是未归一化的原图像素坐标，不是 YOLO 的 `0-1` 归一化格式。
- `bbox` 不是 COCO 的 `[x, y, width, height]`，而是 `[x1, y1, x2, y2]`。
- 官方赛题正文提到 COCO JSON，但 Docker 评测接口要求的是上述 `result.json`。项目内部应同时支持 COCO 格式和官方 result.json 格式，避免后期转换混乱。

## 九、镜像构建要求

在 Linux x86_64 / WSL 的 Bash 中执行：

```bash
docker build \
  --platform linux/amd64 \
  -t detector-team001:1.0 .
```

构建后检查平台：

```bash
docker image inspect detector-team001:1.0 \
  --format '{{.Architecture}}|{{.Os}}'
```

看到：

```text
amd64|linux
```

才符合目标平台。

注意：

- Windows PowerShell 与 Bash 的换行方式不同，官方示例使用 Bash 反斜杠。
- 如果基础镜像 `nvidia/cuda:12.1.1-base-ubuntu22.04` 拉取失败，答疑中提到需要可访问 Docker Hub 的网络环境。
- 构建成功不代表评测一定成功，还必须完成本地 GPU 运行验证。

## 十、评测系统提交流程

正式提交流程分为两条线：

1. Docker 镜像提交评测。
2. 源代码和研究报告 zip 上传。

### Docker 镜像提交

推荐顺序：

1. 登录赛事评测管理系统。
2. 查看页面显示的本次 tag 和镜像地址。
3. 生成 ACR 临时登录命令。
4. 本地执行 `docker login`。
5. 执行 `docker images` 确认本地镜像存在。
6. 执行页面提供的 `docker tag` 命令。
7. 执行页面提供的 `docker push` 命令。
8. 等待 push 完全成功。
9. 回到系统点击“提交评测”。
10. 在“我的提交记录”中查看任务状态和结果。

注意：

- 系统只会从 ACR 仓库拉取镜像，不会读取本地电脑镜像。
- 目标镜像地址和 tag 必须以系统页面显示为准。
- 同一个 tag 不允许重复提交。
- 正式提交最多 5 次。
- 试运行模式不消耗正式提交次数。

### 源码和报告上传

上传要求：

- 文件格式为 `.zip`。
- 内容为源代码和研究报告。
- 不包含训练权重。
- 不包含训练图片和数据集。
- 文件大小上限以系统页面提示为准，通常不超过 500MB。
- 每支队伍只保留最新一次上传成功文件，重新上传会覆盖旧记录。

## 十一、常见失败原因

### 镜像拉取失败

可能原因：

- 镜像没有成功 push。
- tag 与系统页面显示不一致。
- ACR 仓库权限异常。
- ACR 仓库中不存在该镜像。

处理重点：

- 检查 `docker push` 是否完整成功。
- 检查镜像地址和 tag 是否完全一致。
- 必要时联系管理员查看任务日志。

### 容器运行失败

可能原因：

- 启动命令不支持 `--input /input --output /output`。
- 程序没有读取 `/input`。
- 程序没有写出 `/output/result.json`。
- 容器运行超时。
- 缺少依赖。

处理重点：

- 本地先按官方命令模拟运行。
- 确认 `result.json` 存在。
- 确认推理代码真实使用 GPU。
- 确认容器运行时不依赖网络。

### 任务完成但没有分数

可能原因：

- `result.json` 格式不符合要求。
- 评分脚本无法读取结果。
- 输出字段缺失。
- 坐标格式错误。

处理重点：

- 检查 `status`、`images`、`objects`、`category_id`、`score`、`bbox` 等字段。
- 检查 `bbox` 是否为原图像素坐标 `[x1, y1, x2, y2]`。
- 检查所有输入图片是否都有对应结果记录。

## 十二、对我们项目架构的要求

后续代码设计应满足：

1. **训练与交付解耦**
   - 训练代码可以保留在 `shiyan/src/`、`shiyan/scripts/`、`shiyan/configs/`。
   - 最终 Docker 只打包推理必需代码、配置和模型。

2. **推理入口提前兼容官方接口**
   - 支持 `--input` 和 `--output` 参数。
   - 对目录内多张图片逐图推理。
   - 输出 `/output/result.json`。

3. **内部格式与提交格式分开**
   - 训练标签：YOLO 归一化水平框。
   - 内部评估：可使用 COCO JSON 或自定义指标表。
   - 官方 Docker 输出：`result.json`，框为 `[x1, y1, x2, y2]` 原图像素坐标。

4. **大图推理必须工程化**
   - 10000x10000 图像不能简单整图推理。
   - 需要切片、重叠、坐标还原、跨切片 NMS/WBF、耗时统计。
   - 大图推理逻辑应在本地实验阶段就验证，不能等最终 Docker 才补。

5. **无网络运行**
   - 权重、类别表、配置、模型结构代码全部本地化。
   - 禁止运行时自动下载预训练权重。

6. **路径必须跨平台**
   - 使用 `pathlib.Path`。
   - 不写死 Windows 绝对路径。
   - 本地数据路径通过配置或命令行传入。
   - Docker 内部路径固定为 `/app`、`/input`、`/output`、`/app/models`。

## 十三、提交前检查清单

Docker 提交前至少检查：

- [ ] `Dockerfile`、`environment.yml`、`app/`、`models/` 均在交付目录中。
- [ ] `environment.yml` 来自 Linux x86_64 环境，不含 Windows 专用路径和包。
- [ ] 镜像可成功 `docker build --platform linux/amd64`。
- [ ] `docker image inspect` 显示 `amd64|linux`。
- [ ] 容器可用 `--gpus` 运行。
- [ ] 本地模拟命令中使用 `--network none` 仍可推理。
- [ ] `/output/result.json` 正常生成。
- [ ] `result.json` 包含全部输入图片。
- [ ] 每张图片包含 `file_name`、`width`、`height`、`run_end_timestamp`。
- [ ] 每个目标包含 `category_id`、`category_name`、`score`、`bbox`。
- [ ] `bbox` 为原图像素坐标 `[x1, y1, x2, y2]`。
- [ ] 无目标图片输出空 `objects`，而不是漏掉图片记录。
- [ ] 模型权重和配置已打入镜像，不依赖网络下载。
- [ ] 推理速度在本地或近似环境中经过测试。
- [ ] 源码报告 zip 不包含训练图片、数据集、大权重。

## 十四、当前项目状态判断

结合官方 Docker 说明，当前项目状态可以判断为：

- 本机训练环境路线正确：CUDA 12.1 方向与官方 Dockerfile 一致。
- 继续 Windows 本机实验合理：官方只要求最终 `environment.yml` 在 Linux x86_64 生成，不要求研发全过程在 Linux。
- 项目管理路线正确：先数据审计、冻结划分、建立 baseline，再进入模型改进和最终 Docker 封装。
- 后续需要新增的不是“马上写 Docker”，而是从第一版推理代码开始预留官方接口。

下一步合理顺序：

1. 完成本地环境锁定记录。
2. 放入本地数据但不上传 Git。
3. 做数据审计和标签合法性检查。
4. 冻结验证集划分。
5. 跑第一版 YOLO baseline。
6. 基于 baseline 形成可复现推理入口。
7. 模型稳定后再整理 Docker 交付目录和 Linux `environment.yml`。
