# h3img-metal（中文文档）

Apple Silicon 上的 **MiniMax-H3-Image** 原生推理引擎，从零用 C11 与 Objective-C Metal 编写——**不依赖 PyTorch、MLX，运行时无需 Python**。一个二进制覆盖文生图与基于参考图的编辑（`--t2i`、`serve`、`repl`）。每个阶段都对照官方 diffusers 管线与随仓库分发的 fixture 验证：确定性部分做到逐位对拍，完整生成用语义门控。当前工作是在 M3 Max 和 M5 Max 上持续进行 H3-Image 专属的 Metal 性能与内存优化。

[English](README.md)

## MiniMax-H3-Image：原生图像生成

从零开始的 MiniMax-H3-Image 原生引擎（`--t2i`、`serve`、`repl`），并与 diffusers 官方管线（`/Volumes/Work/diffusers-h3img/.venv`）做了逐阶段对拍验证。

### 管线

1. **文本** — 逐字提示词分词（无模板、无特殊 token），随后是 Qwen3-VL 语言模型的前 50 层；pre-norm 终态作为条件行。
2. **视觉（仅 edit）** — 每张参考 PNG 先合成到白底（与参考栈的 alpha 语义一致），再缩放到短边 1024（对齐到 32），由 Qwen3-VL 视觉塔编码为合并后的 5120 维行（以 "Picture i: " 块形式前置于提示词）。同一张参考图另行 VAE 编码（短边钳到 [1024, 2048]），用逐通道潜变量统计量归一化，加噪到 t=0.999 条件强度，并 patchify 为 96 宽的行。
3. **DiT** — 打包序列的 FL2VA transformer（50 块：文本行 + 一个潜变量帧 + 参考行）。采样器是 `pdd_config.json` 中的 pdd 蒸馏六步 Euler/velocity 网格，逐字使用；终止 sigma 为零。**没有 CFG、没有负向提示词**——检查点是引导蒸馏的，步数固定。
4. **VAE** — ViT 解码器按 tile 运行（256 px tile、64 px 重叠、线性 cross-fade 拼接），镜像官方 `_decode` 的分块方式。潜变量在解码前逐通道反归一化；解码输出为 ImageNet 归一化 RGB，PNG 量化前按官方 DecodeStep 用 `pixel_mean/pixel_std` 还原。

### 生成画布（质量与速度）

模型在 2048 级画布上训练：小的直接尺寸属于分布外。与官方管线一样，引擎在请求宽高比下以至少 `min_generation_area`（默认 2048×2048）的画布生成，解码后的 8-bit 图像再经 bicubic 缩回请求尺寸：

| 请求尺寸 | 画布 | 典型耗时（M5 Max） | 峰值内存 |
|---|---|---|---|
| 256×256 | 2048×2048（默认下限） | ~2 分钟 | ~6.6 GB |
| 512×512 | 2048×2048（默认下限） | ~2 分钟（常驻会话热态 ~63 s） | ~6.6 GB |
| 任意 ≤1024² | 1024×1024（`H3IMG_MIN_AREA=1048576`，快档） | ~22 s | ~6.1 GB |
| 2048 级 | 2048×2048 | ~108 s | ~6.6 GB |
| 4096×4096 | 4096×4096 | ~284 s | ~6.4 GB |
| 6272×2688（21:9，官方最大桶） | 6272×2688 | ~296 s | ~5.9 GB |

支持的输出尺寸遵循官方尺寸表：每个桶约 16.8 MP、每边至多 6272 px
（`4096×4096`、`4736×3552`、`5024×3360`、`5472×3072`、`6272×2688` 及各自的
镜像宽高比；全部为 32 的倍数）。每边超过 6272 px 的请求直接拒绝。上表峰值内存取
一次性运行的**进程** footprint（`/usr/bin/time` peak RSS）；`repl` 连续生成 10 张的
会话峰值为 ~8.5 GB，无逐张棘轮增长。

#### "峰值内存"到底是哪个数（实测）

有 **三个不同的数**，不可相加混淆：

| 指标（一次性 6 步生成，M5 Max） | 1024² 快档 | 2048² 默认档 |
|---|---|---|
| 系统内存净增峰值（`active+wired+compressed` 相对空闲基线） | **+38.9 GB** | **+39.5 GB** |
| 其中 file-backed pages（mmap 权重被 fault 进内存） | +40.8 GB | +37.1 GB |
| 进程 `phys_footprint` 峰值（活动监视器"内存"列） | 5.7 GB | 5.6 GB |
| 引擎 `peak_live_bytes`，DiT（逻辑 MTLBuffer 账本，mmap 权重建 buffer 即计入） | 36.2 GiB | 36.9 GiB |
| 真实设备分配，teardown（`MTLDevice.currentAllocatedSize`） | ~2.4–4.0 GiB | ~2.4–4.4 GiB |

约 **39 GB 的系统净增**就是 Web UI / 活动监视器在任务期间显示、被人称作"不到 40G"的那个数。
它主要来自以 `mmap` + `newBufferWithBytesNoCopy` 映射的 62 GB `transformer` 权重：GPU 每读一页权重，
该页就被 fault 进系统文件缓存（`File-backed pages`）。这些页是干净、可回收的（内存紧张时内核直接丢弃），
**不计入进程 footprint**。62 GB 的 `text_encoder` 走 pread 流式读入，只贡献较小的瞬时量。

引擎的 `peak_live_bytes`（~36.9 GiB）是**逻辑**计数器：在 mmap 权重上建零拷贝 Metal buffer 时立即按全尺寸计入，
即使实际只有触碰到的页驻留物理内存。它是归因信号（也是"权重加总看起来很大"的来源），不是物理占用。
真实 GPU 设备分配始终只有几 GiB。

峰值**基本与分辨率无关**——它由"哪些权重被映射/fault 进来"决定，而非画布大小：
1024² 的 +38.9 GB 与 2048² 的 +39.5 GB 相差不足 1 GB。会话 footprint 不棘轮：`repl` 连跑 10 张峰值 ~8.5 GB 并回落基线。

复现方法（需先停掉任何 `serve`/`repl` 释放 GPU 锁）：

```sh
# 一次性生成期间以 4 Hz 采样系统 used + 进程 phys_footprint
python3 /tmp/h3img_mem_probe.py -- \
  ./h3img --t2i -d /Volumes/Work/models/MiniMax-H3-Image \
  -p "A red apple on a wooden table" --width 2048 --height 2048 -o /tmp/out.png
# 加 H3_PROFILE=1 H3_DEBUG_GPU_MEMORY=1 可打印逻辑/真实 GPU 账本
```

`H3IMG_MIN_AREA` 覆盖下限（低于 1024² 钳到 1024²；官方管线接受 1024²..4096²）。默认 2048² 是官方质量行为；1024² 下限是快档。`--latents`（A/B harness）锁定 fixture 几何、跳过画布逻辑。尺寸必须是 32 的倍数。

### 一次性 CLI

```sh
make -j8
H3IMG_MIN_AREA=1048576 ./h3img -d /Volumes/Work/models/MiniMax-H3-Image --t2i \
  -p "A red apple on a wooden table" \
  --width 512 --height 512 --seed 7 -o outputs/apple.png

# 编辑：可重复的 --ref-image；提示词可用 <Picture i> 引用。
./h3img -d /Volumes/Work/models/MiniMax-H3-Image --t2i \
  -p "Turn the apple into a green lime while keeping the composition" \
  --ref-image apple.png --width 512 --height 512 -o outputs/lime.png
```

`--latents FILE` 注入 fp32 初始潜变量（`x.latents` [24, H/16, W/16]，edit 另有 `x.ref_latents_<i>`/`x.ref_noise_<i>`），使两套栈回放完全相同的噪声——引擎自身的噪声 RNG 无法与参考栈逐 RNG 对齐。

### serve：JSONL worker

`h3img serve -d MODEL_DIR` 使用与其他原生引擎一致的 JSONL worker 协议族：stdin 收请求、stdout 出事件（行缓冲）、stderr 出诊断日志。请求串行处理；权重为 mmap 惰性、逐请求加载（相对 30 s+ 的生成只需几秒），因此除组件存在性检查外没有独立加载阶段。

请求：

```json
{"id":"t1","cmd":"ping"}
{"id":"t2","prompt":"A red apple on a wooden table","width":512,
 "height":512,"seed":7,"output":"/tmp/apple.png"}
{"id":"t3","prompt":"Turn the apple into a lime","width":512,"height":512,
 "seed":7,"output":"/tmp/lime.png","images":["/path/apple.png"]}
```

事件：`state`（`loading`/`ready` 带 `loadSeconds`，或 `crashed` 带 `message`）、`progress`（`stage` ∈ encode/sample/decode/save，全局 `percent` 0..1，限频 ~4 Hz）、`heartbeat`（请求进行中每 4 s 一次，按 `id` 关联）、`done`（`path`、`width`、`height`、`steps` = 6、`elapsed`）、`error`（`id`、`message`）。`progress`、`heartbeat`、`done`、`error` 都带请求 `id`（截断到 80 字节），宿主据此把事件与任务对齐。

校验：`id`、`prompt`、`output`（≤512 字节）必填；`width`/`height` 必须是 [32, 8192] 内 32 的倍数的数字；`seed` 是无符号 64 位数字（默认 42）；`images` 是最多 8 个路径字符串的数组。存在但类型错误的数字字段与 `images` 会使请求失败，而不是静默回退默认值。`steps` 与 `negativePrompt` 字段接受但忽略（蒸馏检查点）。

运维说明：

- GPU 单实例锁（`/tmp/h3img-engine-<uid>.lock`，非阻塞 flock）：同一台机器上的第二个 serve/repl 会被拒绝。
- 每个模块都相对进程 cwd 编译 `h3_shaders.metal`；serve/repl 会自动重锚到引擎目录。相对输出路径以该锚点解析（文档化行为）。
- stdin EOF 干净退出；取消由宿主杀进程组实现（进程内无取消）。
- 编译产物与 pipeline state 有进程级缓存（首次编译后同进程内复用）。

### repl：交互会话

`h3img repl -d MODEL_DIR` 维护一份生成设置的会话（linenoise 提示符 `h3img>`）。普通行即提示词；每次生成写出 `PREFIX-<n>.png` 并自动递增 seed。

```text
/width N /height N   输出尺寸，[32, 8192] 内 32 的倍数（默认 1024）
/seed N              下一个 seed（每张之后自动递增）
/out PREFIX          输出文件名前缀（默认 "out"）
/image PATH          添加一张编辑参考（最多 8 张）；裸 /image 或 /image clear
                     清空全部（回到 t2i）
/refs                列出当前参考
/show                当前设置
/help /quit
```

### 测试

`make test` 对每个图像阶段以官方 oracle 设门（fixture 在 `misc/fixtures/`，参考权重在 `/Volumes/Work/models/MiniMax-H3-Image`；任一缺失时对应项跳过并提示）：

| 阶段 | 门限 |
|---|---|
| text parity | layer-49 rel-L2 ≤ ~0.05（实测 0.040） |
| DiT parity | 三个 velocity 步，rel-L2 ≤ ~1.3 混沌带（实测 ~1.1） |
| VAE parity | latents/pixels rel-L2 ≤ 5e-4（实测 ~1e-6） |
| t2i e2e | 与官方做语义门（直方图相关、LapVar 比；fixture 为 1024² 真实图像，实测均差 6.5/255、corr 0.973） |
| edit e2e | 终态潜变量 rel-L2 ≤ 0.35（实测 ~0.13）+ 图像门 |

诊断专用（不在 `make test` 内）：`make h3img-text-fixture`、`make h3img-vision-fixture`（用参考 venv 重新生成 fixture）与 `make h3img-vision-ab`（逐阶段视觉塔 A/B，约 2 分钟）。

### 环境变量

| 变量 | 作用 |
|---|---|
| `H3IMG_MIN_AREA` | 生成画布下限（见上表） |
| `H3IMG_VAE_CPU` | 强制 CPU GEMM（Accelerate）VAE 路径 |
| `H3IMG_VAE_SCALAR` | 强制纯标量 VAE 参考路径 |
| `H3IMG_DUMP_LATENTS` | 把视觉合并行、layout、文本嵌入、终态潜变量 dump 到 `/tmp` |
| `H3IMG_VISION_DUMP=DIR` | 视觉塔逐阶段 dump（配合 `check_vision_ab.py`） |
| `H3IMG_VAE_DUMP` | VAE 解码逐阶段 dump |
| `H3IMG_DIT_DUMP` | DiT 逐阶段 dump（block 0/24 张量、schedule） |
| `H3IMG_DIT_STEP0_DUMP` | dump 第 0 步 DiT 张量 |
| `H3IMG_EDIT_TRACE` | edit 路径 trace 行（layout 分段、参考加载） |
| `H3IMG_TIMING=1` | 向 stderr 打印各管线阶段的累计墙钟秒数（text+vision、DiT load、denoise、VAE decode、PNG） |
| `H3_DISABLE_FUSED_MLP`、`H3_DIT_F32_FINAL` | 由管线自行设置：fused-MLP 快路在此检查点上有已知 Metal 故障，且最终 head 保持 F32 |

`H3_PROFILE` 同样会传入共享的 DiT 内部；在图像路径上它为各组件增加 GPU 账本（wall/encode/wait/device-busy、submissions、direct dispatch、linear encode 次数）。两个开关一起跑一次 2048²，典型值：text 5.1 s、DiT load+AdaLN 7.9 s、6 步去噪 36 s（GPU-bound）、VAE decode 8.4–8.6 s、PNG 0.1 s。前两项是每请求固定开销，`H3IMG_TIMING` 让它们可见。

### 图像解码：命令串链

VAE ViT 解码器每个 block 的五个阶段（QKV、attention、attention-out、FF1、FF2）编成一条 Metal 命令链：阶段 1–4 用 `h3_gpu_continue`（提交但不等待），只有阶段 5 提交，于是每个 block 只有一次 CPU/GPU 同步而非五次。scratch 张量活到那次终止提交之后才释放，避开 qwen-metal P5 踩过的「释放仍被在飞命令引用的 buffer」隐患。这一项把 2048² VAE decode 从 ~9.9 s 降到 ~8.4–8.6 s，输出逐位不变。

融合 Q/K/V linear（三个 `[1024,1024]` matmul 拼成一个 `[1024,3072]` 加 split 内核）曾实现并验证逐位正确，但最终**回退**：虽然 linear encode 次数少了三分之一（17545 → 11737），更大的 matmul 加额外的 split dispatch 让墙钟反而慢了约 5%。那 17545 次 linear 是 24 层 × 6 linear × 121 tile 的 1024² MPS GEMM；当每次都已跑在硬件效率上时，减少次数并无帮助。

### 已知限制

- 采样步数固定为 6（`pdd_config.json`）；无 CFG、无负向提示词、无逐请求步数控制。
- 管线逐请求自包含：文本编码、AdaLN 预计算与惰性权重 mmap 重载合计每请求 ~10 s（M5 Max 实测）。编译的 Metal shader 与 pipeline state 有进程级缓存，同一 serve/repl 会话内的重复模块上下文不会重编。常驻 GPU 重构（权重跨请求驻留，~60 GB 驻留）可行但未启动。

## 构建与测试

```sh
make -j8          # 构建 h3img 二进制
make test         # 上文表中所有图像阶段的对拍门控
```

`make test` 运行确定性的宿主测试与图像各阶段的官方对拍。每个阶段依赖 `misc/fixtures/` 下的 fixture 与 `/Volumes/Work/models/MiniMax-H3-Image` 下的参考权重；缺少对应文件时该阶段打印 `skip`。

前置条件：配备 Metal 3+ 的 Apple Silicon、C11 工具链（Xcode 自带的 `clang`）以及 H3-Image 模型检查点。运行时不需要 Python、PyTorch 或 MLX —— 参考 diffusers venv 只用于通过诊断目标 `make h3img-*-fixture` 重新生成 fixture。

诊断目标（不在 `make test` 内）：`make h3img-text-fixture`、`make h3img-vision-fixture`、`make h3img-vision-ab`（视觉塔逐阶段 A/B，约 2 分钟）。

## 许可

MIT —— 见 [LICENSE](LICENSE)。第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
