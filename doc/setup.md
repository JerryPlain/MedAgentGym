# MedAgentGym Setup Guide (从 0 到首次跑通)

这份文档的目标是：
从一台干净机器开始，把 MedAgentGym 配好，并成功跑出第一条 `history_*.json` 结果。

---

## 1. 你将获得什么
完成本指南后，你会得到：
1. 可运行的 Python/Docker 环境。
2. 可用的 LLM 凭证配置。
3. 一个最小可执行实验配置（`biocoder`）。
4. 首次成功输出：`workdir/.../history_0.json`。

---

## 2. 先决条件

## 2.1 系统与工具
1. macOS / Linux（推荐 Linux + NVIDIA GPU 做完整实验）。
2. `git`。
3. `python3.11`（本地运行推荐）。
4. Docker（如果走容器方案）。

## 2.2 模型凭证
项目运行时会从 `credentials.toml` 读取并注入环境变量。
至少准备其中一套：
1. OpenAI：`OPENAI_API_KEY`
2. Azure OpenAI：`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` + `API_VERSION`

注意：某些 action/任务会用到 `WOLFRAM_ALPHA_APPID`，建议也填上。

## 2.3 数据说明
仓库内置了任务定义文件（`train_tasks.jsonl` / `test_tasks.jsonl`），但部分完整数据需要额外数据协议授权。
首次跑通建议用仓库现有可用任务目录先做 smoke test（例如 `data/biocoder`）。

---

## 3. 获取代码
```bash
git clone <your_repo_url>
cd MedAgentGym
```

---

## 4. 路线 A：本地 Python 环境（推荐先走这条）

## 4.1 创建虚拟环境
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

如果安装 `torch` 有平台问题，可按你的 CUDA/CPU 环境单独安装对应版本。

## 4.2 创建运行目录
`validate_code` 会把临时代码写入 `./cache`，建议先创建：
```bash
mkdir -p cache workdir
```

## 4.3 配置凭证
编辑根目录下 `credentials.toml`。

最小 OpenAI 示例：
```toml
OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY>"
WOLFRAM_ALPHA_APPID = "<YOUR_WOLFRAM_ALPHA_APPID>"
```

最小 Azure 示例：
```toml
AZURE_OPENAI_API_KEY = "<YOUR_AZURE_OPENAI_API_KEY>"
AZURE_OPENAI_ENDPOINT = "<YOUR_AZURE_OPENAI_ENDPOINT>"
API_VERSION = "2024-05-01-preview"
WOLFRAM_ALPHA_APPID = "<YOUR_WOLFRAM_ALPHA_APPID>"
```

---

## 5. 路线 B：Docker 环境

## 5.1 构建镜像
```bash
docker buildx build -t ehr_gym:latest .
```

## 5.2 启动容器
可参考仓库的 `test_docker.sh`（按你的本地路径替换挂载）。

最简建议：先在本地跑通后再迁移到容器，因为容器挂载项较多、排障成本更高。

---

## 6. 创建最小可运行配置
在 `configs/` 下新建 `local-demo-biocoder.yaml`：

```yaml
Agent:
  llm:
    model_type: "OpenAI"
    model_name: "gpt-4.1-mini"
    max_total_tokens: 32768
    max_input_tokens: 8192
    max_new_tokens: 4096
    log_probs: False
    temperature: 0.0
  n_retry: 3
  retry_delay: 5
Data:
  metadata_path: "data/metadata.json"
  data_path: "data/biocoder"
Debugger:
  model_type: "OpenAI"
  model_name: "gpt-4.1-mini"
  max_total_tokens: 32768
  max_input_tokens: 8192
  max_new_tokens: 2048
  log_probs: False
  temperature: 0.0
Env:
  n_retry: 3
task: "biocoder"
credentials_path: "./credentials.toml"
work_dir: "./workdir/demo"
result_dir_tag: "demo-biocoder-k1"
start_idx: 0
end_idx: 1
num_steps: 8
```

如果你用 Azure，把 `model_type` 改成 `Azure`，并补 `deployment_name` 字段。

---

## 7. 首次运行（main.py）
```bash
python main.py --config_path configs/local-demo-biocoder.yaml
```

注意：
1. 入口参数是 `--config_path`，不是 `--config`。
2. 仓库里 `entrypoint.sh` 目前写的是 `--config`，这是旧写法。

---

## 8. 结果验证（确认真的跑起来）

运行成功后，检查目录：
```bash
ls -la workdir/demo/biocoder/demo-biocoder-k1/test/
```

你应看到类似文件：
1. `history_0.json`

再快速看内容：
```bash
sed -n '1,120p' workdir/demo/biocoder/demo-biocoder-k1/test/history_0.json
```

这个文件包含：
1. 模型每轮 action。
2. 每轮执行反馈（代码输出或报错）。
3. 最终 `result: success/failure`。

---

## 9. 运行流程你实际触发了什么
执行 `python main.py --config_path ...` 时：
1. 读取 yaml 配置。
2. 从 `credentials.toml` 注入环境变量。
3. `get_task_class("biocoder")` 选择任务类。
4. 创建 `EHREnv` + `EHRAgent`。
5. `env.reset(0)` 装载第 0 条样本。
6. 循环执行 action -> `env.step` -> `task.validate`。
7. 保存轨迹到 `history_0.json`。

---

## 10. 常见报错与排查

## 10.1 报错：`unrecognized arguments: --config`
原因：使用了旧参数名。
修复：改为 `--config_path`。

## 10.2 报错：`AZURE_OPENAI_ENDPOINT has to be defined`
原因：使用 Azure 但凭证未配置。
修复：检查 `credentials.toml` 是否有完整 Azure 三元组。

## 10.3 报错：写 `./cache/...` 失败
原因：`cache` 目录不存在或权限问题。
修复：`mkdir -p cache workdir`。

## 10.4 报错：任务文件不存在
原因：`data_path` 配错，或该任务数据未就绪。
修复：先切换到已存在目录（如 `data/biocoder`）做最小验证。

## 10.5 报错：LLM API 429/超时
原因：限流或网络问题。
修复：
1. 降低并发（先不用 `--async_run`）。
2. 减小 `max_new_tokens`。
3. 适当增大重试间隔 `retry_delay`。

---

## 11. 跑通后下一步
1. 把 `end_idx` 从 `1` 提升到 `5/10`，观察稳定性。
2. 再尝试 `--async_run --parallel_backend joblib --n_jobs 2` 做并行。
3. 改用 `rollout.py` 生成多轨迹采样结果。
4. 切换到 EHR 类任务（如 `ehrcon`/`eicu`）验证 `request_info` 相关流程。

---

## 12. 一条最短“成功路径”命令清单
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
mkdir -p cache workdir
# 编辑 credentials.toml
# 新建 configs/local-demo-biocoder.yaml
python main.py --config_path configs/local-demo-biocoder.yaml
ls -la workdir/demo/biocoder/demo-biocoder-k1/test/
```
