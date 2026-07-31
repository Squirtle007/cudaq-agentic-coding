# H200 課程母 VM：人工複驗、OpenCode 重現與映像封裝前 Runbook

適用課程：CUDA-Q Agentic Coding
驗收映像：`nchc/cudaq-agentic-coding:2026-07-17`
正式教材：`/home/ubuntu/cudaq-agentic-coding`
乾淨快照：`/opt/nchc-cudaq-course/source`
操作工具：`/home/ubuntu/cudaq-course-kit`

本文件的目的，是讓講師在封裝母 VM 前，以與學生相同的方式手動做一次完整驗收。完成本文件後，依「映像封裝前清理」移除講師個人憑證與測試資料，再刻意保留由專用腳本建立的 root-only 活動憑證包，複製成 40 台。此方案已接受學生可透過 sudo 擷取共用 key 的風險。

## 1. 本版已實測的基準

正式 H200 環境已通過：

- H200、Docker、Compose、NVIDIA container runtime。
- CUDA-Q `qpp-cpu` 與 `nvidia` target smoke test。
- JupyterLab 4.6.1、Jupyter AI 3.0.1、OpenCode 1.18.1、Qiskit 2.5.0。
- JupyterLab 由主機 `0.0.0.0:8888` 發布，並以 Jupyter token 保護。
- `pip check` 無相依衝突。

最新盲測使用公開 `main` commit `1f6358477113b64681c296507854abbf77602438`，結果如下：

| Notebook | 結果 | 實測摘要 |
|---|---|---|
| 01 | PASS | Nemotron 完成；best energy 10.472883；385/20,000 valid，1.925%；clean-kernel 與 validator PASS |
| 02 | FAIL | Nemotron RAP stream 停滯；Gemma 31B fallback 反覆產生無效 CUDA-Q kernel 語法，五個 code cells 未執行 |
| 03 | NOT RUN | 依 AGENTS.md 在第一個實質失敗停止 |
| 04 | NOT RUN | 依 AGENTS.md 在第一個實質失敗停止 |

因此 Step 2–4、Skill 與 BYOC 仍必須由講師依本 runbook 重新人工驗收，不能沿用舊版「全部通過」結論。Agent 若遇 tool-call JSON 錯誤，只允許針對失敗的最小操作重試；若同一 compiler error 重複出現，保存 log 並停止，不以人工答案掩蓋。

## 2. 憑證、model ID 與學生申請方式

### 2.1 課程所用設定名稱

| Provider | Model ID | Key 變數 |
|---|---|---|
| NCHC RAP Nemotron Super（預設） | `NVIDIA-Nemotron-3-Super-120B-A12B` | `RAP_NEMOTRON_3_ULTRA_API_KEY` |
| NCHC RAP Nemotron Ultra | `NVIDIA-Nemotron-3-Ultra-550B-A55B` | 與 Super 共用 `RAP_NEMOTRON_3_ULTRA_API_KEY` |
| NCHC RAP Gemma 26B | `gemma-4-26B-A4B-it` | `RAP_GEMMA_26B_API_KEY` |
| NCHC RAP Gemma 31B | `gemma-4-31B-it` | `RAP_GEMMA_31B_API_KEY` |
| NVIDIA hosted NIM | `nvidia/nemotron-3-ultra-550b-a55b` | `NVIDIA_NIM_API_KEY` |

Lab container 會將 `NVIDIA_NIM_API_KEY` 同值映射為 `NVIDIA_API_KEY`，供 `_intro_Ising_Calibration.ipynb` 使用；學生不需把 key 貼入 Notebook。

Endpoint：

- NCHC RAP：`https://portal.genai.nchc.org.tw/api/v1`
- NVIDIA hosted NIM：`https://integrate.api.nvidia.com/v1`

舊環境若使用拼錯的 `RAP_NENOTRON_3_ULTRA_API_KEY`，kit 仍相容；新設定一律使用正確的 `NEMOTRON` 拼法。

### 2.2 不把實際 key 寫進教材或映像

即使活動 key 會在課後失效，也不要把實值放進這份文件、Notebook、Docker layer、Git 或 shell history。本活動例外以專用腳本放入 VM filesystem 的 root-only 活動憑證包；學生具 sudo 權限仍可擷取，課後必須撤銷。

單台人工驗收，用隱藏輸入的腳本。教材路徑、兩個 endpoint 與五個部署 model ID 已固定；腳本只詢問三個 RAP key 與選填的 NIM key：

```bash
sudo -u ubuntu /home/ubuntu/cudaq-course-kit/configure-course-env.sh
```

NIM key 留空時，OpenCode 設定不會顯示 NIM provider。Jupyter token 由腳本自動產生。

它會建立：

```text
/home/ubuntu/.config/nchc-cudaq-course/course.env
```

確認權限，不顯示內容：

```bash
stat -c '%a %U:%G %n' /home/ubuntu/.config/nchc-cudaq-course/course.env
```

必須是 `600 ubuntu:ubuntu`。

### 2.3 用 key 查詢當天可用 model ID

以下只輸出 model ID，不輸出 key。每把 RAP key 可各做一次：

```bash
read -r -s -p 'RAP API key: ' RAP_TEST_KEY
echo
curl -fsS \
  -H "Authorization: Bearer $RAP_TEST_KEY" \
  -H "x-api-key: $RAP_TEST_KEY" \
  https://portal.genai.nchc.org.tw/api/v1/models |
  jq -r '.data[].id'
unset RAP_TEST_KEY
```

若回傳 401/403，記錄 HTTP status、時間、endpoint、response body 中不含敏感值的錯誤訊息與 request ID；不要把 request header 或 key 交給 API 同仁。

### 2.4 學生之後如何申請自己的 key

- NCHC RAP：學生登入 [NCHC RAP Portal](https://portal.genai.nchc.org.tw/)，依國網提供的帳號、計畫或課程 entitlement 申請 API 使用權。公開頁面目前沒有足以取代國網內部流程的完整申請說明，因此課堂講義應放承辦人提供的正式申請連結或聯絡方式。
- NVIDIA hosted NIM：登入 NVIDIA API Catalog，選模型後使用 `Get API Key`／`Generate Key`；NVIDIA 官方文件要求妥善保存且不要分享 key。
- 取得 key 後，學生執行 `configure-course-env.sh`，輸入對應 key；endpoint 與 model ID 已固定。不要將 `export ...=實值` 貼進 Notebook 或聊天。

## 3. 母 VM 基礎環境人工複驗

先開一個記錄檔，但不要把 `course.env` 內容寫入 log：

```bash
cd /home/ubuntu/cudaq-course-kit
./verify-environment.sh 2>&1 | tee /home/ubuntu/manual-acceptance.log
```

應以這行結束：

```text
All required checks passed.
```

記錄正式 image identity：

```bash
docker image inspect nchc/cudaq-agentic-coding:2026-07-17 \
  --format 'ID={{.Id}} Created={{.Created}} RepoDigests={{json .RepoDigests}}'
```

本次已驗證 image ID：

```text
sha256:25d1fd61b75a47bd3ec338a577d3c9d872eae40397f29fa39834d9091882920b
```

啟動 JupyterLab：

```bash
cd /home/ubuntu/cudaq-course-kit
./start-lab.sh
sleep 20
docker compose \
  --project-directory /home/ubuntu/cudaq-course-kit \
  --env-file /home/ubuntu/.config/nchc-cudaq-course/course.env \
  -f /home/ubuntu/cudaq-course-kit/compose.yaml ps
```

狀態必須為 `healthy`。檢查主機 listen address：

```bash
ss -ltnp | grep ':8888'
```

PASS 應看到 `0.0.0.0:8888`；若仍只有 `127.0.0.1:8888`，不得封裝。NCHC security group／網路 ACL 應只允許課程來源連 TCP 8888，SSH 則使用 TCP 3322。

檢查 container 內版本、GPU 與 Python 相依性：

```bash
docker compose \
  --project-directory /home/ubuntu/cudaq-course-kit \
  --env-file /home/ubuntu/.config/nchc-cudaq-course/course.env \
  -f /home/ubuntu/cudaq-course-kit/compose.yaml \
  exec -T lab bash -lc '
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    jupyter lab --version
    python -c "import importlib.metadata as m; print(m.version('jupyter-ai'))"
    opencode --version
    python -c "import qiskit; print(qiskit.__version__)"
    python -m pip check
  '
```

預期：H200、JupyterLab 4.6.1、Jupyter AI 3.0.1、OpenCode 1.18.1、Qiskit 2.5.0，以及 `No broken requirements found.`。

先從自己的筆電直接開啟 `http://VM_IP:8888/lab`。若 TCP 8888 因網路限制無法直連，再建立備援 tunnel：

```bash
ssh -p 3322 -i bootcamp0807.pem -N \
  -L 8888:127.0.0.1:8888 ubuntu@VM_IP
```

使用直連時開 `http://VM_IP:8888/lab`；使用備援 tunnel 時開 `http://localhost:8888/lab`。依序執行 `_intro_cudaq.ipynb` 與 `00_notebook.ipynb`。完成後停止 Lab，將 GPU 留給隔離的 OpenCode 測試：

```bash
/home/ubuntu/cudaq-course-kit/stop-lab.sh
```

## 4. 建立不可讀 solutions 的 OpenCode 測試區

這段刻意不把 README、`solutions/` 或整個 repo 掛進 Agent container。Agent 只能看到允許的教材與前一步產物。

```bash
MANUAL_ROOT=/home/ubuntu/opencode-manual-validation
SOURCE=/opt/nchc-cudaq-course/source
KIT=/home/ubuntu/cudaq-course-kit
ENV_FILE=/home/ubuntu/.config/nchc-cudaq-course/course.env
COURSE_IMAGE=nchc/cudaq-agentic-coding:2026-07-17

mkdir -p "$MANUAL_ROOT/run" \
  "$MANUAL_ROOT/opencode-data" \
  "$MANUAL_ROOT/opencode-cache"

cp "$SOURCE/_intro_cudaq.ipynb" "$MANUAL_ROOT/run/"
cp "$SOURCE/00_notebook.ipynb" "$MANUAL_ROOT/run/"
cp "$SOURCE/cudaq-doc.md" "$MANUAL_ROOT/run/"
cp "$SOURCE/AGENTS.md" "$MANUAL_ROOT/run/"
cp "$SOURCE/helpers.py" "$MANUAL_ROOT/run/"
cp -a "$SOURCE/data" "$MANUAL_ROOT/run/"

"$KIT/render-opencode-config.sh" \
  "$ENV_FILE" "$MANUAL_ROOT/run/opencode.json"

test ! -e "$MANUAL_ROOT/run/README.md"
test ! -e "$MANUAL_ROOT/run/solutions"
```

套用與正式 `start-lab.sh` 相同的 UID/ACL 邏輯：

```bash
cudaq_user=$(docker run --rm --entrypoint sh "$COURSE_IMAGE" -c \
  'printf "%s:%s" "$(id -u cudaq)" "$(id -g cudaq)"')
course_owner=$(stat -c '%u:%g' "$MANUAL_ROOT/run")
CUDAQ_UID=${cudaq_user%%:*}
COURSE_OWNER_UID=${course_owner%%:*}

docker run --rm --user 0:0 \
  --env CUDAQ_UID="$CUDAQ_UID" \
  --env COURSE_OWNER_UID="$COURSE_OWNER_UID" \
  --volume "$MANUAL_ROOT:/validation" \
  --entrypoint bash "$COURSE_IMAGE" -lc '
    set -euo pipefail
    setfacl -R -m \
      "u:$COURSE_OWNER_UID:rwX,u:$CUDAQ_UID:rwX,m::rwX" /validation
    find /validation -type d -exec setfacl \
      -m "d:u:$COURSE_OWNER_UID:rwx,d:u:$CUDAQ_UID:rwx,d:m::rwx" {} +
    cudaq_group=$(id -g cudaq)
    chown -R "$CUDAQ_UID:$cudaq_group" \
      /validation/opencode-data /validation/opencode-cache
  '
```

再次確認 Agent 看不到答案：

```bash
find "$MANUAL_ROOT/run" -maxdepth 2 -type f -printf '%P\n' | sort
```

輸出中不得有 `README.md` 或 `solutions/`。OpenCode 要求 web search、web fetch、`curl`、`git clone` 或讀取目前目錄以外路徑時，一律拒絕；只批准測試區內的 read/edit 與必要的 Notebook 執行。

## 5. 啟動 Nemotron 並逐段貼 prompt

先載入 model ID，再啟動隔離 container：

```bash
set -a
source /home/ubuntu/.config/nchc-cudaq-course/course.env
set +a

docker run --rm -it --gpus all --network=host \
  --env-file /home/ubuntu/.config/nchc-cudaq-course/course.env \
  --env HOME=/home/cudaq \
  --volume /home/ubuntu/opencode-manual-validation/run:/workspace/validation \
  --volume /home/ubuntu/opencode-manual-validation/opencode-data:/home/cudaq/.local/share/opencode \
  --volume /home/ubuntu/opencode-manual-validation/opencode-cache:/home/cudaq/.cache/opencode \
  --workdir /workspace/validation \
  --entrypoint opencode \
  nchc/cudaq-agentic-coding:2026-07-17 \
  --model "rap-nemotron/$RAP_NEMOTRON_3_SUPER_MODEL"
```

不要一次貼完所有步驟。每一步完成並驗證後，才貼下一段。

### 5.1 唯讀暖身

```text
Read @cudaq-doc.md and @AGENTS.md, then give me a short summary of what
each one covers. Just the wrap-up. No code or files yet. Stay inside the current
working directory. Do not use web search, GitHub, git, curl, or any solutions.
```

PASS：只摘要兩個檔案，沒有建立檔案、沒有網路或 repo 搜尋。

### 5.2 GHZ 工具鏈測試

```text
Write a simple script ghz.py using CUDA-Q to prepare a 20-qubit GHZ state on the GPU
("nvidia" target), sample 1000 shots, and print the counts. Refer to @cudaq-doc.md if needed,
then run it and show me the output. Stay inside the current working directory and do not
use web search, GitHub, git, curl, README.md, or any solutions.
```

PASS：`ghz.py` 執行成功；結果主要是全 0 與全 1，總 shots 為 1000。

### 5.3 Step 1

```text
Build and execute 01_notebook.ipynb here, applying AGENTS.md conventions, from
00_notebook.ipynb (read it): reuse its 16-qubit region problem and p=1 kernel exactly
(A=4.0/B=1.0, only 1-/2-qubit gates), target "nvidia" single-GPU backend, and FP32 precision.

Structure:
1. FIRST, BEFORE import cudaq:
   os.environ["CUDAQ_FUSION_MAX_QUBITS"] = "3"
   os.environ["CUDAQ_FUSION_NUM_HOST_THREADS"] = "4"
   Rebuild problem.
2. The QAOA kernel and energy(params) from 00.
3. Optimize: cudaq.optimizers.NelderMead, num_iterations=50, initial_parameters=[0.1,0.1],
   timed. Each energy() call prints its GPU milliseconds alongside the energy; print best
   energy, total time. Nelder-Mead lands far below 00's COBYLA on the same 50-evaluation budget.
4. Sample 20,000 shots after optimization; summarize_samples; perform a single assert
   helpers.is_valid(best) check. Report a valid rate well above 00's.
5. Draw coloring (region level, unit_labels on). Compare best energy and valid rate vs 00.

Do not use web search, GitHub, git, curl, README.md, or solutions. Keep all work in the
current directory. Use at most six code cells. Execute from a clean kernel with a 600-second
timeout, save outputs, and report any real error instead of fabricating a result.
```

### 5.4 Step 2

```text
Build and execute 02_notebook.ipynb here from notebooks 00/01 (read them; keep Nelder-Mead).

Problem: 9 zones * 4 colors -> 36 qubits, using the zones listed in helpers.ZONE_ORDER and
the adjacency specified in load_map()["zone_edges"]. A=8.0/B=1.0. Target "tensornet"
(tensor-network) backend, and FP32 precision. Markdown: why tensornet — 36 qubits =
2^36x8 = 550 GB, beyond most single GPU.

Structure:
1. QUBO via helpers; print term counts.
2. ONE kernel, xy_kernel (not plain rx), 1-/2-qubit gates only:
   - W-state per zone: x(q[base]); for k in 0..2:
     ry.ctrl(W_ANGLES[k], q[base+k], q[base+k+1]) then
     x.ctrl(q[base+k+1], q[base+k]);
     W_ANGLES = [2*math.acos(1/math.sqrt(4)), 2*math.acos(1/math.sqrt(3)),
     2*math.acos(1/math.sqrt(2))]
     (leverage the math library — no long decimal literals).
   - 00's cost layer: rz for Z terms, cx-rz-cx for ZZ.
   - XY ring mixer over ring pairs (0,1),(1,2),(2,3),(3,0):
     XX: h,h / cx / rz(2*beta) / cx / h,h
     YY: rx(+math.pi/2),rx(+math.pi/2) / cx / rz(2*beta) / cx /
         rx(-math.pi/2),rx(-math.pi/2).
3. Optimize: Nelder-Mead, max_iterations=5, start [0.1,0.1], timed; print best energy,
   angles, seconds/evaluation (total/5). No energy calls outside these 5. Do not add a
   diagnostic energy([0,0]) call.
4. SHOTS=2000; sample best angles ONLY after optimization; helpers.summarize_samples;
   THE assert: helpers.is_valid(best).
5. Map: counties by zone PALETTE color, zone graph
   (helpers.group_positions(taiwan,"zone")), color legend; per-eval + per-shot seconds; theta*.

Do not use web search, GitHub, git, curl, README.md, or solutions. Keep all work in the
current directory. Use at most six code cells. Execute from a clean kernel with a 600-second
timeout and save all outputs.
```

### 5.5 Step 3

```text
Build and execute 03_notebook.ipynb here from 02_notebook.ipynb (read it; reuse
problem/xy_kernel exactly — A=8.0/B=1.0, Nelder-Mead, tensornet fp32).

Structure:
1. FIRST cell, BEFORE import cudaq:
   os.environ["CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE"] = "TRUE"
   os.environ["CUDAQ_TENSORNET_NUM_HYPER_SAMPLES"] = "32"
   Rebuild problem.
2. Same xy_kernel and energy(params).
3. Retrieve elapsed time from 02, or rerun its optimization if no timing result is available,
   SAME size: start [0.1,0.1], max_iterations=5; print best energy, angles,
   elapsed vs 02's total (hardcoded, provenance) — print speedup.
4. SHOTS=2000; sample best angles (post-optimization); summarize_samples; THE assert:
   helpers.is_valid(best).
5. Map (as 02) + table: seconds/evaluation and optimize seconds (02 vs this);
   speedup+verdict; theta* AND exact energy (04 needs them).

Do not use web search, GitHub, git, curl, README.md, or solutions. Keep all work in the
current directory. Use at most six code cells. Execute from a clean kernel with a 600-second
timeout and save all outputs.
```

### 5.6 Step 4

```text
Build and execute 04_notebook.ipynb here from 03_notebook.ipynb (read it; same problem and
xy_kernel, A=8.0/B=1.0, Nelder-Mead).

Structure:
1. FIRST cell, BEFORE import cudaq:
   os.environ["CUDAQ_MPS_MAX_BOND"]="16"
   os.environ["CUDAQ_MPS_ABS_CUTOFF"]="1e-4"
   cudaq.set_target("tensornet-mps", option="fp32"); rebuild the zone problem.
   Markdown, ONE honest disclosure: the MPS energy readout is untrustworthy at this
   compression; judge by sampled colorings, not the readout.
2. Same xy_kernel and energy(params).
3. Nelder-Mead from [0.1,0.1] FRESH (never 03's final angles).
   max_iterations=5; timed; print seconds/evaluation and best readout.
4. SHOTS=2000; sample ONCE at optimized angles (post-optimization), timed;
   summarize_samples; THE assert: helpers.is_valid(best).
5. Map (counties by zone) + timing table, ONLY time rows: seconds/evaluation
   (03's vs chi=16's), ratio; verdict: sampled quality matches the exact backend at
   at-or-below cost, memory far smaller, readout untrustworthy — answers verified exactly
   after sampling.

Do not use web search, GitHub, git, curl, README.md, or solutions. Keep all work in the
current directory. Use at most six code cells. Execute from a clean kernel with a 600-second
timeout and save all outputs.
```

## 6. 每一步的獨立驗證

另開第二個 SSH session。每完成一份 notebook，先做 clean-kernel 重跑：

```bash
NOTEBOOK=01_notebook.ipynb

docker run --rm --gpus all --network=host \
  --volume /home/ubuntu/opencode-manual-validation/run:/workspace/validation \
  --workdir /workspace/validation \
  --entrypoint jupyter \
  nchc/cudaq-agentic-coding:2026-07-17 \
  nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 "$NOTEBOOK"
```

依序將 `NOTEBOOK` 改為 02、03、04。然後執行新增的驗證器：

```bash
python3 /home/ubuntu/cudaq-course-kit/validate-agent-notebooks.py \
  /home/ubuntu/opencode-manual-validation/run \
  01_notebook.ipynb
```

四份完成後一次總驗：

```bash
python3 /home/ubuntu/cudaq-course-kit/validate-agent-notebooks.py \
  /home/ubuntu/opencode-manual-validation/run
```

PASS 標準：

- JSON 可讀、kernel 是 `python3`、1–6 個 code cells。
- 每個 code cell 都有 execution count、可編譯且無 error output。
- 全 notebook 恰好一個 Python `assert`，且執行通過。
- 指定 backend、FP32、shots、環境變數與 Step-specific 結構存在。
- 01 的 best energy 與 valid rate 明顯優於 00，地圖有效。
- 02–04 至少抽到有效 coloring；03 顯示 path reuse 的 timing 比較；04 明說 energy readout 不可信、以抽樣後 exact validity 為準。
- 02 若出現 direct `energy([...])`，驗證器會 WARN；應交回 Agent 刪除額外診斷呼叫，再重跑。

要在 JupyterLab 肉眼檢查地圖，可在 Agent 完成後才將結果複製回學生 repo；此時 Agent 測試已結束，不會因此看到 `solutions/`：

```bash
mkdir -p /home/ubuntu/cudaq-agentic-coding/manual-validated
cp /home/ubuntu/opencode-manual-validation/run/0[1-4]_notebook.ipynb \
  /home/ubuntu/cudaq-agentic-coding/manual-validated/
```

再啟動 Lab，逐份確認圖、表格與文字敘述。

## 7. OpenCode 錯誤時的恢復方式

Tool-call JSON 壞掉時，只貼這一段：

```text
上一個 tool call 的 JSON 不合法。請不要重做整份 notebook，也不要送出巨大的
單一 tool call。先檢查目前檔案狀態，再用合法 JSON 重試剛才失敗的最小一步；
一次只讀、改或執行一個小範圍。不得使用 web、GitHub、git、curl 或 solutions。
```

若最後文字摘要卡住：

1. 在第二個 SSH session 看 notebook 檔案是否已寫入。
2. 跑 clean-kernel `nbconvert`。
3. 跑 `validate-agent-notebooks.py`。
4. 兩者 PASS 就保存 notebook 與 log；不因缺少聊天結語重做 GPU 運算。

提供 API 同仁的 debug 資訊只包含：

- UTC 時間、provider/model ID、endpoint（不含 query secret）。
- HTTP status、response body 中的錯誤 type/message/request ID。
- OpenCode 版本、重試次數、tool 名稱與「JSON parse failed」摘要。
- 不附 request headers、`course.env`、OpenCode database 或任何 key。

## 8. Fallback 模型驗收

Nemotron 完整 Step 1–4 通過後，Gemma fallback 至少做「唯讀暖身 + GHZ」。在相同隔離目錄，可改用新 session：

```bash
set -a
source /home/ubuntu/.config/nchc-cudaq-course/course.env
set +a

docker run --rm -it --gpus all --network=host \
  --env-file /home/ubuntu/.config/nchc-cudaq-course/course.env \
  --env HOME=/home/cudaq \
  --volume /home/ubuntu/opencode-manual-validation/run:/workspace/validation \
  --volume /home/ubuntu/opencode-manual-validation/opencode-data:/home/cudaq/.local/share/opencode \
  --volume /home/ubuntu/opencode-manual-validation/opencode-cache:/home/cudaq/.cache/opencode \
  --workdir /workspace/validation \
  --entrypoint opencode \
  nchc/cudaq-agentic-coding:2026-07-17 \
  --model "rap-gemma-31b/$RAP_GEMMA_31B_MODEL"
```

再以 `rap-gemma-26b/$RAP_GEMMA_26B_MODEL` 重複。不要覆蓋 Nemotron 已驗證的 `ghz.py`；請分別要求 `ghz_gemma31.py`、`ghz_gemma26.py`。若時間足夠，再讓 fallback 做 Step 1，輸出另命名，避免混淆正式結果。

建議 fallback 順序：

1. Nemotron 3 Super（預設）：完整 Step 1–4。
2. Nemotron 3 Ultra（共用 RAP key）：暖身、GHZ，選配 Step 1。
3. Gemma 31B：暖身、GHZ，選配 Step 1。
4. Gemma 26B：暖身、GHZ，作為最後備援。
5. NVIDIA hosted NIM：只有在當天模型與 key 都已核准時測試。
6. OpenCode Zen Nemotron：先確認 `opencode models opencode | grep -Fx opencode/nemotron-3-ultra-free`；若講師有個人 Zen API key，再於 TUI 使用 `/connect` 連接 `OpenCode Zen`，選擇 `opencode/nemotron-3-ultra-free` 做唯讀暖身與 GHZ。

Zen 的 `Nemotron 3 Ultra Free` 是限時免費 trial，並非免登入：仍需 OpenCode Zen
帳號與個人 API key，也可能有 rate limit。不要將 Zen key 放入活動憑證包、
`course.env`、Notebook、chat、Git 或 VM image。官方聲明 free endpoint 的試用
資料會用於安全管理與 NVIDIA 產品／服務改善，因此不得提交個資、機密資料或課程
憑證。封裝前執行 `reset-manual-validation.sh`，確認 Zen 認證也隨 OpenCode runtime
移出作用中環境。

## 9. 建立活動憑證包並清理映像

先把 `manual-acceptance.log`、四份驗證結果與需要的截圖下載到管理者電腦。完成最後一次 Super 手動測試後，執行：

```bash
sudo /home/ubuntu/cudaq-course-kit/finalize-vm-image.sh
```

腳本成功後不要再啟動 Lab，直接交付國網關機封裝。以下保留詳細步驟供人工稽核；若已使用上述腳本，不需重複執行。接著停止服務：

```bash
/home/ubuntu/cudaq-course-kit/stop-lab.sh
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

不得有課程或 OpenCode 測試 container 仍在執行。

先從已驗證的 `course.env` 建立映像內唯一允許保留的活動憑證包，且不顯示內容：

```bash
sudo /home/ubuntu/cudaq-course-kit/prepare-activity-image-credentials.sh
sudo stat -c '%a %U:%G %n' \
  /opt/nchc-cudaq-course/activity/activity-keys.env
```

必須是 `600 root:root`。接著處理主 `course.env` 與帶時間戳的備份；先只列檔名：

```bash
sudo find /home/ubuntu/.config/nchc-cudaq-course \
  -maxdepth 1 -type f -name 'course.env*' -printf '%m %u:%g %p\n'
```

確認範圍只有課程 env 後，才刪除所有主檔與備份：

```bash
sudo find /home/ubuntu/.config/nchc-cudaq-course \
  -maxdepth 1 -type f -name 'course.env*' -delete
```

清除只屬於驗收的 runtime、OpenCode session、Agent 產物與 log：

```bash
sudo find /home/ubuntu/cudaq-course-kit/.runtime -mindepth 1 -delete
sudo find /home/ubuntu/opencode-manual-validation -mindepth 1 -delete
sudo rmdir /home/ubuntu/opencode-manual-validation
sudo rm -f /home/ubuntu/manual-acceptance.log
```

若曾在 shell 直接 export 過實際 key，應一併清理該帳號的 shell history；較好的做法是整個流程都只用隱藏輸入或 provisioning secret store。

做不顯示匹配內容的 secret audit：

```bash
if sudo rg -l --hidden --glob '!*.ipynb' \
  '(sk-[A-Za-z0-9_-]{12,}|nvapi-[A-Za-z0-9_-]{12,})' \
  /opt/nchc-cudaq-course/source \
  /opt/nchc-cudaq-course/kit \
  /home/ubuntu/cudaq-course-kit \
  /home/ubuntu/cudaq-agentic-coding >/dev/null; then
  echo 'FAIL: possible credential remains'
else
  echo 'PASS: no unexpected credential pattern outside the authorized activity bundle'
fi

if docker history --no-trunc nchc/cudaq-agentic-coding:2026-07-17 |
  rg '(sk-[A-Za-z0-9_-]{12,}|nvapi-[A-Za-z0-9_-]{12,})' >/dev/null; then
  echo 'FAIL: possible credential in Docker history'
else
  echo 'PASS: no credential pattern in Docker history'
fi

test ! -e /home/ubuntu/.config/nchc-cudaq-course/course.env
test -z "$(docker ps -q)"
```

映像中要保留：

- `/opt/nchc-cudaq-course/source` 乾淨教材快照。
- `/opt/nchc-cudaq-course/kit` 與 `/home/ubuntu/cudaq-course-kit`。
- `/opt/nchc-cudaq-course/activity/activity-keys.env`（刻意保留，必須為 `600 root:root`）。
- `/home/ubuntu/cudaq-agentic-coding` 的乾淨學生副本。
- 已驗證 Docker image 與固定 tag。

映像中不要保留：

- 學生家目錄中的 `course.env*`、舊 Jupyter token、OpenCode data/cache/database；root-only 活動憑證包是唯一例外。
- 人工驗收 notebooks、chat/session、API debug response。
- 個人 SSH key、shell history 中的 key、臨時下載或 log。

依 NCHC 映像平台流程關機封裝。只有當平台要求 clone 後重新產生 identity 時，才在最後依其 runbook 執行 `cloud-init clean --logs --machine-id`；不得自行假設平台行為。40 台 VM 必須各有唯一 hostname、machine-id 與 SSH host key。

## 10. 部署 40 台後的必要步驟

1. 從含 root-only 活動憑證包的母映像建立 40 台 VM。
2. 每台 VM 建立唯一 hostname、machine-id 與 SSH host key。
3. 學生登入後只執行 `~/cudaq-course-kit/activate-student-course.sh`。
4. 啟用腳本為該 VM 產生唯一 Jupyter token、建立 mode 600 的 `course.env`、驗證並啟動 Lab。
5. 確認每台 Lab healthy、主機發布 `0.0.0.0:8888`，且從允許的課程網路可直接開啟 `http://VM_IP:8888/lab`。
6. 抽測 OpenCode `/models`、唯讀暖身與 GHZ；至少一台做完整 Step 1。
7. 記錄 VM IP、image ID、GPU、verify 結果、Jupyter health 與 OpenCode provider，不記錄 key。
8. 上課前向 NCHC/NVIDIA 確認共用 key 的同時連線、token/minute、request/minute 與每日 quota 足以支援 40 台。
9. 明確告知學生：共用 key 只供本次活動，日後須執行 `configure-course-env.sh` 改用個人 key。
10. 活動結束後立即撤銷共用 RAP/NIM key，並回收或清除學生 VM。

部署後抽查：

```bash
cd /home/ubuntu/cudaq-course-kit
./activate-student-course.sh
stat -c '%a %U:%G %n' /home/ubuntu/.config/nchc-cudaq-course/course.env
ss -ltnp | grep ':8888'
./stop-lab.sh
```

完整環境建置與母映像說明另見 `instructor-image-guide.zh-TW.md`；學生上課流程另見 `student-guide.zh-TW.md`。
