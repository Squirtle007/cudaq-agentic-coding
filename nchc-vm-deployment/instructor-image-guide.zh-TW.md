# 教師／國網維運手冊：建立 H200 CUDA-Q Agentic Coding VM Image

本文件用來建立國網課堂母 VM。學生 VM 由這個已驗收環境複製部署；學生不負責安裝、build 或 clone。

預計由國網準備約 40 台學生 VM。每位學生取得一個指定 IP，並以活動用 `bootcamp0807.pem` 登入，例如：

```bash
ssh -p 3322 -i bootcamp0807.pem ubuntu@140.110.109.XXX
```

## 最終 VM 內容

```text
/opt/nchc-cudaq-course/
├── source/    # root 擁有的乾淨教材快照
├── kit/       # root 擁有的啟動、驗證與環境工具
└── activity/  # root-only 活動憑證包；課後撤銷

/home/ubuntu/
├── cudaq-agentic-coding/   # 學生可寫工作副本
└── cudaq-course-kit/       # 學生操作工具副本

Docker image:
nchc/cudaq-agentic-coding:2026-07-17
```

本活動經講師明確接受風險後，將共用 RAP/NIM key 放入 VM image 的 root-only 活動憑證包：

```text
/opt/nchc-cudaq-course/activity/activity-keys.env  # 600 root:root
```

學生執行一次 `activate-student-course.sh`，才會建立自己的 `course.env`、獨立 Jupyter token，接著自動驗證並啟動 Lab。具 sudo 權限的學生仍能擷取共用 key，因此活動結束必須立即撤銷。

## 0. 準備材料

將下列檔案放到母 VM 的暫存目錄：

- `cudaq-agentic-coding-main.zip`
- 本 VM kit 資料夾

範例：

```bash
scp -P 3322 cudaq-agentic-coding-main.zip ubuntu@VM_IP:/tmp/
scp -P 3322 -r nchc-cudaq-agentic-coding-vm-kit ubuntu@VM_IP:/tmp/
ssh -p 3322 ubuntu@VM_IP
```

讓腳本可執行：

```bash
cd /tmp/nchc-cudaq-agentic-coding-vm-kit
chmod +x ./*.sh
```

## 1. 驗證母 VM 的 GPU

```bash
cat /etc/os-release
uname -m
df -h /
free -h
nvidia-smi
```

驗收基準：

- Ubuntu 22.04 或 24.04
- `x86_64`
- NVIDIA H200
- Driver `595.71.05` 或經實測相容的新版本
- `nvidia-smi` 顯示 CUDA capability 13.2 不代表主機必須另外安裝 CUDA Toolkit；本課程 CUDA runtime 位於 container

## 2. 安裝 Docker 與 NVIDIA Container Toolkit

新母 VM 可執行：

```bash
cd /tmp/nchc-cudaq-agentic-coding-vm-kit
./install-host-docker.sh
```

腳本會依官方 apt repository 安裝：

- Docker Engine、Buildx、Compose plugin
- NVIDIA Container Toolkit
- `nvidia-ctk runtime configure --runtime=docker`

完成後登出再登入，使 docker group生效：

```bash
exit
ssh -p 3322 ubuntu@VM_IP
```

驗證：

```bash
docker version
docker compose version
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.15.0
```

## 3. 安裝乾淨教材快照與 VM kit

```bash
cd /tmp/nchc-cudaq-agentic-coding-vm-kit
./prepare-course-source.sh /tmp/cudaq-agentic-coding-main.zip
```

這個步驟也會套用 `course-repo.patch`：修正暖身 Notebook 檔名、補上 NCHC 預建 VM 說明，並將 Qiskit 加入開源版 requirements。

確認：

```bash
find /opt/nchc-cudaq-course/source -maxdepth 2 -type f | sort
find /opt/nchc-cudaq-course/kit -maxdepth 1 -type f | sort
```

壓縮檔實際包含：

- `_intro_cudaq.ipynb`
- `_intro_Ising_Calibration.ipynb`
- `00_notebook.ipynb`
- `cudaq-doc.md`、`AGENTS.md`、`helpers.py`
- `data/taiwan_map_xy.json`
- `my_code.py`
- `NCHC_SOURCE_REVISION`（匯入時產生，記錄 upstream commit 與 archive SHA-256）

公開 `main` 不含 `solutions/`；答案分支只供隔離的講師唯讀比較，不得掛入學生或 Agent 工作區。

## 4. 建立固定版本課程 container

```bash
cd /opt/nchc-cudaq-course/kit
docker build \
  --platform linux/amd64 \
  --build-arg OPENCODE_VERSION=1.18.1 \
  -t nchc/cudaq-agentic-coding:2026-07-17 \
  .
```

建置內容：

- CUDA-Q `cu13-0.15.0` stable image，固定 amd64 manifest digest
- JupyterLab `4.6.1`
- Jupyter AI `3.0.1`
- OpenCode `1.18.1`
- Qiskit `2.5.0`
- 原課程需要的 NumPy、SciPy、Matplotlib、NetworkX

記錄 image identity 與實際 Python lock：

```bash
docker image inspect nchc/cudaq-agentic-coding:2026-07-17 \
  --format 'ID={{.Id}} Created={{.Created}}'

docker run --rm --entrypoint cat \
  nchc/cudaq-agentic-coding:2026-07-17 \
  /opt/nchc-cudaq-course/resolved-requirements.txt
```

將 image ID 與 `resolved-requirements.txt` 保存到課程驗收紀錄。部署以已驗證 image 為準，不在上課當天重新解析 Python 套件。

## 5. 建立學生工作副本

如果母 VM 會直接以 `ubuntu` 帳號封裝：

```bash
cp -a /opt/nchc-cudaq-course/source /home/ubuntu/cudaq-agentic-coding
cp -a /opt/nchc-cudaq-course/kit /home/ubuntu/cudaq-course-kit
sudo chown -R ubuntu:ubuntu \
  /home/ubuntu/cudaq-agentic-coding \
  /home/ubuntu/cudaq-course-kit
chmod +x /home/ubuntu/cudaq-course-kit/*.sh
```

若同一 VM 有多位帳號，請在 provisioning 階段為每位使用者各自建立副本；不要多人共用同一個可寫 repo 或同一組 Compose project name。

## 6. 建立 VM image 內的活動憑證包

### 已接受的風險

- 共用 key 不寫入 Dockerfile、container layer、Git、Notebook 或教材 source。
- Key 會刻意放入 VM filesystem 的 root-only 檔案，並隨母映像複製到 40 台 VM。
- 學生帳號具有 sudo 能力，因此 root-only 無法阻止刻意擷取，只能避免日常誤讀。
- 活動結束必須立即撤銷所有 RAP/NIM key。

先用固定 endpoint/model ID 的互動腳本建立驗收用 `course.env`；畫面只詢問三個 RAP key 與選填 NIM key：

```bash
sudo -u ubuntu /home/ubuntu/cudaq-course-kit/configure-course-env.sh
sudo -iu ubuntu /home/ubuntu/cudaq-course-kit/verify-environment.sh
```

驗收通過後，將其中的 API key 複製成不含 Jupyter token 的 root-only 活動憑證包：

```bash
sudo /home/ubuntu/cudaq-course-kit/prepare-activity-image-credentials.sh
sudo stat -c '%a %U:%G %n' \
  /opt/nchc-cudaq-course/activity/activity-keys.env
```

必須顯示：

```text
600 root:root /opt/nchc-cudaq-course/activity/activity-keys.env
```

腳本不輸出 key。封裝前刪除講師個人的 `course.env*`、runtime、OpenCode session 與舊 Jupyter token，但刻意保留上述 root-only 活動憑證包。

學生登入 clone 後只執行：

```bash
cd ~/cudaq-course-kit
./activate-student-course.sh
```

啟用腳本會顯示活動 key 到期/BYOK 通知、建立 mode 600 的個人 `course.env`、產生該 VM 專屬 Jupyter token、執行 `verify-environment.sh` 並啟動 JupyterLab。

若臨時需要兩位學員共用同一台 VM，不必重建舊 image；只要用新版
`activate-student-course.sh` 覆蓋學生 home 內的舊檔即可。第一個 Lab 正在執行時，第二次執行
相同啟用指令會建立 port 8889、第二份 repo/runtime/Compose project/token；既有第二份環境則
保留 token 並重新顯示完整連結。完整路徑、操作與 GPU 共用限制請見
[兩位學員共用一台 VM](two-students-one-vm.zh-TW.md)。NCHC security group／ACL 也必須允許
課程來源網段連入 TCP 8889。

## 7. 執行環境驗收

```bash
sudo -iu ubuntu
cd ~/cudaq-course-kit
./verify-environment.sh
```

這會檢查：

- H200、Driver、Docker、Compose
- 預載 course image
- 教材必要檔案
- container 內的 CUDA-Q、`nvq++`、JupyterLab、Jupyter AI、OpenCode、Qiskit
- `qpp-cpu` 與 `nvidia` GHZ smoke test

再啟動服務：

```bash
./start-lab.sh
docker compose \
  --project-directory ~/cudaq-course-kit \
  --env-file ~/.config/nchc-cudaq-course/course.env \
  -f ~/cudaq-course-kit/compose.yaml \
  ps
```

確認 Jupyter 對外發布在 VM TCP 8888：

```bash
docker port nchc-cudaq-course-lab-1
ss -ltnp | grep ':8888'
```

預期看到 `0.0.0.0:8888`。另由 NCHC security group／網路 ACL 將 TCP 8888 限制在課程允許的來源網段，並開放 SSH TCP 3322。從筆電直接測試 `http://VM_IP:8888/lab`；Jupyter token 仍為必要驗證。

## 8. 完整執行 Notebook 驗收

在學生工作副本執行，不修改 `/opt` 的乾淨快照：

```bash
docker compose \
  --project-directory ~/cudaq-course-kit \
  --env-file ~/.config/nchc-cudaq-course/course.env \
  -f ~/cudaq-course-kit/compose.yaml \
  exec lab bash
```

Container 內：

```bash
cd /workspace/cudaq-agentic-coding

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=550 _intro_cudaq.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=550 _intro_Ising_Calibration.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=550 00_notebook.ipynb

python my_code.py
python -m pip check
```

公開 `main` 只預載上述基礎／特別單元與 baseline；01–04 必須依人工驗收 runbook 由 OpenCode 在隔離目錄逐本建立。不要平行執行 GPU Notebook；Step 2 的 36-qubit tensornet 最接近 timeout，必須在最終 H200 image 實測。

OpenCode 建立的 01–04 每本驗收：

- 零 error output
- validity assert 通過
- 產生最終地圖
- 不超過教材預算
- 若結果或時間不同，回報實測，不硬改答案

## 9. 驗證 OpenCode 與 Jupyter AI

在 container Terminal：

```bash
cd /workspace/cudaq-agentic-coding
opencode
```

驗證：

1. `/models` 顯示五個部署的 RAP/NIM 模型，且預設為 Nemotron 3 Super。
2. `opencode models opencode | grep -Fx opencode/nemotron-3-ultra-free` 確認 OpenCode 1.18.1 能發現 Zen fallback；這一步不需要也不得封裝 Zen key。
3. 若講師有個人 Zen API key，以 `/connect` 選擇 `OpenCode Zen` 後，另開 session 用 `opencode/nemotron-3-ultra-free` 做唯讀暖身與 GHZ。Free 是限時 trial，仍需 Zen 帳號/API key。
4. OpenCode 能讀 `cudaq-doc.md` 與 `AGENTS.md`。
5. 建立 20-qubit GHZ 程式，經批准後能在 `nvidia` target執行。
6. Jupyter AI Chat 的 `@` 清單出現 `OpenCode`。
7. `@OpenCode` 能讀 `00_notebook.ipynb`。
8. 修改、shell 與 Notebook/MCP 執行都會要求批准。
9. OpenCode sharing 維持 disabled。

模型矩陣：

| Backend | 完整 model ID | 憑證 |
|---|---|---|
| NCHC RAP Nemotron Super（預設） | `rap-nemotron/NVIDIA-Nemotron-3-Super-120B-A12B` | `RAP_NEMOTRON_3_ULTRA_API_KEY` |
| NCHC RAP Nemotron Ultra | `rap-nemotron/NVIDIA-Nemotron-3-Ultra-550B-A55B` | 與 Super 共用 `RAP_NEMOTRON_3_ULTRA_API_KEY` |
| NCHC RAP Gemma 26B | `rap-gemma-26b/gemma-4-26B-A4B-it` | `RAP_GEMMA_26B_API_KEY` |
| NCHC RAP Gemma 31B | `rap-gemma-31b/gemma-4-31B-it` | `RAP_GEMMA_31B_API_KEY` |
| NVIDIA hosted NIM | `nvidia-nim/nvidia/nemotron-3-ultra-550b-a55b` | `NVIDIA_NIM_API_KEY` |
| OpenCode Zen Nemotron | `opencode/nemotron-3-ultra-free` | Free pricing；個人 OpenCode Zen API key（`/connect`） |

Zen free endpoint 是限時 trial，可能有 rate limit，且官方聲明試用資料會用於安全管理與 NVIDIA 產品／服務改善；不得提交個資、機密資料或任何課程憑證。

## 10. VM image 封裝前檢查

先停止服務：

```bash
cd /home/ubuntu/cudaq-course-kit
./stop-lab.sh
```

本活動的母 image 刻意保留一份 root-only 共用活動憑證包，但不保留講師的個人 `course.env`、Jupyter token 或 OpenCode runtime。完成最後一次 Super 手動測試後，建議直接執行：

```bash
sudo /home/ubuntu/cudaq-course-kit/finalize-vm-image.sh
```

執行後不要再啟動 Lab，直接依國網流程關機封裝。下列為腳本內部執行的關鍵檢查，保留供人工稽核：

```bash
sudo stat -c '%a %U:%G %n' \
  /opt/nchc-cudaq-course/activity/activity-keys.env
sudo find /home/ubuntu/.config/nchc-cudaq-course \
  -maxdepth 1 -type f -name 'course.env*' -delete
sudo find /home/ubuntu/cudaq-course-kit/.runtime -mindepth 1 -delete
```

活動憑證包必須是 `600 root:root`。確認 key 沒有出現在 Docker layer、Git、教材或一般 kit 檔；以下檢查刻意不掃描 activity 目錄：

```bash
if sudo rg -l '(sk-[A-Za-z0-9_-]{12,}|nvapi-[A-Za-z0-9_-]{12,})' \
  /opt/nchc-cudaq-course/source \
  /opt/nchc-cudaq-course/kit \
  /home/ubuntu/cudaq-course-kit \
  /home/ubuntu/cudaq-agentic-coding >/dev/null; then
  echo 'FAIL: unexpected credential outside activity bundle'
else
  echo 'PASS: credentials exist only in authorized activity bundle'
fi

if docker history --no-trunc nchc/cudaq-agentic-coding:2026-07-17 | \
  rg '(sk-[A-Za-z0-9_-]{12,}|nvapi-[A-Za-z0-9_-]{12,})' >/dev/null; then
  echo 'FAIL: possible credential in Docker history'
else
  echo 'PASS: no credential in Docker history'
fi

test ! -e /home/ubuntu/.config/nchc-cudaq-course/course.env
test -x /opt/nchc-cudaq-course/kit/activate-student-course.sh
docker image inspect nchc/cudaq-agentic-coding:2026-07-17 >/dev/null
```

依 NCHC 平台流程關機封裝。活動憑證包會隨 VM filesystem 進入 image；Docker image 本身仍不含 key。

## 11. 從 image重新部署驗收

國網以母 image 建立一台全新 VM，以學生 `ubuntu` 身份只執行：

```bash
cd ~/cudaq-course-kit
./activate-student-course.sh
```

這一個腳本必須完成憑證帶入、唯一 Jupyter token 產生、環境驗證與 Lab 啟動。

驗收條件：

- 顯示活動 key 課後失效與日後 BYOK 通知
- `course.env` 是 `600 ubuntu:ubuntu`
- 不需 apt、pip、Docker build、NGC pull、GitHub 登入或手動貼 key
- 10 分鐘內完成環境驗證並開啟 JupyterLab
- 筆電可直接開啟 `http://VM_IP:8888/lab` 並以獨立 token 登入；SSH 3322 tunnel 只作備援
- Container 重啟後 Notebook 與 OpenCode user data 仍存在
- `course-reset.sh` 會先保留可回復備份，再還原乾淨教材
- 活動結束立即撤銷映像內所有共用 key

### 40 台 VM 課前連線驗收

國網／助教應維護不含 private key 的學生對照表：

```text
座位或學員編號, VM IP, VM狀態, SSH驗證, GPU驗證, Jupyter驗證
```

課前逐台或以受控自動化驗證：

- IP 可連線且帳號為 `ubuntu`
- `bootcamp0807.pem` 對應的 public key 已在 `authorized_keys`
- 每台 VM hostname或分配標籤可區分
- `verify-environment.sh` 通過
- 8888 未暴露到公網
- 學生工作目錄互相隔離

Private key只透過受控管道提供給學員，不得放在教材 repo、VM 公開目錄或課程網站。活動結束後：

- 從 VM 移除對應 public key或直接回收全部 VM
- 要求學生刪除 `bootcamp0807.pem`
- 下一場活動產生新的 key pair，不重用這把 private key

## 12. 未來公開 GitHub repo 後

目前課堂仍以 VM 中的已驗證快照為準。Repo 公開後，教師可在下一版母 image更新：

```bash
git clone REPOSITORY_URL /tmp/cudaq-agentic-coding-new
```

完成完整 Notebook、環境與時間驗收後，才替換 `/opt/nchc-cudaq-course/source` 並發布新 image tag。不要讓學生上課當天直接追蹤 `main`，避免教材與環境在課中漂移。

## 13. 活動結束

- 撤銷所有 RAP 與 NVIDIA NIM活動 key。
- 終止或回收學生 VM。
- 保存不含秘密的 image digest、套件 lock、驗收結果與教材 commit／archive SHA。
- 下一場活動使用新 key 與新 image tag，不重用舊 `.env`。
