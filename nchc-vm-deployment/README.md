# 國網中心 H200 CUDA-Q Agentic Coding VM Kit

這個資料夾是 `cudaq-agentic-coding` 課程的 VM 環境套件。它將 CUDA-Q、JupyterLab、Jupyter AI、OpenCode 與課程依賴封裝在同一個 GPU container；國網先將 container image 與教材快照放進 VM image，學生上課時不需要安裝或 build。

## 兩種使用角色

- 學生：使用繁中版 [Markdown](student-guide.zh-TW.md)／[PDF](student-guide.zh-TW.pdf)，或英文版 [Markdown](student-guide.en.md)／[PDF](student-guide.en.pdf)，再執行驗證與啟動腳本。
- 教師／國網維運：閱讀 [instructor-image-guide.zh-TW.md](instructor-image-guide.zh-TW.md)，建立、驗收及封裝母 VM。
- 封裝前人工複驗：閱讀 [manual-validation-before-image.zh-TW.md](manual-validation-before-image.zh-TW.md)，隔離驗證 OpenCode 並建立活動憑證包。

課堂預設由國網準備約 40 台 VM；學生使用指定 IP、帳號 `ubuntu` 與活動用 `bootcamp0807.pem`，從 macOS Terminal 或 Windows PowerShell 連線。

## 固定版本

| 元件 | 版本 |
|---|---|
| CUDA-Q container | `nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.15.0` |
| JupyterLab | `4.6.1` |
| Jupyter AI | `3.0.1` |
| OpenCode | `1.18.1` |
| Qiskit | `2.5.0` |

CUDA-Q base image使用 `linux/amd64` manifest digest固定，適用本課程的 x86_64 H200 VM。

## 主要指令

```bash
# 講師驗收後，將目前 key 封裝成 VM image 的 root-only 活動憑證包
./configure-course-env.sh
sudo ./prepare-activity-image-credentials.sh

# 學生只執行一次：帶入活動 key、產生個人 token、驗證並啟動 Lab
./activate-student-course.sh

# 後續停止／重新啟動／查看 JupyterLab
./stop-lab.sh
./start-lab.sh
./stop-lab.sh
./lab-logs.sh

# 可恢復地重設完整手動驗收狀態
./reset-manual-validation.sh

# 完成最後一次測試後，清除測試狀態並交付國網封裝（執行後不要再啟動 Lab）
sudo ./finalize-vm-image.sh

# 先備份目前作業，再還原乾淨教材
./course-reset.sh
```

## 安全邊界

- API key 不得寫入 Dockerfile、container layer、Git、Notebook 或一般教材檔。
- 本活動經講師明確接受風險後，例外將共用 key 放在 VM image 的 `/opt/nchc-cudaq-course/activity/activity-keys.env`；檔案為 `600 root:root`。
- 學生具備 VM sudo 權限，因此仍可擷取共用 key；root-only 只防止誤讀。活動結束必須立即撤銷。
- 學生啟用後的個人 `course.env` 為 mode 600，Jupyter token 每台獨立產生。
- JupyterLab 預設發布在 VM 的 `0.0.0.0:8888`，學生以 VM IP 與獨立 token 直接連線；SSH TCP 3322 tunnel 僅作備援。
- OpenCode sharing 預設關閉；shell、修改與 Notebook/MCP 執行需要使用者批准。
- NCHC RAP `rap-nemotron` 同時提供 Nemotron 3 Super 與 Ultra，兩者共用同一把 RAP key；課程預設為 Super。
- OpenCode Zen 的 `opencode/nemotron-3-ultra-free` 作為 RAP 變慢時的限時免費 fallback；它是內建 provider，學生必須以自己的 Zen API key 執行 `/connect`，不得將 Zen key 封裝進 VM 或課程 env。

## 檔案用途

| 檔案 | 用途 |
|---|---|
| `Dockerfile` | 建立固定版本課程 container |
| `compose.yaml` | 學生啟動預建 image，不允許自動 build |
| `configure-course-env.sh` | 使用固定 endpoint/model ID，只隱藏輸入 key 並建立權限 600 的憑證檔 |
| `provision-course-env.sh` | 啟用腳本內部使用固定設定，建立學生的 mode 600 course env |
| `render-opencode-config.sh` | 只為已提供 key 的固定 provider 產生不含明文 key 的 `opencode.json` |
| `prepare-activity-image-credentials.sh` | 講師將驗證過的 key 建成 VM image 內 root-only 活動憑證包 |
| `activate-student-course.sh` | 學生一鍵帶入 key、產生 token、驗證並啟動 Lab |
| `course-reset.sh` | 備份目前學生作業，再從正式教材快照還原乾淨工作區 |
| `reset-manual-validation.sh` | 可恢復地重設教材、Notebook/程式衍生物、Jupyter/OpenCode runtime、stats/session/cache/log 與已安裝的課程 skills，保留課程憑證並驗證 workspace 等同唯讀 source |
| `finalize-vm-image.sh` | 最後一次人工測試後停止服務、更新 root-only 活動憑證包，清除 env/token/runtime/驗收備份與 build workspace，完成不顯示內容的 secret audit |
| `vm-image-handoff.zh-TW.md` | Nemotron 3 Super 最後手測、封裝清理與國網 VM image 交付順序 |
| `verify-environment.sh` | H200、Docker、CUDA-Q、Jupyter、OpenCode 與 repo 驗收 |
| `validate-course-notebooks.py` | 檢查既有 Notebook JSON、Python 語法、kernel、執行狀態與 error outputs |
| `prepare-course-source.sh` | 從課程 zip 安裝唯讀教材快照與 VM kit |
| `install-host-docker.sh` | 新 Ubuntu 母 VM 的 Docker/NVIDIA runtime 安裝參考 |
