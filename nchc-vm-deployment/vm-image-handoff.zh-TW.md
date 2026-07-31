# NCHC H200 母 VM：Super 手動測試與映像交付

本版 OpenCode 預設模型：

```text
rap-nemotron/NVIDIA-Nemotron-3-Super-120B-A12B
```

同一個 `rap-nemotron` provider 也保留：

```text
rap-nemotron/NVIDIA-Nemotron-3-Ultra-550B-A55B
```

Super 與 Ultra 共用 `RAP_NEMOTRON_3_ULTRA_API_KEY`，不需要第二把 key。

## 1. 手動測試前

不要先執行 `finalize-vm-image.sh`。確認 Lab：

```bash
cd ~/cudaq-course-kit
./start-lab.sh
```

腳本會在目前 Terminal 顯示完整 JupyterLab 登入連結。連結包含 token，不要貼到聊天、文件或截圖。

只檢查預設模型 ID，不顯示設定或 key：

```bash
sudo jq -r '.model' ~/cudaq-course-kit/.runtime/opencode.json
```

預期：

```text
rap-nemotron/NVIDIA-Nemotron-3-Super-120B-A12B
```

## 2. 在 JupyterLab Terminal 測試 Super

```bash
cd /workspace/cudaq-agentic-coding
opencode
```

執行 `/models`，確認目前是 Nemotron 3 Super；Ultra 應同時可選。

第一個 prompt 只做唯讀確認：

```text
Read @cudaq-doc.md and @AGENTS.md, then give me a short summary of what
each one covers. Just the wrap-up. No code or files yet.
```

確認成功後，另一個 prompt 測試 CUDA-Q 工具鏈：

```text
Write a simple script ghz.py using CUDA-Q to prepare a 20-qubit GHZ state on the GPU
("nvidia" target), sample 1000 shots, and print the counts. Refer to @cudaq-doc.md if needed,
then run it and show me the output.
```

不要一次貼多個教材 Step。若需要明確指定模型，可用：

```bash
opencode --model rap-nemotron/NVIDIA-Nemotron-3-Super-120B-A12B
```

## 3. 測試完成後才執行最終封裝清理

先把需要保留且不含憑證的測試摘要下載到管理者電腦，結束 Codex 管理工作階段，再從獨立 SSH terminal 執行（腳本會刪除 VM 上的 `.codex` 本機狀態）：

```bash
sudo /home/ubuntu/cudaq-course-kit/finalize-vm-image.sh
```

腳本會：

- 停止並移除課程 container。
- 從已驗證 `course.env` 更新唯一允許保留的 root-only 活動憑證包。
- 確認活動憑證包為 `600 root:root`。
- 刪除 `course.env*`、Jupyter token、OpenCode/Jupyter runtime、Codex/CLI 本機狀態、使用者 cache、session、history 與 log。
- 刪除舊 course/kit/source 備份、驗收工作區、build workspace 與使用者 shell history。
- 移除課程衍生 skills。
- 驗證正式 Docker image、乾淨教材與學生 workspace仍存在。
- 掃描 Docker history 與正式教材路徑，只回報是否有憑證 pattern，不顯示匹配內容。

成功後不要再啟動 Lab，直接依 NCHC 平台流程關機並建立 VM image。

## 4. 國網封裝時保留

```text
/opt/nchc-cudaq-course/source
/opt/nchc-cudaq-course/kit
/opt/nchc-cudaq-course/activity/activity-keys.env
/home/ubuntu/cudaq-agentic-coding
/home/ubuntu/cudaq-course-kit
/home/ubuntu/vm-image-handoff
nchc/cudaq-agentic-coding:2026-07-17
```

`activity-keys.env` 是唯一刻意保留的共用 key 檔案，必須為 `600 root:root`。學生具 sudo 權限，活動結束後必須立即撤銷所有共用 key。

是否執行 `cloud-init clean --logs --machine-id`、重建 SSH host key、hostname 與 machine-id，必須依 NCHC 映像平台 runbook 決定，不要在母 VM 自行假設。
