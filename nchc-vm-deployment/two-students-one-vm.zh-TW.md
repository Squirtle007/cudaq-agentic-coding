# 兩位學員共用一台 NCHC H200 VM

這一版適用於兩位學員共用同一個 VM IP 與 `ubuntu` SSH 帳號。兩位學員仍共用同一張
H200 GPU 與活動 API key，但各自使用不同的教材工作副本、Jupyter/OpenCode runtime、
Docker Compose project、TCP port 與 Jupyter token，Notebook、Chat、session 與 token
不會互相覆蓋。

| 項目 | 學員 1 | 學員 2 |
|---|---|---|
| Jupyter port | `8888` | `8889` |
| 教材 | `~/cudaq-agentic-coding` | `~/cudaq-agentic-coding-student2` |
| 工具／runtime | `~/cudaq-course-kit` | `~/cudaq-course-kit-student2` |
| Course env | `~/.config/nchc-cudaq-course/course.env` | `~/.config/nchc-cudaq-course-student2/course.env` |
| Compose project | `nchc-cudaq-course` | `nchc-cudaq-course-student2` |

> 兩個 container 會競爭同一張 GPU 的計算與記憶體。請勿同時執行長時間、高 GPU
> 記憶體的 Notebook；這份隔離設計不會把一張實體 GPU 分割成兩張。

## 只替換啟用腳本（既有 VM image）

舊 VM 不必重建 image。先登入 VM，下載新版腳本覆蓋原檔：

```bash
cd ~/cudaq-course-kit
downloaded_script=$(mktemp ./activate-student-course.sh.download.XXXXXX)
curl -fL \
  https://raw.githubusercontent.com/Squirtle007/cudaq-agentic-coding/opencode-nchc-rap/nchc-vm-deployment/activate-student-course.sh \
  -o "$downloaded_script" && \
  chmod 0755 "$downloaded_script" && \
  mv -f -- "$downloaded_script" ./activate-student-course.sh
rm -f "$downloaded_script"
```

下載先寫入同一個工具目錄的暫存檔；只有 `curl` 與 `chmod` 都成功才用同 filesystem 的
`mv` 原子覆蓋舊腳本，下載中斷不會留下半份啟用程式。

新版腳本會以 `sudo` 執行「目前這一份下載檔」，不再跳回 VM image 內的舊版 canonical
腳本。因此只覆蓋 `~/cudaq-course-kit/activate-student-course.sh` 就能啟用雙學員功能。

## 學員 1

第一位學員先執行：

```bash
cd ~/cudaq-course-kit
./activate-student-course.sh
```

第一次執行會建立或啟動學員 1 的環境，最後顯示含 token 的完整連結：

```text
http://你的VM-IP:8888/lab?token=學員1的token
```

請保持學員 1 的 JupyterLab container 執行，再讓學員 2 執行下一節的同一個指令。

## 學員 2

第二位學員同樣登入 `ubuntu`，再次執行完全相同的啟用指令：

```bash
cd ~/cudaq-course-kit
./activate-student-course.sh
```

腳本看到學員 1 的 Compose project 正在執行後，會自動：

1. 從 VM 的 root-owned 唯讀 source 建立 `~/cudaq-agentic-coding-student2`。
2. 建立 `~/cudaq-course-kit-student2` 與獨立 `.runtime`。
3. 建立 mode 600 的第二份 `course.env` 與新的 Jupyter token。
4. 使用 Compose project `nchc-cudaq-course-student2` 發布 TCP `8889`。
5. 驗證環境，並顯示學員 2 含 token 的完整連結。

```text
http://你的VM-IP:8889/lab?token=學員2的token
```

若學員 2 的環境已存在，再次執行同一個啟用指令不會換掉 token 或重建 workspace；
腳本會啟動既有 Lab（如果已停止），並再次顯示包含原 token 的完整登入連結。

啟用程序使用主機鎖避免兩位學員同時執行時搶用同一個 slot。若第二份 container 存在但
第二份 `course.env` 遺失，腳本會停止並請助教復原，不會偷偷產生一個對不上既有服務的 token。

## 個別啟動、停止與查看 log

```bash
# 學員 1
~/cudaq-course-kit/start-lab.sh
~/cudaq-course-kit/lab-logs.sh
~/cudaq-course-kit/stop-lab.sh

# 學員 2
~/cudaq-course-kit-student2/start-lab.sh
~/cudaq-course-kit-student2/lab-logs.sh
~/cudaq-course-kit-student2/stop-lab.sh
```

學員 2 工具目錄中的 wrapper 會自動選擇第二份 `course.env`，不需要手動 export。

## SSH tunnel 備案

直接連線不可用時，每位學員使用自己的 port。請在各自筆電執行：

```bash
# 學員 1
ssh -p 3322 -i ~/Downloads/bootcamp0807.pem \
  -N -L 8888:127.0.0.1:8888 ubuntu@你的VM-IP

# 學員 2
ssh -p 3322 -i ~/Downloads/bootcamp0807.pem \
  -N -L 8889:127.0.0.1:8889 ubuntu@你的VM-IP
```

學員 1 開啟 `http://localhost:8888/lab?token=...`；學員 2 開啟
`http://localhost:8889/lab?token=...`。不要互傳或貼出 token。

## 教師／網路端檢查

- NCHC security group／ACL 除了 TCP 8888，還要允許課程來源網段連入 TCP 8889。
- 用 `docker ps` 確認兩個 Compose project 各有一個 `lab` container。
- 用 `ss -ltnp | grep -E ':8888|:8889'` 確認兩個 publication。
- 不要把兩位學員改成共用同一個可寫 repo 或同一份 `course.env`。
