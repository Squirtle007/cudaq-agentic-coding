# OpenCode Setup

> The instructions below are for post-bootcamp setup - OpenCode is preinstalled for the bootcamp, so no installation is needed.

Get [**OpenCode**](https://opencode.ai/docs/) — an open-source coding agent that runs in your
terminal — working with **NCHC RAP Nemotron 3** in about five minutes. Nemotron 3 Super is
the default; Nemotron 3 Ultra is available in the same config and selectable mid-session.

This folder is self-contained. It sets up the agent itself, not any particular project, so
once you finish you can point OpenCode at whatever you like.

**You need:** a terminal on Linux, macOS, or Windows/WSL2 · `curl` · an API key for [**NCHC RAP**](https://portal.genai.nchc.org.tw) from the portal or your course organizer.

---

## 1. Install OpenCode

```bash
curl -fsSL https://opencode.ai/install | bash
```

That one command covers **Linux, macOS, and WSL**:

> **On Windows, use WSL.** It's what OpenCode recommends, and what this guide assumes.
> If you don't have it yet, follow the
> [WSL setup guide](https://opencode.ai/docs/windows-wsl), open the WSL terminal, then
> run the command above.

Prefer a package manager? Any of these work equally well
([all install options](https://opencode.ai/docs/)):

```bash
npm install -g opencode-ai          # npm — Linux / macOS / WSL / Windows
brew install anomalyco/tap/opencode # macOS / Homebrew
sudo pacman -S opencode             # Arch Linux
```

Check it landed:

```bash
opencode --version
```

> **`opencode: command not found`?** The installer puts the binary in `~/.opencode/bin`.
> Reopen your terminal, or add it to your path for this session:
> ```bash
> export PATH="$HOME/.opencode/bin:$PATH"
> ```

---

## 2. Get your NCHC RAP API key

Sign in to the [NCHC RAP portal](https://portal.genai.nchc.org.tw) and issue an API key, or
ask your course organizer for the one provided for your event.

One key serves both Nemotron 3 Super and Nemotron 3 Ultra, so you only do this once.

---

## 3. Install the config

> **Re-run this step after every config change.** OpenCode reads the *installed* copy in
> `~/.config/opencode/`, not the one in this folder. So whenever you reconfigure
> `opencode.json` here — switching the model, editing the provider — copy it across again
> with the two commands below, then restart `opencode`.

Copy the `opencode.json` from this folder into OpenCode's global config directory:

```bash
mkdir -p ~/.config/opencode
cp opencode.json ~/.config/opencode/
```

Global config applies in every folder you work in. See
[configuration](https://opencode.ai/docs/config/) for the full list of options.

> You may notice OpenCode already put an `opencode.jsonc` in that folder on first run.
> That's harmless — leave it alone. Your `opencode.json` is read as well.

---

## 4. Set your key as an environment variable

The key stays out of the config file. Export it in the same terminal you'll start OpenCode
from, using the key from step 2:

```bash
export RAP_API_KEY="YOUR_RAP_API_KEY"
```

Use the complete key exactly as issued — don't add a `Bearer ` prefix. The config reads
`{env:RAP_API_KEY}` at startup, so nothing secret is ever written to disk.

The export only lasts for the current terminal session. If you want it to survive a restart,
add it to your shell startup file or whatever secret manager you already use — but never
commit it.

Here is the whole file for reference. The RAP endpoint wants the key in two places, and both
read the same variable, so the one export above covers them:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "rap-nemotron/NVIDIA-Nemotron-3-Super-120B-A12B",
  "small_model": "rap-nemotron/NVIDIA-Nemotron-3-Super-120B-A12B",
  "share": "disabled",
  "provider": {
    "rap-nemotron": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NCHC RAP Nemotron 3",
      "options": {
        "baseURL": "https://portal.genai.nchc.org.tw/api/v1",
        "apiKey": "{env:RAP_API_KEY}",                 // ← your key is read from here
        "headers": {
          "x-api-key": "{env:RAP_API_KEY}"             // ← and here — same variable
        }
      },
      "models": {
        "NVIDIA-Nemotron-3-Super-120B-A12B": { "name": "NCHC RAP Nemotron 3 Super" },
        "NVIDIA-Nemotron-3-Ultra-550B-A55B": { "name": "NCHC RAP Nemotron 3 Ultra" }
      }
    }
  }
}
```

---

## 5. Verify it works

First confirm the config parses and the model is registered
([CLI reference](https://opencode.ai/docs/cli/)):

```bash
opencode models rap-nemotron
```

You should see both models listed — Super and Ultra. Super is the one your config uses by
default.

Then make a real call:

```bash
opencode run "hello opencode"
```

> If a model name is ever rejected, ask the API which names it currently serves and copy the
> exact string into `models`:
> ```bash
> curl -s -H "x-api-key: $RAP_API_KEY" \
>   https://portal.genai.nchc.org.tw/api/v1/models | grep -o '"id":"[^"]*"'
> ```

---

## 6. Use it

```bash
cd /path/to/your/project
opencode
```

Useful commands inside the session:

| | |
|---|---|
| `@path/to/file` | pull a file into the conversation |
| `/init` | write an `AGENTS.md` for a project that doesn't have one yet |
| `/models` | switch model mid-session |
| `/undo` | revert the last change the agent made |
| `Ctrl+C` / `/exit` | quit |


**`AGENTS.md` is worth knowing about.** If a folder contains one, OpenCode reads it
automatically and follows it as project instructions — conventions, build commands, house
style. Project files win over the global `~/.config/opencode/AGENTS.md`; see
[rules](https://opencode.ai/docs/rules/).

---

## 7. Add the official CUDA-Q skill

A skill is a reusable instruction set that teaches your agent a topic. NVIDIA publishes
[CUDA-X skills for agents](https://github.com/NVIDIA/skills), and
[**`cudaq-guide`**](https://github.com/NVIDIA/skills/tree/main/skills/cudaq-guide) covers
CUDA-Q installation, GPU simulation, QPU hardware, and applications:

```bash
npx skills add nvidia/skills --skill cudaq-guide --agent opencode --yes
```

It lands in `~/.config/opencode/skills/cudaq-guide/`. Restart `opencode`, then just ask a
CUDA-Q question — the agent picks the skill up on its own.

> Needs Node for `npx`. Skills are portable: swap `--agent` for `claude-code`, `codex`, or
> `cursor` to install the same skill elsewhere.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `opencode: command not found` | Binary is in `~/.opencode/bin` — reopen the terminal, or `export PATH="$HOME/.opencode/bin:$PATH"` |
| `401` / `403` / "invalid API key" | `RAP_API_KEY` probably isn't set in the terminal that launched OpenCode. Check with `[ -n "$RAP_API_KEY" ] && echo SET || echo UNSET` in that same terminal — that tests it without printing it — and redo step 4 if needed. Use the bare token, no `Bearer ` prefix. |
| `404` / "model not found" | The model name drifted. Run the `curl` in step 5 and copy the exact string into `models`. |
| Config seems ignored | A project-level `opencode.json` in your current folder overrides the global one. Check for it. |
| Changes to `opencode.json` do nothing | Either you edited the copy in this folder instead of the installed one — redo step 3 — or `opencode` is still running. Restart it; config is read at startup. |
| "Unexpected token" / config won't load | The copied file got mangled — a stray comma or missing quote. Validate with `python3 -m json.tool ~/.config/opencode/opencode.json`, then redo step 3 from a clean copy. |
| Slow or truncated replies | Expected on a 550B model for long inputs. Narrow the request and try again. |
| The model stops responding | Type `/models` in the session to list what's available and switch to another one. |
