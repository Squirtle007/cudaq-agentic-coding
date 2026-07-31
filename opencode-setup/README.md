# OpenCode Setup

> The instructions below are for post-bootcamp setup - OpenCode is preinstalled for the bootcamp, so no installation is needed.

Get [**OpenCode**](https://opencode.ai/docs/) — an open-source coding agent that runs in your
terminal — working with **NVIDIA Nemotron 3 Ultra** in about five minutes. The main path uses
NVIDIA hosted NIM; an additional tested config is included for NCHC RAP users.

This folder is self-contained. It sets up the agent itself, not any particular project, so
once you finish you can point OpenCode at whatever you like.

**You need:** a terminal on Linux, macOS, or Windows/WSL2 · `curl` · a free account at [**NVIDIA Inference Microservices (NIM)**](https://build.nvidia.com).

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

## 2. Get your NVIDIA API key

Open the
[Nemotron 3 Ultra model page](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b),
sign in, and click **Get API Key**. Copy the token — it starts with `nvapi-`.

The same key works for every model in the
[NVIDIA API catalog](https://build.nvidia.com/models), so you only do this once.

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

Keep the credential out of the JSON file. Export it in the same terminal where you will
start OpenCode:

```bash
export NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
```

Use your complete key, including its required prefix, but do not add `Bearer `. OpenCode
supports `{env:VARIABLE_NAME}` substitution, so the checked-in config reads the value at
startup without storing it in Git.

Here is the relevant part of the installed config:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "nvidia/nemotron-3-ultra-550b-a55b",
  "provider": {
    "nvidia": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NVIDIA NIM",
      "options": {
        "baseURL": "https://integrate.api.nvidia.com/v1",
        "apiKey": "{env:NVIDIA_API_KEY}"
      },
      "models": {
        "nemotron-3-ultra-550b-a55b": {
          "id": "nvidia/nemotron-3-ultra-550b-a55b",
          "name": "NVIDIA Nemotron 3 Ultra 550B A55B",
          "limit": { "context": 1000000, "output": 65536 }
        }
      }
    }
  }
}
```

The export lasts for the current terminal session. Add it to a secure shell startup or
environment-management mechanism if you need it across restarts; never commit the value.

---

## 5. Verify it works

First confirm the config parses and the model is registered
([CLI reference](https://opencode.ai/docs/cli/)):

```bash
opencode models nvidia | grep nemotron-3-ultra
```

You should see `nvidia/nemotron-3-ultra-550b-a55b` — that's the one your config uses.

> The unfiltered `opencode models nvidia` prints ~100 entries, because OpenCode also
> auto-discovers NVIDIA's whole catalog. You'll spot a near-duplicate
> `nvidia/nvidia/nemotron-3-ultra-550b-a55b` in there; that's the auto-discovered form.
> Either works — the single-prefix one is what this config defines.

Then make a real call:

```bash
opencode run "hello opencode"
```

> If the model name is ever rejected, ask the API which names it currently serves and copy
> the exact string into the `id` field:
> ```bash
> curl -s https://integrate.api.nvidia.com/v1/models | grep -o '"id":"[^"]*"'
> ```

### NCHC RAP alternative

NCHC RAP users can use the included `opencode.nchc-rap.json` instead. It registers both
Nemotron 3 models under one provider:

- `NVIDIA-Nemotron-3-Super-120B-A12B` — default
- `NVIDIA-Nemotron-3-Ultra-550B-A55B` — selectable fallback

The two models share one RAP API key. Obtain the key from NCHC RAP or your course organizer,
then install the alternative config and export the key in the terminal where OpenCode runs:

```bash
cp opencode.nchc-rap.json ~/.config/opencode/opencode.json
export RAP_NEMOTRON_3_ULTRA_API_KEY="YOUR_RAP_API_KEY"
opencode models rap-nemotron
opencode
```

Use `/models` inside OpenCode to switch between Super and Ultra. The example sends the same
environment variable as both the OpenAI-compatible `apiKey` and `x-api-key` header required
by the RAP endpoint; the key value is never stored in the tracked JSON file.

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
| `401` / `403` / "invalid API key" | Confirm the matching environment variable is set in the terminal that launched OpenCode: `NVIDIA_API_KEY` for NIM or `RAP_NEMOTRON_3_ULTRA_API_KEY` for NCHC RAP. Use the bare token, with no `Bearer ` prefix. |
| `404` / "model not found" | The model name drifted. Run the `curl` in step 5 and copy the exact string into `id`. |
| Config seems ignored | A project-level `opencode.json` in your current folder overrides the global one. Check for it. |
| Changes to `opencode.json` do nothing | Either you edited the copy in this folder instead of the installed one — redo step 3 — or `opencode` is still running. Restart it; config is read at startup. |
| "Unexpected token" / config won't load | A stray comma or missing quote from step 4. Validate with `python3 -m json.tool ~/.config/opencode/opencode.json` |
| Slow or truncated replies | Expected on a 550B model for long inputs. Narrow the request and try again. |
| The model stops responding | Type `/models` in the session to list what's available and switch to another one. |
