# OpenCode Setup

Get [**OpenCode**](https://opencode.ai/docs/) — an open-source coding agent that runs in your
terminal — working with **NVIDIA Nemotron 3 Ultra** in about five minutes.

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

## 4. Paste your key into it

Open the copied file and replace the placeholder with the key from step 2:

```bash
nano ~/.config/opencode/opencode.json     # or vim, code, gedit — any editor
```

```jsonc
"apiKey": "nvapi-PASTE-YOUR-KEY-HERE"     // ← replace this line's value with your key
```

It should end up looking like `"apiKey": "nvapi-xxxxxxxxxxxxxxxx"`. Must keep the prefix: `nvapi-`, but don't add a `Bearer `.

Here is the whole file for reference, with the one line you edit marked:

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
        "apiKey": "nvapi-PASTE-YOUR-KEY-HERE"        // ← your key goes here!
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

A one-word answer means install, key, config, and routing are all correct. **You're done.**

> If the model name is ever rejected, ask the API which names it currently serves and copy
> the exact string into the `id` field:
> ```bash
> curl -s https://integrate.api.nvidia.com/v1/models | grep -o '"id":"[^"]*"'
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
| `/init` | scan the project and write an `AGENTS.md` for it |
| `/models` | switch model mid-session |
| `/undo` | revert the last change the agent made |
| `Ctrl+C` / `/exit` | quit |


**`AGENTS.md` is worth knowing about.** If a folder contains one, OpenCode reads it
automatically and follows it as project instructions — conventions, build commands, house
style. Project files win over the global `~/.config/opencode/AGENTS.md`; see
[rules](https://opencode.ai/docs/rules/).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `opencode: command not found` | Binary is in `~/.opencode/bin` — reopen the terminal, or `export PATH="$HOME/.opencode/bin:$PATH"` |
| `401` / `403` / "invalid API key" | The placeholder is probably still in place. Check `grep apiKey ~/.config/opencode/opencode.json` — if it still says `PASTE-YOUR-KEY-HERE`, redo step 4. Paste the bare token, no `Bearer ` prefix. |
| `404` / "model not found" | The model name drifted. Run the `curl` in step 5 and copy the exact string into `id`. |
| Config seems ignored | A project-level `opencode.json` in your current folder overrides the global one. Check for it. |
| Changes to `opencode.json` do nothing | Restart `opencode` — config is read at startup. |
| "Unexpected token" / config won't load | A stray comma or missing quote from step 4. Validate with `python3 -m json.tool ~/.config/opencode/opencode.json` |
| Slow or truncated replies | Expected on a 550B model for long inputs. Narrow the request and try again. |
