#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 COURSE_ENV_FILE OUTPUT_JSON" >&2
  exit 2
fi

course_env_file=$1
output_json=$2

if [[ ! -f "$course_env_file" ]]; then
  echo "Course env file not found: $course_env_file" >&2
  exit 1
fi

set -a
# The course env file is created by configure-course-env.sh and is trusted input.
# shellcheck disable=SC1090
source "$course_env_file"
set +a

nemotron_key_env=RAP_NEMOTRON_3_ULTRA_API_KEY
if [[ -z "${RAP_NEMOTRON_3_ULTRA_API_KEY:-}" && -n "${RAP_NENOTRON_3_ULTRA_API_KEY:-}" ]]; then
  nemotron_key_env=RAP_NENOTRON_3_ULTRA_API_KEY
fi

nemotron_enabled=false
gemma26_enabled=false
gemma31_enabled=false
nim_enabled=false
[[ -n "${RAP_NEMOTRON_3_ULTRA_API_KEY:-${RAP_NENOTRON_3_ULTRA_API_KEY:-}}" ]] && nemotron_enabled=true
[[ -n "${RAP_GEMMA_26B_API_KEY:-}" ]] && gemma26_enabled=true
[[ -n "${RAP_GEMMA_31B_API_KEY:-}" ]] && gemma31_enabled=true
[[ -n "${NVIDIA_NIM_API_KEY:-}" ]] && nim_enabled=true
default_model=
if [[ "$nemotron_enabled" == true && -n "${NCHC_RAP_BASE_URL:-}" && -n "${RAP_NEMOTRON_3_SUPER_MODEL:-}" ]]; then
  default_model="rap-nemotron/${RAP_NEMOTRON_3_SUPER_MODEL}"
elif [[ "$nemotron_enabled" == true && -n "${NCHC_RAP_BASE_URL:-}" && -n "${RAP_NEMOTRON_3_ULTRA_MODEL:-}" ]]; then
  default_model="rap-nemotron/${RAP_NEMOTRON_3_ULTRA_MODEL}"
elif [[ "$gemma31_enabled" == true && -n "${NCHC_RAP_BASE_URL:-}" && -n "${RAP_GEMMA_31B_MODEL:-}" ]]; then
  default_model="rap-gemma-31b/${RAP_GEMMA_31B_MODEL}"
elif [[ "$gemma26_enabled" == true && -n "${NCHC_RAP_BASE_URL:-}" && -n "${RAP_GEMMA_26B_MODEL:-}" ]]; then
  default_model="rap-gemma-26b/${RAP_GEMMA_26B_MODEL}"
elif [[ "$nim_enabled" == true && -n "${NVIDIA_NIM_MODEL:-}" ]]; then
  default_model="nvidia-nim/${NVIDIA_NIM_MODEL}"
fi


mkdir -p "$(dirname "$output_json")"

jq -n \
  --arg rap_base "${NCHC_RAP_BASE_URL:-}" \
  --arg nemotron_super_model "${RAP_NEMOTRON_3_SUPER_MODEL:-}" \
  --arg nemotron_ultra_model "${RAP_NEMOTRON_3_ULTRA_MODEL:-}" \
  --arg nemotron_key_env "$nemotron_key_env" \
  --arg gemma26_model "${RAP_GEMMA_26B_MODEL:-}" \
  --arg gemma31_model "${RAP_GEMMA_31B_MODEL:-}" \
  --arg nim_base "${NVIDIA_NIM_BASE_URL:-https://integrate.api.nvidia.com/v1}" \
  --arg nim_model "${NVIDIA_NIM_MODEL:-}" \
  --arg default_model "$default_model" \
  --argjson nemotron_enabled "$nemotron_enabled" \
  --argjson gemma26_enabled "$gemma26_enabled" \
  --argjson gemma31_enabled "$gemma31_enabled" \
  --argjson nim_enabled "$nim_enabled" \
  '
  def rap_provider($display; $model; $key_env): {
    npm: "@ai-sdk/openai-compatible",
    name: $display,
    options: {
      baseURL: $rap_base,
      apiKey: ("{env:" + $key_env + "}"),
      headers: {"x-api-key": ("{env:" + $key_env + "}")}
    },
    models: {($model): {name: $display}}
  };
  def rap_nemotron_provider($super_model; $ultra_model; $key_env): {
    npm: "@ai-sdk/openai-compatible",
    name: "NCHC RAP Nemotron 3",
    options: {
      baseURL: $rap_base,
      apiKey: ("{env:" + $key_env + "}"),
      headers: {"x-api-key": ("{env:" + $key_env + "}")}
    },
    models: (
      {}
      + (if $super_model != "" then
          {($super_model): {name: "NCHC RAP Nemotron 3 Super"}}
         else {} end)
      + (if $ultra_model != "" then
          {($ultra_model): {name: "NCHC RAP Nemotron 3 Ultra"}}
         else {} end)
    )
  };
  def nim_provider($display; $model): {
    npm: "@ai-sdk/openai-compatible",
    name: $display,
    options: {
      baseURL: $nim_base,
      apiKey: "{env:NVIDIA_NIM_API_KEY}"
    },
    models: {($model): {name: $display}}
  };
  {
    "$schema": "https://opencode.ai/config.json",
    share: "disabled",
    permission: {
      read: "allow",
      grep: "allow",
      glob: "allow",
      bash: "ask",
      edit: "ask",
      webfetch: "ask",
      websearch: "ask",
      "jupyter_*": "ask"
    },
    provider: (
      {}
      + (if ($nemotron_enabled and $rap_base != "" and
             ($nemotron_super_model != "" or $nemotron_ultra_model != "")) then
          {"rap-nemotron": rap_nemotron_provider(
            $nemotron_super_model; $nemotron_ultra_model; $nemotron_key_env)}
         else {} end)
      + (if ($gemma26_enabled and $rap_base != "" and $gemma26_model != "") then
          {"rap-gemma-26b": rap_provider("NCHC RAP Gemma 26B"; $gemma26_model; "RAP_GEMMA_26B_API_KEY")}
         else {} end)
      + (if ($gemma31_enabled and $rap_base != "" and $gemma31_model != "") then
          {"rap-gemma-31b": rap_provider("NCHC RAP Gemma 31B"; $gemma31_model; "RAP_GEMMA_31B_API_KEY")}
         else {} end)
      + (if ($nim_enabled and $nim_base != "" and $nim_model != "") then
          {"nvidia-nim": nim_provider("NVIDIA NIM"; $nim_model)}
         else {} end)
    )
  } + (if $default_model != "" then
         {model: $default_model, small_model: $default_model}
       else {} end)
  ' > "$output_json"

chmod 600 "$output_json"
echo "Rendered OpenCode config: $output_json"
provider_count=$(jq '.provider | length' "$output_json")
if [[ "$provider_count" -eq 0 ]]; then
  echo "No OpenCode model was configured; JupyterLab can start, but LLM features are disabled." >&2
elif ! jq -e '.model != null and .model != ""' "$output_json" >/dev/null; then
  echo "OpenCode providers exist, but no default model was selected." >&2
  exit 1
else
  echo "Default OpenCode model: $(jq -r '.model' "$output_json")"
fi
echo "Optional Zen fallback: opencode/nemotron-3-ultra-free (Free pricing; use /connect with a personal OpenCode Zen API key)."
