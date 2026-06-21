#!/usr/bin/env bash

load_env_preserving_overrides() {
  local env_file="$1"
  shift
  [ -f "$env_file" ] || return 0

  declare -A overrides=()
  local name
  for name in "$@"; do
    if [ "${!name+x}" = x ]; then
      overrides["$name"]="${!name}"
    fi
  done

  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a

  for name in "${!overrides[@]}"; do
    export "$name=${overrides[$name]}"
  done
}
