#!/usr/bin/env bash

# Smarter deploy script for "oxi"
# - Uses rsync instead of tar/scp
# - Preserves remote target/ for incremental cargo builds
# - Does NOT delete the remote project directory, so you can stay cd'd in
# - Optional: remote cargo build, systemd restart, custom commands, dry-run

set -euo pipefail

# Defaults (can be overridden via flags or .env)
REMOTE_HOST=${REMOTE_HOST:-"ubuntu@$REMOTE_IP"}
REMOTE_DIR=${REMOTE_DIR:-"/home/ubuntu/oxi"}
LOCAL_DIR=${LOCAL_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"}

# Behavior flags
DO_DELETE=false          # delete extraneous files on remote (safe; respects excludes)
DO_DRY_RUN=false         # preview rsync actions only
DO_BUILD=false           # run cargo build on remote after sync
SYSTEMD_SERVICE=""       # if set, restart this systemd service after build
REMOTE_CMD=""            # if set, run this command after sync/build

# Read .env if present (allows overriding REMOTE_HOST/REMOTE_DIR/etc.)
if [[ -f "${LOCAL_DIR}/.env" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_DIR}/.env"
fi

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Sync the local project to the remote host without deleting the app directory or the remote target/.

Options:
  -h <host>         Remote SSH host (default: ${REMOTE_HOST})
  -d <dir>          Remote project directory (default: ${REMOTE_DIR})
  -n                Dry run (preview rsync changes)
  -N                No delete (do not remove extraneous remote files)
  -b                After sync, run: cargo build --release --bin oxi on remote
  -s <service>      After build, restart this systemd service on remote
  -c <command>      After sync/build, run this shell command on remote (in project dir)
  -S <ssh_opts>     Extra SSH options (quoted) e.g. -S "-o ProxyJump=..."
  -x <pattern>      Extra rsync exclude pattern (can be provided multiple times)
  --                End of options

Examples:
  $(basename "$0")
  $(basename "$0") -b -s oxi.service
  $(basename "$0") -n -N
  $(basename "$0") -c 'cargo run --release --bin oxi -- --help'
EOF
}

EXTRA_EXCLUDES=()

while getopts ":h:d:nNbS:s:c:x:-:" opt; do
  case "$opt" in
    h) REMOTE_HOST="$OPTARG" ;;
    d) REMOTE_DIR="$OPTARG" ;;
    n) DO_DRY_RUN=true ;;
    N) DO_DELETE=false ;;
    b) DO_BUILD=true ;;
    s) SYSTEMD_SERVICE="$OPTARG" ;;
    c) REMOTE_CMD="$OPTARG" ;;
    S) SSH_OPTS="$SSH_OPTS $OPTARG" ;;
    x) EXTRA_EXCLUDES+=("$OPTARG") ;;
    -) case "$OPTARG" in
         help) usage; exit 0 ;;
         "") break ;;
         *) echo "Unknown long option --$OPTARG" >&2; usage; exit 2 ;;
       esac ;;
    \?) usage; exit 2 ;;
    :) echo "Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND-1)) || true

echo "Deploying to ${REMOTE_HOST}:${REMOTE_DIR}"

# Pre-flight checks
command -v rsync >/dev/null || { echo "rsync not found on local machine" >&2; exit 1; }
if ! ssh ${SSH_OPTS} "${REMOTE_HOST}" "command -v rsync >/dev/null"; then
  echo "rsync not found on remote. Install and retry, e.g.:" >&2
  echo "  ssh ${REMOTE_HOST} 'sudo apt-get update && sudo apt-get install -y rsync'" >&2
  exit 1
fi

# Ensure remote directory exists, but don't delete it
ssh ${SSH_OPTS} "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"

# Rsync options
RSYNC_OPTS=(
  -az
  --progress
  --stats
  --human-readable
  --filter ":- .gitignore"   # respect .gitignore
  --exclude ".git"
  --exclude "target"         # preserve remote target/ for incremental builds
  --exclude "checkpoints"    # preserve remote checkpoints/
  --exclude ".venv"
  --exclude "__pycache__"
  --exclude "*.log"
  --exclude "*.pyc"
  --exclude "model"
  --exclude "node_modules"
  --exclude "cubecl.log"
  --exclude "tests/data"
  --exclude "db.sqlite3"
)

if [ "${EXTRA_EXCLUDES+x}" = x ] && [ "${#EXTRA_EXCLUDES[@]}" -gt 0 ]; then
  for patt in "${EXTRA_EXCLUDES[@]}"; do
    RSYNC_OPTS+=( --exclude "$patt" )
  done
fi

if [ "${DO_DELETE}" = true ]; then
  RSYNC_OPTS+=( --delete )
fi
if [ "${DO_DRY_RUN}" = true ]; then
  RSYNC_OPTS+=( --dry-run )
fi

# Prefer SSH multiplexing for speed if available
RSYNC_RSH="ssh ${SSH_OPTS} -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=600"

echo "Syncing project (rsync)..."
rsync "${RSYNC_OPTS[@]}" -e "${RSYNC_RSH}" \
  "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

if [ "${DO_BUILD}" = true ]; then
  echo "Building on remote (cargo build --release --bin oxi)..."
  ssh ${SSH_OPTS} -t "${REMOTE_HOST}" "set -euo pipefail; cd '${REMOTE_DIR}'; cargo build --release --bin oxi"
fi

if [[ -n "${SYSTEMD_SERVICE}" ]]; then
  echo "Restarting systemd service: ${SYSTEMD_SERVICE}"
  ssh ${SSH_OPTS} -t "${REMOTE_HOST}" "sudo systemctl daemon-reload; sudo systemctl restart '${SYSTEMD_SERVICE}'; sudo systemctl is-active '${SYSTEMD_SERVICE}' >/dev/null && echo 'Service ${SYSTEMD_SERVICE} is active.' || (echo 'Service failed' >&2; exit 1)"
fi

if [[ -n "${REMOTE_CMD}" ]]; then
  echo "Running remote command: ${REMOTE_CMD}"
  ssh ${SSH_OPTS} -t "${REMOTE_HOST}" "set -euo pipefail; cd '${REMOTE_DIR}'; ${REMOTE_CMD}"
fi

echo "Done."
