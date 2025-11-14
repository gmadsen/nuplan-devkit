#!/usr/bin/env bash
# ABOUTME: nuPlan dataset downloader with progress tracking and resume capability
# ABOUTME: Supports mini (50GB) and full (10TB) datasets with persistent state management
set -euo pipefail
IFS=$'\n\t'

###############################################################################
# COLOR & FORMATTING UTILITIES
###############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info() {
  echo -e "${CYAN}[$(date +'%H:%M:%S')]${NC} $*"
}

log_success() {
  echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓${NC} $*"
}

log_warn() {
  echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠${NC} $*"
}

log_error() {
  echo -e "${RED}[$(date +'%H:%M:%S')] ✗${NC} $*"
}

log_section() {
  echo -e "\n${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}${BLUE} $*${NC}"
  echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}\n"
}

format_bytes() {
  local bytes=$1
  if (( bytes < 1024 )); then
    echo "${bytes}B"
  elif (( bytes < 1048576 )); then
    echo "$((bytes / 1024))KB"
  elif (( bytes < 1073741824 )); then
    echo "$((bytes / 1048576))MB"
  else
    printf "%.2fGB" "$(echo "scale=2; $bytes / 1073741824" | bc)"
  fi
}

###############################################################################
# CONFIGURATION
###############################################################################

BASE="https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
MAPS_ZIP="$BASE/nuplan-maps-v1.1.zip"
MINI_DB_ZIP="$BASE/nuplan-v1.1_mini.zip"
SENSORS_PREFIX_MINI="$BASE/sensor_blobs/mini_set/"
SENSORS_PREFIX_FULL="$BASE/sensor_blobs/"

REQUIRED_SPACE_MINI_GB=300
REQUIRED_SPACE_FULL_GB=11000  # ~10TB + overhead
MAX_RETRIES=5

SCRIPT_START=$(date +%s)

###############################################################################
# STATE MANAGEMENT
###############################################################################

STATE_FILE=""

init_state_file() {
  STATE_FILE="$NUPLAN_DATA_ROOT/.nuplan_dl_state.json"

  if [[ ! -f "$STATE_FILE" ]]; then
    log_info "Creating new state file: $STATE_FILE"
    cat > "$STATE_FILE" <<EOF
{
  "version": "1.0",
  "dataset_mode": "none",
  "completed_files": [],
  "total_downloaded_bytes": 0,
  "last_updated": "$(date -Iseconds)"
}
EOF
  else
    log_info "Resuming from existing state file"
  fi
}

get_state_value() {
  local key=$1
  python3 -c "import json; print(json.load(open('$STATE_FILE'))['$key'])" 2>/dev/null || echo ""
}

update_state() {
  local key=$1
  local value=$2
  python3 <<EOF
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
state['$key'] = $value
state['last_updated'] = '$(date -Iseconds)'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
EOF
}

mark_file_completed() {
  local filename=$1
  local filesize=$2
  python3 <<EOF
import json
with open('$STATE_FILE', 'r') as f:
    state = json.load(f)
if '$filename' not in state['completed_files']:
    state['completed_files'].append('$filename')
    state['total_downloaded_bytes'] += $filesize
state['last_updated'] = '$(date -Iseconds)'
with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)
EOF
}

is_file_completed() {
  local filename=$1
  python3 -c "import json; print('$filename' in json.load(open('$STATE_FILE'))['completed_files'])"
}

get_completed_count() {
  python3 -c "import json; print(len(json.load(open('$STATE_FILE'))['completed_files']))"
}

get_total_downloaded() {
  python3 -c "import json; print(json.load(open('$STATE_FILE'))['total_downloaded_bytes'])"
}

###############################################################################
# STATUS COMMAND
###############################################################################

show_status() {
  if [[ ! -f "$STATE_FILE" ]]; then
    log_error "No state file found. Dataset download not started."
    exit 1
  fi

  log_section "nuPlan Dataset Download Status"

  local mode=$(get_state_value "dataset_mode")
  local completed=$(get_completed_count)
  local total_bytes=$(get_total_downloaded)
  local last_update=$(get_state_value "last_updated")

  echo "Dataset Mode:      $mode"
  echo "Completed Files:   $completed"
  echo "Total Downloaded:  $(format_bytes "$total_bytes")"
  echo "Last Updated:      $last_update"
  echo ""

  if (( completed > 0 )); then
    log_info "Completed files:"
    python3 -c "import json; print('\n'.join(['  - ' + f for f in json.load(open('$STATE_FILE'))['completed_files']]))"
  fi

  exit 0
}

###############################################################################
# 0. PARSE ARGUMENTS
###############################################################################

if [[ "${1:-}" == "status" ]]; then
  if [[ -z "${NUPLAN_DATA_ROOT:-}" ]]; then
    log_error "NUPLAN_DATA_ROOT not set."
    echo 'Example: export NUPLAN_DATA_ROOT="/mnt/tower_data/datasets/nuplan/dataset"'
    exit 1
  fi
  STATE_FILE="$NUPLAN_DATA_ROOT/.nuplan_dl_state.json"
  show_status
fi

###############################################################################
# 1. REQUIRE NUPLAN_DATA_ROOT
###############################################################################

if [[ -z "${NUPLAN_DATA_ROOT:-}" ]]; then
  log_error "NUPLAN_DATA_ROOT not set."
  echo 'Example: export NUPLAN_DATA_ROOT="/mnt/tower_data/datasets/nuplan/dataset"'
  exit 1
fi

log_info "Using NUPLAN_DATA_ROOT = $NUPLAN_DATA_ROOT"
echo

init_state_file

###############################################################################
# 2. DATASET SELECTION MENU
###############################################################################

log_section "Dataset Selection"

echo "Choose download mode:"
echo ""
echo "  1) Mini dataset only (~50GB)"
echo "     - Good for tutorials and learning"
echo "     - Contains subset of scenarios"
echo ""
echo "  2) Full dataset (~10TB)"
echo "     - Complete dataset for research"
echo "     - All scenarios from all cities"
echo ""
echo "  3) Mini first, then full later"
echo "     - Start with mini for quick setup"
echo "     - Can resume to download full later"
echo ""

read -r -p "Select option [1-3]: " dataset_choice

case $dataset_choice in
  1)
    DATASET_MODE="mini"
    REQUIRED_SPACE_GB=$REQUIRED_SPACE_MINI_GB
    log_info "Selected: Mini dataset (~50GB)"
    ;;
  2)
    DATASET_MODE="full"
    REQUIRED_SPACE_GB=$REQUIRED_SPACE_FULL_GB
    log_info "Selected: Full dataset (~10TB)"
    ;;
  3)
    DATASET_MODE="mini"
    REQUIRED_SPACE_GB=$REQUIRED_SPACE_MINI_GB
    log_info "Selected: Mini dataset (can expand to full later)"
    ;;
  *)
    log_error "Invalid selection"
    exit 1
    ;;
esac

update_state "dataset_mode" "\"$DATASET_MODE\""
echo

###############################################################################
# 3. VALIDATE CIFS MOUNT
###############################################################################

log_section "Validating Destination"

mount_type=$(stat -f -c %T "$NUPLAN_DATA_ROOT")
if [[ "$mount_type" != "cifs" ]]; then
  log_warn "Destination is not a CIFS mount!"
  log_warn "Detected FS: $mount_type"
  log_warn "If this is gvfs/fuse: DO NOT PROCEED."
  read -r -p "Continue anyway? [y/N] " x
  [[ "$x" =~ ^[yY] ]] || exit 1
fi

log_success "Mount type: $mount_type"

###############################################################################
# 4. CHECK FREE SPACE
###############################################################################

log_section "Checking Free Space"

available_gb=$(df -BG "$NUPLAN_DATA_ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}')
if (( available_gb < REQUIRED_SPACE_GB )); then
  log_error "Not enough free space."
  echo "Required: ${REQUIRED_SPACE_GB}GB, Available: ${available_gb}GB"
  exit 1
fi

log_success "Free space OK: ${available_gb}GB (required: ${REQUIRED_SPACE_GB}GB)"

###############################################################################
# 5. PREPARE DIRECTORIES
###############################################################################

log_section "Preparing Directories"

ZIPS_DIR="$NUPLAN_DATA_ROOT/_zips"
mkdir -p "$ZIPS_DIR"
log_success "Created zips directory: $ZIPS_DIR"

URLS="$ZIPS_DIR/urls.txt"
rm -f "$URLS"

###############################################################################
# 6. BUILD URL LIST
###############################################################################

log_section "Building Download List"

echo "$MAPS_ZIP" >> "$URLS"
echo "$MINI_DB_ZIP" >> "$URLS"

log_info "Fetching sensor-blob zip list..."

if [[ "$DATASET_MODE" == "mini" ]]; then
  SENSOR_LIST=$(curl -s "$SENSORS_PREFIX_MINI" | grep -o 'nuplan-v1\.1_mini[^"]*\.zip' || true)
  SENSORS_PREFIX="$SENSORS_PREFIX_MINI"
else
  # Full dataset - fetch all sensor blobs
  SENSOR_LIST=$(curl -s "$SENSORS_PREFIX_FULL" | grep -o 'nuplan-v1\.1[^"]*\.zip' | grep -v '_mini' || true)
  SENSORS_PREFIX="$SENSORS_PREFIX_FULL"
fi

if [[ -z "$SENSOR_LIST" ]]; then
  log_error "Could not parse sensor zips."
  exit 1
fi

log_success "Found sensor zip files:"
echo "$SENSOR_LIST" | sed 's/^/  - /' | head -20
if [[ $(echo "$SENSOR_LIST" | wc -l) -gt 20 ]]; then
  echo "  ... and $(($(echo "$SENSOR_LIST" | wc -l) - 20)) more"
fi

while read -r f; do
  echo "$SENSORS_PREFIX$f" >> "$URLS"
done <<< "$SENSOR_LIST"

TOTAL_FILES=$(wc -l < "$URLS")
log_success "Built download list: $TOTAL_FILES files total"

# Show already completed files
COMPLETED=$(get_completed_count)
if (( COMPLETED > 0 )); then
  log_info "Already completed: $COMPLETED files ($(format_bytes "$(get_total_downloaded)"))"
  log_info "Remaining: $((TOTAL_FILES - COMPLETED)) files"
fi

echo
read -r -p "Begin download? [y/N] " xx
[[ "$xx" =~ ^[yY] ]] || exit 1

###############################################################################
# 7. DOWNLOAD FUNCTIONS WITH RETRY & LOGGING
###############################################################################

download_with_retries() {
  local url="$1"
  local outfile="$2"
  local current_num="$3"
  local total_num="$4"

  # Check if already completed
  if [[ $(is_file_completed "$outfile") == "True" ]]; then
    log_info "[$current_num/$total_num] Skipping (already completed): $outfile"
    return 0
  fi

  # Check if file exists and is valid
  if [[ -s "$ZIPS_DIR/$outfile" ]]; then
    if unzip -t "$ZIPS_DIR/$outfile" > /dev/null 2>&1; then
      local filesize=$(stat -c%s "$ZIPS_DIR/$outfile")
      log_success "[$current_num/$total_num] Already downloaded & verified: $outfile ($(format_bytes "$filesize"))"
      mark_file_completed "$outfile" "$filesize"
      return 0
    else
      log_warn "[$current_num/$total_num] Existing file corrupt, re-downloading: $outfile"
      rm -f "$ZIPS_DIR/$outfile"
    fi
  fi

  for ((i=1; i<=MAX_RETRIES; i++)); do
    log_info "[$current_num/$total_num] Downloading (attempt $i/$MAX_RETRIES): $outfile"

    if aria2c \
      --allow-overwrite=false \
      --continue=true \
      --file-allocation=falloc \
      --enable-http-pipelining=true \
      --max-connection-per-server=16 \
      --max-concurrent-downloads=1 \
      --min-split-size=4M \
      --split=16 \
      --summary-interval=10 \
      --console-log-level=notice \
      --disable-ipv6=true \
      --dir="$ZIPS_DIR" \
      --out="$outfile" \
      "$url"; then

      # Verify download
      if [[ -s "$ZIPS_DIR/$outfile" ]]; then
        local filesize=$(stat -c%s "$ZIPS_DIR/$outfile")
        log_success "[$current_num/$total_num] Download OK: $outfile ($(format_bytes "$filesize"))"
        mark_file_completed "$outfile" "$filesize"
        return 0
      fi
    fi

    log_warn "[$current_num/$total_num] Download failed, retry $i/$MAX_RETRIES"
    sleep 2
  done

  log_error "[$current_num/$total_num] Download failed permanently: $outfile"
  exit 1
}

###############################################################################
# 8. DOWNLOAD ALL FILES WITH PROGRESS TRACKING
###############################################################################

log_section "Downloading Files (${TOTAL_FILES} total)"

current=0
while read -r url; do
  ((current++))
  fname=$(basename "$url")
  download_with_retries "$url" "$fname" "$current" "$TOTAL_FILES"
done < "$URLS"

log_success "All files downloaded!"

###############################################################################
# 9. INTEGRITY VERIFICATION
###############################################################################

log_section "Verifying Integrity"

verify_count=0
for z in "$ZIPS_DIR"/*.zip; do
  ((verify_count++))
  fname=$(basename "$z")

  log_info "[$verify_count/$TOTAL_FILES] Verifying: $fname"

  if ! unzip -t "$z" > /dev/null 2>&1; then
    log_error "ZIP integrity check failed: $fname"
    log_info "Re-downloading..."

    # Find the URL for this file
    url=$(grep "/$fname\$" "$URLS" | head -1)
    if [[ -z "$url" ]]; then
      log_error "Could not find URL for $fname"
      exit 1
    fi

    rm -f "$z"
    download_with_retries "$url" "$fname" "$verify_count" "$TOTAL_FILES"

    # Verify again
    if ! unzip -t "$z" > /dev/null 2>&1; then
      log_error "Re-downloaded file still corrupt: $fname"
      exit 1
    fi
  fi

  log_success "[$verify_count/$TOTAL_FILES] Verified: $fname"
done

log_success "All files verified!"

###############################################################################
# 10. EXTRACT FILES
###############################################################################

log_section "Extracting Archives"

extract_count=0
for z in "$ZIPS_DIR"/*.zip; do
  ((extract_count++))
  fname=$(basename "$z")
  log_info "[$extract_count/$TOTAL_FILES] Extracting: $fname"
  unzip -n -q "$z" -d "$NUPLAN_DATA_ROOT"
  log_success "[$extract_count/$TOTAL_FILES] Extracted: $fname"
done

###############################################################################
# 11. SUMMARY
###############################################################################

SCRIPT_END=$(date +%s)
ELAPSED=$((SCRIPT_END - SCRIPT_START))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

TOTAL_SIZE=$(get_total_downloaded)

log_section "Download Complete! 🎉"

echo -e "${GREEN}${BOLD}SUCCESS!${NC} nuPlan $DATASET_MODE dataset downloaded"
echo ""
echo "Summary:"
echo "  Files Downloaded:  $TOTAL_FILES"
echo "  Total Size:        $(format_bytes "$TOTAL_SIZE")"
echo "  Time Elapsed:      ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  Location:          $NUPLAN_DATA_ROOT"
echo ""
echo "Next steps:"
echo "  1. Verify with:    ls -la $NUPLAN_DATA_ROOT"
echo "  2. Check maps:     ls $NUPLAN_DATA_ROOT/maps/"
echo "  3. Check DB:       ls $NUPLAN_DATA_ROOT/nuplan-v1.1/"
echo "  4. Run tutorials:  just tutorial"
echo ""
echo "To check status later: $(basename "$0") status"
echo ""

if [[ "$dataset_choice" == "3" ]]; then
  log_info "To upgrade to full dataset later, re-run this script and select option 2"
fi
