#!/usr/bin/env bash
# patch_chameleon_fix.sh
# Copies custom chameleon files into a venv Transformers install and patches generation/utils.py
# Prints clear SUCCESS/ERROR messages and exits non-zero on failure.

set -euo pipefail

ok()   { printf "\033[1;32m[SUCCESS]\033[0m %s\n" "$*"; }
info() { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERROR]\033[0m %s\n" "$*" >&2; }

# ----- CONFIG -----
BASE_DIR="$HOME/ICoT"
CML_SRC1="$BASE_DIR/processing_chameleon.py"
CML_SRC2="$BASE_DIR/modeling_chameleon.py"

# Detect transformers dir (lib vs lib64)
TF_DIR_CAND1="$BASE_DIR/.venv/lib/python3.10/site-packages/transformers"
TF_DIR_CAND2="$BASE_DIR/.venv/lib64/python3.10/site-packages/transformers"
if [[ -d "$TF_DIR_CAND1" ]]; then
  TRANSFORMERS_DIR="$TF_DIR_CAND1"
elif [[ -d "$TF_DIR_CAND2" ]]; then
  TRANSFORMERS_DIR="$TF_DIR_CAND2"
else
  err "Could not find transformers in: $TF_DIR_CAND1 or $TF_DIR_CAND2"
  exit 1
fi

DEST_CML_DIR="$TRANSFORMERS_DIR/models/chameleon"
UTILS_PY="$TRANSFORMERS_DIR/generation/utils.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="${UTILS_PY}.${STAMP}.bak"

info "Using transformers dir: $TRANSFORMERS_DIR"

# ----- PRE-CHECKS -----
if [[ ! -f "$CML_SRC1" ]]; then err "Missing: $CML_SRC1"; exit 1; fi
if [[ ! -f "$CML_SRC2" ]]; then err "Missing: $CML_SRC2"; exit 1; fi
if [[ ! -f "$UTILS_PY" ]]; then err "Missing: $UTILS_PY"; exit 1; fi
ok "All required paths found."

# ----- ENSURE DEST DIR -----
info "Ensuring destination directory exists: $DEST_CML_DIR"
mkdir -p "$DEST_CML_DIR"
ok "Destination directory ready."

# ----- COPY FILES -----
info "Copying processing_chameleon.py -> $DEST_CML_DIR"
cp "$CML_SRC1" "$DEST_CML_DIR/"
ok "Copied processing_chameleon.py"

info "Copying modeling_chameleon.py -> $DEST_CML_DIR"
cp "$CML_SRC2" "$DEST_CML_DIR/"
ok "Copied modeling_chameleon.py"

# ----- BACKUP UTILS.PY -----
info "Backing up utils.py -> $BACKUP"
cp "$UTILS_PY" "$BACKUP"
ok "Backup created."

# ----- PATCH UTILS.PY (indentation-aware) -----
info "Patching utils.py with indentation-aware block…"
export UTILS_PY  # make available to the embedded Python

python3 - <<'PY'
import io, sys, re, os

utils_path = os.environ["UTILS_PY"]
with io.open(utils_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

marker = "# update generated ids, model inputs, and length for next step"
idx = None
for i, line in enumerate(lines):
    if marker in line:
        idx = i
        break

if idx is None:
    print("[ERROR] Marker line not found in utils.py", file=sys.stderr)
    sys.exit(2)

# Find the line to replace: next non-empty line after the marker
j = idx + 1
while j < len(lines) and lines[j].strip() == "":
    j += 1
if j >= len(lines):
    print("[ERROR] Could not find target line after the marker.", file=sys.stderr)
    sys.exit(3)

target_line = lines[j]
indent = re.match(r"^(\s*)", target_line).group(1)

block = [
    f"{indent}if 'selected_vokens' in outputs and outputs['selected_vokens'] is not None:\n",
    f"{indent}    input_ids = torch.cat([input_ids, outputs['selected_vokens'], next_tokens[:, None]], dim=-1)\n",
    f"{indent}else:\n",
    f"{indent}    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)\n",
]

# Replace exactly the one line after the marker
new_lines = lines[:j] + block + lines[j+1:]

with io.open(utils_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("[SUCCESS] utils.py patched with proper indentation.")
PY

ok "utils.py patched."

# ----- QUICK SYNTAX CHECK -----
info "Import-check: transformers.generation.utils"
python3 - <<'PY'
import sys
try:
    import importlib
    import transformers
    m = importlib.import_module("transformers.generation.utils")
    print("[SUCCESS] transformers.generation.utils imported successfully.")
except Exception as e:
    print("[ERROR] Import failed:", e)
    sys.exit(1)
PY
ok "Import check passed."

# ----- SHOW CONTEXT (optional, first 6 lines after marker) -----
info "Context around patched marker:"
nl -ba "$UTILS_PY" | awk '
/update generated ids, model inputs, and length for next step/ {print; c=1; next}
c>0 {print; c++; if (c>6) exit}
'

# ----- DONE -----
ok "All steps completed successfully."
info "Backup available at: $BACKUP"
