#!/usr/bin/env bash
# Publish GBC book to vadimsokolov.github.io
# Usage: ./publish.sh [--skip-render]

set -euo pipefail

BOOK_DIR="/Users/vsokolov/Dropbox/papers/gbc-book"
WWW_DIR="/Users/vsokolov/Dropbox/www"
DEST="$WWW_DIR/html/gbc-book"

cd "$BOOK_DIR"

# Step 1: render
if [[ "${1:-}" != "--skip-render" ]]; then
  echo "==> Rendering book (HTML)..."
  quarto render --to html
else
  echo "==> Skipping render (--skip-render)"
fi

if [[ ! -f "$BOOK_DIR/_book/index.html" ]]; then
  echo "ERROR: _book/index.html not found. Render may have failed." >&2
  exit 1
fi

# Step 2: sync to website html folder
echo "==> Syncing _book/ -> $DEST"
mkdir -p "$DEST"
rsync -a --delete "$BOOK_DIR/_book/" "$DEST/"

# Step 3: publish via upd.sh
echo "==> Publishing website..."
cd "$WWW_DIR"
./upd.sh
echo "==> Done. Book live at https://vadimsokolov.github.io/html/gbc-book/"
