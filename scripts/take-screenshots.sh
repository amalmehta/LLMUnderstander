#!/bin/bash
# Builds the app and retakes every chapter screenshot in docs/screenshots (the README uses them).
# The app trains its tiny model, visits each chapter and saves the window; it needs Screen Recording
# permission for the terminal you run this from (System Settings → Privacy & Security).
set -euo pipefail
cd "$(dirname "$0")/.."
xcodebuild -project "LLM Understander.xcodeproj" -scheme "LLM Understander" -configuration Release \
  -derivedDataPath build/dd build -quiet
pkill -x "LLM Understander" 2>/dev/null || true
"build/dd/Build/Products/Release/LLM Understander.app/Contents/MacOS/LLM Understander" --snapshots docs/screenshots
echo "Saved to docs/screenshots. Check them, then commit."
