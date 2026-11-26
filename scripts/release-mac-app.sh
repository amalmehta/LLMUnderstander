#!/bin/bash
# Builds a universal (Apple silicon + Intel) LLM Understander.app, signs it with the
# Developer ID Application certificate and the hardened runtime, notarizes it with Apple,
# staples the ticket and zips it to build/release/LLM-Understander.zip.
#
#   scripts/release-mac-app.sh               build, sign, notarize, staple, zip
#   scripts/release-mac-app.sh --publish     also create the GitHub release v<MARKETING_VERSION>
#
# Needs, once per Mac:
#   - a "Developer ID Application" certificate for team K25J24URWV in the login keychain
#     (Xcode → Settings → Accounts → Manage Certificates → + → Developer ID Application)
#   - a notarytool keychain profile (the password is typed at the prompt, never stored here):
#       xcrun notarytool store-credentials notary-K25J24URWV --apple-id <Apple ID> --team-id K25J24URWV
set -euo pipefail
cd "$(dirname "$0")/.."

TEAM_ID=K25J24URWV
NOTARY_PROFILE="${NOTARY_PROFILE:-notary-$TEAM_ID}"
PROJECT="LLM Understander.xcodeproj"
SCHEME="LLM Understander"
APP_NAME="LLM Understander"
ZIP_NAME="LLM-Understander.zip"
OUT=build/release

IDENTITY=$(security find-identity -v -p codesigning \
  | awk -F'"' -v t="($TEAM_ID)" '/Developer ID Application: / && index($2, t) {print $2; exit}')
if [ -z "$IDENTITY" ]; then
  echo "No Developer ID Application certificate for team $TEAM_ID in the keychain." >&2
  exit 1
fi
if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
  echo "No working notarytool keychain profile named $NOTARY_PROFILE (see the top of this script)." >&2
  exit 1
fi
VERSION=$(sed -n 's/^ *MARKETING_VERSION: *"\{0,1\}\([0-9.]*\)"\{0,1\}/\1/p' project.yml | head -1)
echo "Building $APP_NAME $VERSION, signed as: $IDENTITY"

rm -rf "$OUT"
mkdir -p "$OUT"
xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration Release -derivedDataPath "$OUT/dd" \
  ARCHS="arm64 x86_64" ONLY_ACTIVE_ARCH=NO \
  CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY="$IDENTITY" DEVELOPMENT_TEAM="$TEAM_ID" \
  ENABLE_HARDENED_RUNTIME=YES CODE_SIGN_INJECT_BASE_ENTITLEMENTS=NO OTHER_CODE_SIGN_FLAGS="--timestamp" \
  build | tail -3

APP="$OUT/$APP_NAME.app"
ditto "$OUT/dd/Build/Products/Release/$APP_NAME.app" "$APP"
lipo -archs "$APP/Contents/MacOS/$APP_NAME"
codesign --verify --deep --strict --verbose=2 "$APP"
codesign -dvv "$APP" 2>&1 | grep -E '^(Authority=Developer ID Application|Timestamp|Runtime Version|flags)'

echo "Submitting to Apple's notary service (usually a few minutes)…"
ditto -c -k --keepParent "$APP" "$OUT/notarize.zip"
xcrun notarytool submit "$OUT/notarize.zip" --keychain-profile "$NOTARY_PROFILE" --wait --timeout 1h \
  --output-format json | tee "$OUT/notary.json"
if ! grep -q '"status" *: *"Accepted"' "$OUT/notary.json"; then
  ID=$(sed -n 's/.*"id" *: *"\([^"]*\)".*/\1/p' "$OUT/notary.json" | head -1)
  [ -n "$ID" ] && xcrun notarytool log "$ID" --keychain-profile "$NOTARY_PROFILE" >&2
  echo "Notarization was not accepted." >&2
  exit 1
fi

xcrun stapler staple "$APP"
xcrun stapler validate "$APP"
spctl -a -t exec -vv "$APP"
rm "$OUT/notarize.zip"
ditto -c -k --keepParent "$APP" "$OUT/$ZIP_NAME"
echo "Ready: $OUT/$ZIP_NAME"

if [ "${1:-}" = "--publish" ]; then
  NOTES="**Download:** \`$ZIP_NAME\` below. It runs on Apple silicon and Intel, macOS 14 or later.

### Install
Unzip the download, move **$APP_NAME** to Applications and open it. It's signed with a Developer ID and notarized by Apple, so it opens like any other app.

The same 18 chapters are also on the web: https://amalmehta.github.io/LLMUnderstander/"
  gh release create "v$VERSION" "$OUT/$ZIP_NAME" --title "$APP_NAME $VERSION" \
    --notes "${RELEASE_NOTES:-$NOTES}" --target "$(git rev-parse HEAD)"
fi
