#!/bin/sh
# Splash M1 installer for Apple Silicon Macs.
#
#   curl -qfsSL https://github.com/paulouski/splash-m1/releases/latest/download/install.sh | bash
#
# Downloads a release from GitHub Releases, verifies its SHA-256, unpacks it
# under ~/Library/Application Support/Splash M1/app, and writes `splash-m1` to PATH.
# It also installs the desktop app (Splash M1.app) into /Applications, or
# ~/Applications when /Applications is not writable; files fetched by curl carry
# no quarantine flag, so the ad-hoc-signed app opens without Gatekeeper's dialog.
# Running it again upgrades in place; model weights and sessions are untouched.
# Pass `--cli-only` (or `--no-app`) to skip the app:  ... | bash -s -- --cli-only
#
#   SPLASH_VERSION   install this version instead of the latest release
#   SPLASH_REPO      GitHub repository (default: paulouski/splash-m1)
#   SPLASH_BASE_URL  custom asset base, e.g. a private Hugging Face repo or local fixture
#   SPLASH_TOKEN     optional Bearer token for a private custom asset base
#   SPLASH_APP_DIR   where to put Splash M1.app (default: /Applications or ~/Applications)
#   SPLASH_BIN_DIR   where to put `splash-m1` (default: Homebrew bin or ~/.local/bin)
set -eu

REPO=${SPLASH_REPO:-paulouski/splash-m1}
BASE_URL=${SPLASH_BASE_URL:-}
TOKEN=${SPLASH_TOKEN:-}
APP="$HOME/Library/Application Support/Splash M1/app"
MARKER="Splash M1/app/current"

fail() { echo "splash-m1 install: $*" >&2; exit 1; }

want_app=1
for arg in "$@"; do
    case "$arg" in
        --cli-only|--no-app) want_app=0 ;;
        *) fail "unknown option: $arg" ;;
    esac
done

valid_version() {
    case "$1" in
        [A-Za-z0-9]*) ;;
        *) return 1 ;;
    esac
    case "$1" in
        *[!A-Za-z0-9._-]*) return 1 ;;
    esac
    return 0
}

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail "Splash M1 runs on Apple Silicon Macs only."
os=$(sw_vers -productVersion)
major=${os%%.*}
case "$major" in ''|*[!0-9]*) fail "could not read macOS version: $os.";; esac
[ "$major" -ge 15 ] 2>/dev/null || fail "Splash M1 requires macOS 15.0 or newer; this Mac runs $os."
command -v curl >/dev/null 2>&1 || fail "curl is required."
mem=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
[ "$mem" -ge 30000000000 ] 2>/dev/null || echo "Note: this Mac has under 32 GB of memory; only the 2-bit Prism checkpoint (Ternary Bonsai 2) is supported." >&2
version=${SPLASH_VERSION:-}
[ -z "$version" ] || valid_version "$version" || fail "invalid release version."
if [ -z "$BASE_URL" ]; then
    case "$REPO" in */*) ;;
        *) fail "invalid GitHub repository; expected owner/repository.";;
    esac
    owner=${REPO%%/*}
    repository=${REPO#*/}
    case "$owner$repository" in *[!A-Za-z0-9_.-]*) fail "invalid GitHub repository.";; esac
    [ -n "$owner" ] && [ -n "$repository" ] || fail "invalid GitHub repository."
    case "$repository" in */*) fail "invalid GitHub repository.";; esac
fi

dir=${SPLASH_BIN_DIR:-}
[ -n "$dir" ] || for candidate in /opt/homebrew/bin /usr/local/bin; do
    if [ -d "$candidate" ] && [ -w "$candidate" ]; then dir=$candidate; break; fi
done
[ -n "$dir" ] || dir="$HOME/.local/bin"
wrapper="$dir/splash-m1"
if [ "$want_app" = 1 ]; then
    appdir=${SPLASH_APP_DIR:-}
    if [ -z "$appdir" ]; then
        if [ -w /Applications ]; then appdir=/Applications; else appdir="$HOME/Applications"; fi
    fi
    mkdir -p "$appdir" || fail "could not create $appdir."
    [ -w "$appdir" ] || fail "$appdir is not writable; set SPLASH_APP_DIR."
    dest="$appdir/Splash M1.app"
    if pgrep -x SplashM1 >/dev/null 2>&1; then fail "quit Splash M1 before installing."; fi
fi
if [ -e "$wrapper" ] && ! grep -q "$MARKER" "$wrapper" 2>/dev/null; then
    fail "$wrapper exists and was not created by this installer; remove it first."
fi

fetch() {
    if [ -n "$TOKEN" ]; then
        curl -q -fsSL --retry 3 --config "$work/curl.conf" -o "$2" "$BASE_URL/$1"
    else
        curl -q -fsSL --retry 3 -o "$2" "$BASE_URL/$1"
    fi || fail "could not download release asset $1."
}

work=$(mktemp -d "${TMPDIR:-/tmp}/splash-install.XXXXXX")
stage=
trap 'rm -rf "$work" ${stage:+"$stage"}' EXIT
if [ -n "$TOKEN" ]; then
    case "$TOKEN" in *[!A-Za-z0-9_./~+=-]*) fail "invalid access token format.";; esac
    (umask 077; printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" > "$work/curl.conf")
fi

if [ -z "$version" ]; then
    [ -n "$BASE_URL" ] || BASE_URL="https://github.com/$REPO/releases/latest/download"
    BASE_URL=${BASE_URL%/}
    fetch latest "$work/latest"
    version=$(tr -d '[:space:]' < "$work/latest")
    [ -n "$version" ] || fail "the release index is empty."
    valid_version "$version" || fail "invalid release version in the latest asset."
    if [ -z "${SPLASH_BASE_URL:-}" ]; then
        BASE_URL="https://github.com/$REPO/releases/download/$version"
    fi
elif [ -z "$BASE_URL" ]; then
    BASE_URL="https://github.com/$REPO/releases/download/$version"
fi
BASE_URL=${BASE_URL%/}
name="splash-m1-$version-arm64-macos15"

mkdir -p "$APP"
candidate="$APP/$name"
if [ -f "$candidate/release.json" ]; then
    echo "Splash M1 $version is already downloaded."
else
    echo "Downloading Splash M1 $version..."
    fetch "$name.tar.gz" "$work/$name.tar.gz"
    fetch "$name.tar.gz.sha256" "$work/$name.tar.gz.sha256"
    expected=$(cut -d' ' -f1 < "$work/$name.tar.gz.sha256")
    actual=$(shasum -a 256 "$work/$name.tar.gz" | cut -d' ' -f1)
    [ -n "$expected" ] && [ "$expected" = "$actual" ] || fail "checksum mismatch for $name.tar.gz."
    mkdir "$work/extract"
    tar -xzf "$work/$name.tar.gz" -C "$work/extract"
    [ -f "$work/extract/$name/release.json" ] || fail "unexpected archive layout."
    candidate="$work/extract/$name"
fi
# Validate before touching an installed version or waiting on its lifecycle lock.
if ! PYTHONDONTWRITEBYTECODE=1 "$candidate/python/bin/python3" -u \
        "$candidate/install/launcher.py" --help >/dev/null 2>&1; then
    fail "Splash M1 $version fails 'splash-m1 --help'; nothing was changed."
fi
if [ "$want_app" = 1 ]; then
    app="$name-app.zip"
    fetch "$app" "$work/$app"
    fetch "$app.sha256" "$work/$app.sha256"
    expected=$(cut -d' ' -f1 < "$work/$app.sha256")
    actual=$(shasum -a 256 "$work/$app" | cut -d' ' -f1)
    [ -n "$expected" ] && [ "$expected" = "$actual" ] || fail "checksum mismatch for $app."
    # Stage beside the destination so the final moves are same-volume renames.
    stage=$(mktemp -d "$appdir/.splash-app.XXXXXX") || fail "could not stage the app in $appdir."
    ditto -x -k "$work/$app" "$stage" || fail "could not extract $app."
    [ -f "$stage/Splash M1.app/Contents/Info.plist" ] || fail "unexpected app archive layout."
    xattr -dr com.apple.quarantine "$stage/Splash M1.app" 2>/dev/null || true
fi
cat > "$work/apply.sh" <<'INSTALL'
set -eu
APP=$1
name=$2
candidate=$3
wrapper=$4
dir=${wrapper%/*}
fail() { echo "splash-m1 install: $*" >&2; exit 1; }
if [ "$candidate" != "$APP/$name" ] && [ ! -f "$APP/$name/release.json" ]; then
    rm -rf "$APP/$name"
    mv "$candidate" "$APP/$name"
fi
# Stage the wrapper before switching current; restore the old link if the
# final rename fails. Remove older versions only after both changes succeed.
mkdir -p "$dir"
cat > "$wrapper.tmp" <<WRAPPER || fail "could not write $wrapper; Splash $name was not installed."
#!/bin/sh
export PYTHONDONTWRITEBYTECODE=1
exec "$APP/current/python/bin/python3" -u "$APP/current/install/launcher.py" "\$@"
WRAPPER
chmod 0755 "$wrapper.tmp"
previous=$(readlink "$APP/current" 2>/dev/null || true)
ln -sfn "$APP/$name" "$APP/current"
if ! mv -f "$wrapper.tmp" "$wrapper"; then
    if [ -n "$previous" ]; then ln -sfn "$previous" "$APP/current"; else rm -f "$APP/current"; fi
    rm -f "$wrapper.tmp"
    fail "could not write $wrapper; Splash $name was not installed."
fi
for old in "$APP"/splash-m1-*-arm64-macos15; do
    [ -d "$old" ] && [ "$old" != "$APP/$name" ] && rm -rf "$old"
done
exit 0
INSTALL
# Use the validated bundled Python, so the installer needs no system Python.
# The descriptor survives exec and protects every switch, replacement and prune.
PYTHONDONTWRITEBYTECODE=1 "$candidate/python/bin/python3" - \
    "$APP/../runtime/serve.lock" "$work/apply.sh" "$APP" "$name" "$candidate" "$wrapper" <<'PYTHON'
import fcntl
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a+") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("splash-m1 install: stop the running Splash M1 server before upgrading; another installer may also hold the lock.")
    os.set_inheritable(lock.fileno(), True)
    os.execv("/bin/sh", ["/bin/sh", *sys.argv[2:]])
PYTHON

if [ "$want_app" = 1 ]; then
    # Swap in the staged app; put the old one back if the move fails.
    if [ -e "$dest" ]; then
        mv "$dest" "$stage/old.app" || fail "could not replace $dest; the CLI was installed."
        mv "$stage/Splash M1.app" "$dest" || { mv "$stage/old.app" "$dest"; fail "could not install $dest; the CLI was installed."; }
    else
        mv "$stage/Splash M1.app" "$dest" || fail "could not install $dest; the CLI was installed."
    fi
fi

echo
echo "Splash M1 $version installed: $wrapper"
if [ "$want_app" = 1 ]; then
    echo "Desktop app installed: $dest"
    echo "  Open it with:  open -a \"Splash M1\"   (or open \"$dest\")"
fi
case ":$PATH:" in
    *":$dir:"*) ;;
    *) echo "Add it to your PATH first:  export PATH=\"$dir:\$PATH\"" ;;
esac
echo "  splash-m1 serve --model mlx-community/Qwen3.8-27B-4bit --language-only --max-context 32K"
if [ -f "$APP/current/install/completions/splash.bash" ] && [ -f "$APP/current/install/completions/_splash" ]; then
    echo "  Optional shell completion (Zsh needs compinit initialized):"
    echo '    Bash: source "$HOME/Library/Application Support/Splash M1/app/current/install/completions/splash.bash"'
    echo '    Zsh:  source "$HOME/Library/Application Support/Splash M1/app/current/install/completions/_splash"'
fi
echo "  Upgrade: run this installer again.  Uninstall: rm -rf \"$APP\" \"$wrapper\""
