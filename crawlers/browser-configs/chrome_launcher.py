#!/usr/bin/env python3
"""Launch a genuine Google Chrome binary with CDP for the RQ2 environment grid.

Configs
-------
instrumented     -- not launched here; Browser-Use starts its own Chromium.
chrome_incognito -- real Chrome, incognito window, throwaway user-data-dir.
chrome_cold      -- real Chrome Guest session (``--guest``), throwaway user-data-dir.
chrome_full      -- real Chrome signed into the research Google account
                    (``--profile-directory=Profile 6``) on a dedicated copy
                    of the parent user-data-dir.

The launcher never touches a Chrome instance it did not start. Daily Chrome
at ~/.config/google-chrome can keep running while incognito/guest trials use
a separate --user-data-dir. The full-profile cell needs either a dedicated
copy (``--prepare-full-profile``) or exclusive access to the source dir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CHROME_BIN = os.getenv("RQ2_CHROME_BIN", "").strip() or shutil.which(
    "google-chrome"
) or shutil.which("google-chrome-stable") or shutil.which("chromium-browser")
DEFAULT_SOURCE_PROFILE = Path(
    os.getenv("RQ2_CHROME_SOURCE_PROFILE", str(Path.home() / ".config" / "google-chrome"))
).expanduser()
DEFAULT_FULL_PROFILE = Path(
    os.getenv(
        "RQ2_CHROME_FULL_PROFILE",
        str(ROOT / "profiles" / "chrome-full-research"),
    )
).expanduser()
DEFAULT_PROFILE_DIRECTORY = (
    os.getenv("RQ2_CHROME_PROFILE_DIRECTORY", "Profile 6").strip() or "Profile 6"
)
DEFAULT_HOST = "127.0.0.1"
CDP_WAIT_SECONDS = float(os.getenv("RQ2_CDP_WAIT", "15"))

CHROME_CONFIGS = ("chrome_incognito", "chrome_cold", "chrome_full")


@dataclass
class ChromeHandle:
    config: str
    pid: int
    cdp_url: str
    port: int
    user_data_dir: str
    profile_directory: str | None
    incognito: bool
    guest: bool
    chrome_bin: str
    launched_utc: str
    log_path: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def find_free_port(host: str = DEFAULT_HOST, start: int = 9222, span: int = 80) -> int:
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free CDP port in {start}-{start + span - 1}")


def cdp_ready(cdp_url: str, timeout: float = 1.5) -> dict | None:
    version_url = cdp_url.rstrip("/") + "/json/version"
    try:
        with urllib.request.urlopen(version_url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("webSocketDebuggerUrl"):
            return payload
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    return None


def wait_for_cdp(cdp_url: str, timeout: float = CDP_WAIT_SECONDS) -> dict:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        info = cdp_ready(cdp_url)
        if info:
            return info
        last_error = f"CDP not ready at {cdp_url}"
        time.sleep(0.25)
    raise TimeoutError(last_error or f"Timed out waiting for {cdp_url}")


def profile_is_locked(user_data_dir: Path) -> bool:
    for name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
        if (user_data_dir / name).exists():
            return True
    return False


def profile_has_live_process(user_data_dir: Path) -> bool:
    live = subprocess.run(
        ["pgrep", "-f", f"--user-data-dir={user_data_dir}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return live.returncode == 0


def clear_stale_profile_lock(user_data_dir: Path) -> bool:
    """Remove Singleton* files if no process holds this user-data-dir."""
    if not profile_is_locked(user_data_dir):
        return False
    if profile_has_live_process(user_data_dir):
        return False
    for name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
        path = user_data_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()
    print(f"[chrome] cleared stale profile lock in {user_data_dir}", flush=True)
    return True


def default_chrome_bin() -> str:
    if not DEFAULT_CHROME_BIN:
        raise SystemExit(
            "Could not find google-chrome. Set RQ2_CHROME_BIN to the binary path."
        )
    return DEFAULT_CHROME_BIN


def build_chrome_args(
    *,
    chrome_bin: str,
    port: int,
    user_data_dir: Path,
    incognito: bool,
    guest: bool = False,
    profile_directory: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    args = [
        chrome_bin,
        f"--remote-debugging-port={port}",
        f"--remote-debugging-address={DEFAULT_HOST}",
        "--remote-allow-origins=*",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-session-crashed-bubble",
        "--hide-crash-restore-bubble",
        "--start-maximized",
    ]
    if profile_directory:
        args.append(f"--profile-directory={profile_directory}")
    if incognito:
        args.append("--incognito")
    if guest:
        args.append("--guest")
    if extra_args:
        args.extend(extra_args)
    args.append("about:blank")
    return args


def resolve_user_data_dir(config: str, persist_dir: Path | None = None) -> Path:
    if config == "chrome_full":
        return Path(persist_dir or DEFAULT_FULL_PROFILE).expanduser().resolve()
    prefix = {
        "chrome_incognito": "rq2-chrome-incognito-",
        "chrome_cold": "rq2-chrome-guest-",
    }[config]
    return Path(tempfile.mkdtemp(prefix=prefix))


def launch_chrome(
    config: str,
    *,
    chrome_bin: str | None = None,
    user_data_dir: Path | None = None,
    profile_directory: str | None = None,
    port: int | None = None,
    log_dir: Path | None = None,
) -> ChromeHandle:
    if config not in CHROME_CONFIGS:
        raise ValueError(f"Unknown Chrome config '{config}'. Use one of {CHROME_CONFIGS}")

    chrome_bin = chrome_bin or default_chrome_bin()
    incognito = config == "chrome_incognito"
    guest = config == "chrome_cold"
    # Incognito and guest get a throwaway dir. Full uses the research profile.
    # A dedicated user-data-dir is still required so this instance does not
    # collide with a daily Chrome already running.
    resolved_dir = Path(user_data_dir) if user_data_dir else resolve_user_data_dir(config)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    if config == "chrome_full" and profile_is_locked(resolved_dir):
        if profile_has_live_process(resolved_dir):
            raise SystemExit(
                f"Chrome profile is locked (another instance is using {resolved_dir}).\n"
                "Close that Chrome, or copy the profile with:\n"
                f"  {sys.executable} {Path(__file__).name} --prepare-full-profile"
            )
        clear_stale_profile_lock(resolved_dir)

    if config == "chrome_full" and not any(resolved_dir.iterdir()):
        raise SystemExit(
            f"Full-profile directory is empty: {resolved_dir}\n"
            "Prepare an aged copy first:\n"
            f"  {sys.executable} {Path(__file__).name} --prepare-full-profile\n"
            "Or point RQ2_CHROME_FULL_PROFILE at an existing user-data-dir."
        )

    profile_dir = None
    if config == "chrome_full":
        profile_dir = profile_directory or DEFAULT_PROFILE_DIRECTORY

    port = port or find_free_port()
    cdp_url = f"http://{DEFAULT_HOST}:{port}"
    cmd = build_chrome_args(
        chrome_bin=chrome_bin,
        port=port,
        user_data_dir=resolved_dir,
        incognito=incognito,
        guest=guest,
        profile_directory=profile_dir,
    )

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")

    log_path = None
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"chrome_{config}_{port}.log")
        log_file = open(log_path, "ab")
        stdout = log_file
        stderr = log_file

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    handle = ChromeHandle(
        config=config,
        pid=proc.pid,
        cdp_url=cdp_url,
        port=port,
        user_data_dir=str(resolved_dir),
        profile_directory=profile_dir,
        incognito=incognito,
        guest=guest,
        chrome_bin=chrome_bin,
        launched_utc=datetime.now(timezone.utc).isoformat(),
        log_path=log_path,
    )
    try:
        wait_for_cdp(cdp_url)
    except Exception:
        stop_chrome(handle)
        raise
    print(
        f"[chrome] {config} pid={handle.pid} cdp={handle.cdp_url} "
        f"user-data-dir={handle.user_data_dir}"
        + (f" profile={handle.profile_directory}" if handle.profile_directory else ""),
        flush=True,
    )
    return handle


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _signal_chrome(handle: ChromeHandle, sig: int) -> None:
    try:
        os.killpg(handle.pid, sig)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        os.kill(handle.pid, sig)
    except ProcessLookupError:
        pass


def stop_chrome(handle: ChromeHandle, timeout: float = 8.0) -> None:
    if handle.pid <= 0:
        return
    _signal_chrome(handle, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(handle.pid) and cdp_ready(handle.cdp_url, timeout=0.2) is None:
            return
        time.sleep(0.2)
    _signal_chrome(handle, signal.SIGKILL)


def prepare_full_profile(
    source: Path | None = None,
    dest: Path | None = None,
) -> Path:
    source = Path(source or DEFAULT_SOURCE_PROFILE).expanduser().resolve()
    dest = Path(dest or DEFAULT_FULL_PROFILE).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source Chrome profile not found: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    excludes = [
        "SingletonLock",
        "SingletonSocket",
        "SingletonCookie",
        "GPUCache",
        "GrShaderCache",
        "ShaderCache",
        "GraphiteDawnCache",
        "Crash Reports",
        "BrowserMetrics",
        "DeferredBrowserMetrics",
        "Safe Browsing",
        "component_crx_cache",
        "optimization_guide_model_store",
        "WasmTtsEngine",
    ]
    print(f"[chrome] Copying aged profile\n  from: {source}\n  to:   {dest}", flush=True)
    rsync = shutil.which("rsync")
    if rsync:
        cmd = [rsync, "-a", "--delete"]
        for name in excludes:
            cmd.extend(["--exclude", name])
        cmd.extend([str(source) + "/", str(dest) + "/"])
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"rsync failed with code {completed.returncode}")
    else:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            source,
            dest,
            ignore=shutil.ignore_patterns(*excludes),
            dirs_exist_ok=False,
        )
    print("[chrome] Full profile copy ready.", flush=True)
    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=CHROME_CONFIGS,
        help="Chrome configuration to launch",
    )
    parser.add_argument("--chrome-bin", default="", help="Path to google-chrome")
    parser.add_argument("--user-data-dir", default="", help="Override user-data-dir")
    parser.add_argument(
        "--profile-directory",
        default="",
        help="Chrome --profile-directory (full profile only, default Default)",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--state-file",
        default="",
        help="Write JSON handle (pid, cdp_url, ...) to this path",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Leave Chrome running; print CDP URL and exit",
    )
    parser.add_argument(
        "--prepare-full-profile",
        action="store_true",
        help="Copy ~/.config/google-chrome into profiles/chrome-full-research",
    )
    parser.add_argument("--source-profile", default="", help="Source user-data-dir to copy")
    parser.add_argument("--dest-profile", default="", help="Destination for the copy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_full_profile:
        prepare_full_profile(
            source=Path(args.source_profile) if args.source_profile else None,
            dest=Path(args.dest_profile) if args.dest_profile else None,
        )
        return 0

    if not args.config:
        raise SystemExit("Pass --config chrome_incognito|chrome_cold|chrome_full, or --prepare-full-profile")

    handle = launch_chrome(
        args.config,
        chrome_bin=args.chrome_bin or None,
        user_data_dir=Path(args.user_data_dir) if args.user_data_dir else None,
        profile_directory=args.profile_directory or None,
        port=args.port or None,
    )
    payload = handle.to_json()
    print(json.dumps(payload, indent=2), flush=True)
    if args.state_file:
        state_path = Path(args.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.keep_open:
        print(
            f"Chrome left running (pid={handle.pid}). Connect with "
            f"BROWSER_USE_CDP_URL={handle.cdp_url}",
            flush=True,
        )
        return 0

    try:
        print("Press Ctrl+C to stop Chrome.", flush=True)
        while True:
            time.sleep(1)
            try:
                os.kill(handle.pid, 0)
            except ProcessLookupError:
                print("Chrome exited.", flush=True)
                return 0
    except KeyboardInterrupt:
        print("\nStopping Chrome...", flush=True)
        stop_chrome(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
