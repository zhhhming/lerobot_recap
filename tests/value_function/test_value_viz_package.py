import os
import subprocess
import sys
import zipfile


def test_wheel_contains_and_serves_value_viz_assets(tmp_path):
    repo = __import__("pathlib").Path(__file__).parents[2]
    wheel_dir = tmp_path / "wheel"
    target = tmp_path / "installed"
    pip_env = os.environ.copy()
    pip_env["PIP_REQUIRE_VIRTUALENV"] = "0"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=repo,
        env=pip_env,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("lerobot-*.whl"))
    expected = {
        "lerobot/scripts/value_viz/index.html",
        "lerobot/scripts/value_viz/app.js",
        "lerobot/scripts/value_viz/style.css",
    }
    with zipfile.ZipFile(wheel) as archive:
        assert expected <= set(archive.namelist())

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        cwd=tmp_path,
        env=pip_env,
        check=True,
        capture_output=True,
        text=True,
    )
    script = """
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import lerobot.scripts.lerobot_value_viz as viz

assert Path(viz.__file__).resolve().is_relative_to(Path(__import__('os').environ['EXPECTED_ROOT']))
server = ThreadingHTTPServer(('127.0.0.1', 0), viz.Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
opener = build_opener(ProxyHandler({}))
try:
    base = f'http://127.0.0.1:{server.server_port}'
    requests = [
        ('/', b'Value Curves'),
        ('/app.js', b'const state'),
        ('/style.css', b':root'),
    ]
    for path, marker in requests:
        with opener.open(base + path, timeout=10) as response:
            assert response.status == 200
            assert marker in response.read()
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    env["EXPECTED_ROOT"] = str(target)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
