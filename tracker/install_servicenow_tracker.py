#!/usr/bin/env python3
"""Create or update the global UI Script that loads the TrackUI tracker on a ServiceNow instance.

  python3 servicenow/install_tracker.py --check                     # credentials + instance reachable
  python3 servicenow/install_tracker.py --base https://HOST/js      # install / update
  python3 servicenow/install_tracker.py --remove                    # deactivate

Credentials come from SN_INSTANCE, SN_USER, SN_PASSWORD (never written to disk).

--base is the directory that serves the four TrackUI library files (124912bt, icekhqfb, gmjkx9hp,
deblr51g). They are served by the TraceHub tracker itself (hotel:~/tracker/public/js) at https://api.thetracehub.com/js.
"""

from __future__ import annotations

import argparse
import sys

from sn import SN

NAME = "tracehub_tracker"
POST_SERVER = "https://api.thetracehub.com"
DEFAULT_BASE = "https://api.thetracehub.com/js"  # served by ~/tracker/public/js on the hotel host

TEMPLATE = r"""
(function () {
  if (window.__tracehubLoaded) return;
  window.__tracehubLoaded = true;

  // Keep the agent=/task=/model=/hum= start-URL tags on every frame, so each event's url carries them.
  try {
    var KEYS = ['agent', 'task', 'model', 'hum'];
    var top_ = window.top;
    var q = new URLSearchParams(top_.location.search);
    var saved = JSON.parse(top_.sessionStorage.getItem('__trace_tags') || '{}');
    KEYS.forEach(function (k) { if (q.get(k)) saved[k] = q.get(k); });
    top_.sessionStorage.setItem('__trace_tags', JSON.stringify(saved));
    var here = new URLSearchParams(window.location.search), changed = false;
    KEYS.forEach(function (k) { if (saved[k] && !here.get(k)) { here.set(k, saved[k]); changed = true; } });
    if (changed && window.history && history.replaceState) {
      history.replaceState(history.state, '', window.location.pathname + '?' + here.toString() + window.location.hash);
    }
  } catch (e) { /* cross-origin or storage blocked: carry on without tags */ }

  var BASE = '%(base)s';
  var FILES = ['124912bt.min.js', 'icekhqfb.min.js', 'gmjkx9hp.min.js', 'deblr51g.min.js'];  // the loader's own list

  function start() {
    try {
      TrackUI.record({ postServer: '%(post)s', regularEvents: '*', pollingEvents: '*',
                       pollingMs: 100, debug: false, postInterval: 1 });
    } catch (e) { /* tracker failed to start */ }
  }

  // Not using 0laso3vm.min.js: it finds its siblings from the last <script>'s src, which is an inline
  // script inside ServiceNow, so it requests them from the instance root and 404s.
  function loadNext(i) {
    if (i >= FILES.length) return start();
    var s = document.createElement('script');
    s.type = 'text/javascript';
    s.src = BASE + '/' + FILES[i];
    s.onload = function () { loadNext(i + 1); };
    (document.head || document.documentElement).appendChild(s);
  }
  loadNext(0);
})();
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--base", default=DEFAULT_BASE, help="URL directory serving the TrackUI .js files")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--post-server", default=POST_SERVER)
    args = ap.parse_args()

    sn = SN.from_env()
    pw = sn.password
    quotes = ("'", '"')
    has_quote = any(q in pw for q in quotes)
    print("instance=%s user=%r password_length=%d leading_or_trailing_space=%s has_quote=%s"
          % (sn.instance, sn.user, len(pw), pw != pw.strip(), has_quote))
    try:
        existing = sn.query("sys_ui_script", f"name={NAME}", fields="sys_id,name,active,global")
    except RuntimeError as exc:
        print(f"\nREST call failed: {exc}\n")
        if "HTTP 401" in str(exc):
            print("401 = the instance rejected this user/password. Check, in this order:\n"
                  "  1. password_length above matches your password (a quote, space or newline changes it)\n"
                  "  2. the same user/password works at <instance>/login.do in a browser\n"
                  "  3. the user is Active, not Locked out, and 'Password needs reset' is off\n"
                  "  4. try a dedicated API user (RUNBOOK section 1)")
        elif "HTTP 403" in str(exc):
            print("403 = login worked but the user lacks a role (give it admin).")
        return 2
    if args.check:
        me = sn.query("sys_user", f"user_name={sn.user}", fields="user_name,roles")
        print(f"reachable: {sn.instance}; signed in as {sn.user} ({len(me)} user row)")
        print(f"UI script '{NAME}':", existing[0] if existing else "not installed")
        return 0
    if args.remove:
        for row in existing:
            sn._call("PATCH", f"table/sys_ui_script/{row['sys_id']}", body={"active": "false"})
        print(f"deactivated {len(existing)} script(s)")
        return 0

    body = {"name": NAME, "active": "true", "global": "true", "ui_type": "0",
            "description": "TrackUI tracker for the agent-objective study",
            "script": TEMPLATE % {"base": args.base.rstrip("/"), "post": args.post_server}}
    if existing:
        sn._call("PATCH", f"table/sys_ui_script/{existing[0]['sys_id']}", body=body)
        print("updated", existing[0]["sys_id"])
    else:
        print("created", sn.insert("sys_ui_script", body)["sys_id"])
    print("Now clear the cache (navigate to /cache.do), reload a list page and check the network tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
