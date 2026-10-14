#!/usr/bin/env python3
"""End-to-end test of stopping a job that is generating, on a real sd-server.

It needs a real model: any model the server can load, on any backend. The jobs are small (64x64) and the
ones that get stopped never run to the end, so how long the test takes does not depend on --steps. The
standard library is enough.

  python3 test/server_cancel_e2e.py --server build/bin/sd-server --model /path/to/model.safetensors

Extra options for the server (a backend, --diffusion-fa, ...) go after --extra, one each, for example
--extra=--diffusion-fa. To use a server that is already running: --url http://127.0.0.1:1234 (add
--api-key if it asks for one); the check of the server log is then skipped. Exit code 0 = all checks passed.

The first generation can take minutes when the weights are read from a slow disk, so the test starts with
a warm-up job (--warmup-timeout) and only then looks at how a stop behaves.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

failures = 0


def check(condition, message):
    global failures
    if condition:
        print("ok:   " + message, flush=True)
    else:
        print("FAIL: " + message, flush=True)
        failures += 1
    return condition


class Abort(Exception):
    """The test cannot go on: what comes next depends on what just failed."""


class Client:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def call(self, method, path, body=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = raw.decode(errors="replace")
        return status, parsed, time.time() - started

    def submit(self, steps, seed, size):
        status, job, _ = self.call("POST", "/sdcpp/v1/img_gen", {
            "prompt": "a test",
            "width": size,
            "height": size,
            "seed": seed,
            "sample_params": {"sample_steps": steps},
        })
        if status != 202:
            raise Abort("the server did not accept a job: %s %s" % (status, job))
        return job["id"]

    def job(self, job_id):
        return self.call("GET", "/sdcpp/v1/jobs/" + job_id)[1]

    def wait_for(self, job_id, wanted, timeout):
        started = time.time()
        while time.time() - started < timeout:
            job = self.job(job_id)
            if job["status"] in wanted:
                return job
            if job["status"] in ("completed", "failed", "cancelled"):
                raise Abort("job %s ended as %s, expected %s: %s" % (job_id, job["status"], wanted, job.get("error")))
            time.sleep(0.1)
        raise Abort("job %s did not become %s within %d s" % (job_id, wanted, timeout))

    def health(self):
        return self.call("GET", "/sdcpp/v1/health")[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(args, log):
    port = args.port or free_port()
    command = [args.server, "-m", args.model, "--listen-ip", "127.0.0.1", "--listen-port", str(port)]
    if args.api_key:
        command += ["--api-key", args.api_key]
    command += args.extra
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    return process, "http://127.0.0.1:%d" % port


def wait_ready(client, process, timeout):
    started = time.time()
    while time.time() - started < timeout:
        if process is not None and process.poll() is not None:
            raise Abort("the server stopped at start-up (exit code %s)" % process.returncode)
        try:
            status, health, _ = client.call("GET", "/sdcpp/v1/health", timeout=5)
            if status == 200 and health.get("status") in ("ok", "degraded"):
                return health
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise Abort("the server was not ready after %d s" % timeout)


def png_looks_valid(image_b64):
    return isinstance(image_b64, str) and image_b64.startswith("iVBORw0KGgo")


def run_scenarios(client, args, mark_log):
    size, steps = args.size, args.steps

    status, capabilities, _ = client.call("GET", "/sdcpp/v1/capabilities")
    features = capabilities.get("features", {}) if isinstance(capabilities, dict) else {}
    check(status == 200 and features.get("cancel_generating") is True, "capabilities report cancel_generating")

    print("warming up (the first generation may read the weights from disk)...", flush=True)
    started = time.time()
    warm = client.wait_for(client.submit(2, 1, size), ("completed",), args.warmup_timeout)
    check(warm["status"] == "completed", "warm-up job completed in %.1f s" % (time.time() - started))
    start = client.health()
    before = start["counters"]
    mark_log()

    # 1. A job that is generating is stopped.
    job_id = client.submit(steps, 2, size)
    client.wait_for(job_id, ("generating",), 60)
    time.sleep(1.5)
    status, body, seconds = client.call("POST", "/sdcpp/v1/jobs/%s/cancel" % job_id)
    if status == 200 and body["status"] == "completed":
        raise Abort("the job finished before it could be cancelled: raise --steps or --size")
    check(status in (200, 202), "cancelling a generating job answers 200 or 202 (got %s)" % status)
    if status == 202:
        check(body.get("cancel_requested") is True and body["status"] == "generating",
              "a 202 carries the job with cancel_requested")
        body = client.wait_for(job_id, ("cancelled",), 120)
    check(body["status"] == "cancelled" and body["error"]["code"] == "cancelled",
          "the job ends as cancelled with the code cancelled")
    print("      the request returned after %.1f s" % seconds, flush=True)
    check(body["started"] is not None and body["completed"] is not None, "it records when it started and ended")

    # 2. Asking again is harmless.
    status, again, _ = client.call("POST", "/sdcpp/v1/jobs/%s/cancel" % job_id)
    check(status == 200 and again["status"] == "cancelled", "cancelling a cancelled job answers 200 with the job")
    check(client.job(job_id)["status"] == "cancelled", "and polling it still shows it cancelled")

    # 3. The same context keeps working: a stop request does not leak into the next job.
    job_id = client.submit(3, 3, size)
    done = client.wait_for(job_id, ("completed",), args.warmup_timeout)
    images = done["result"]["images"] if done.get("result") else []
    check(len(images) == 1 and png_looks_valid(images[0]["b64_json"]), "the next job completes and returns an image")
    health = client.health()
    now = health["counters"]
    check(health["status"] == start["status"] and health["profile"] == start["profile"],
          "the server is as healthy as before, on the same profile")
    check(now["cancelled"] - before["cancelled"] == 1, "the health report counts one cancelled job")
    check(now["failed"] == before["failed"] and now["step_downs"] == before["step_downs"],
          "which is neither a failure nor a step down")
    check(health["last_error"] == start["last_error"], "and not an error")

    # 4. A job that is still queued, then the one that is generating, with ?wait=0.
    running = client.submit(steps, 4, size)
    queued = client.submit(steps, 5, size)
    client.wait_for(running, ("generating",), 60)
    status, body, _ = client.call("POST", "/sdcpp/v1/jobs/%s/cancel" % queued)
    check(status == 200 and body["status"] == "cancelled" and body["started"] is None,
          "a queued job is cancelled at once, without ever starting")
    time.sleep(1.0)
    status, body, seconds = client.call("POST", "/sdcpp/v1/jobs/%s/cancel?wait=0" % running)
    check(status in (200, 202) and seconds < 2.0, "?wait=0 answers at once")
    if status == 202:
        check(body.get("cancel_requested") is True, "and says the stop is pending")
        check(client.job(running)["status"] in ("generating", "cancelled"),
              "polling shows the job stopping or already stopped")
    final = client.wait_for(running, ("cancelled",), 120)
    check(final["status"] == "cancelled", "the job ends as cancelled")
    check(client.job(queued)["status"] == "cancelled", "the queued job stayed cancelled and never ran")

    # 5. A request that waits behind a job being cancelled still completes.
    long_job = client.submit(steps, 6, size)
    client.wait_for(long_job, ("generating",), 60)
    outcome = {}

    def synchronous_request():
        outcome["result"] = client.call("POST", "/sdapi/v1/txt2img", {
            "prompt": "a sync test", "width": size, "height": size, "steps": 2, "seed": 7,
        }, timeout=args.warmup_timeout)

    thread = threading.Thread(target=synchronous_request)
    thread.start()
    time.sleep(1.5)
    status, body, _ = client.call("POST", "/sdcpp/v1/jobs/%s/cancel" % long_job)
    if status == 202:
        body = client.wait_for(long_job, ("cancelled",), 120)
    check(body["status"] == "cancelled", "the long job is cancelled while a synchronous request waits for the model")
    thread.join(args.warmup_timeout)
    sync_status, sync_body, _ = outcome.get("result", (None, None, 0))
    check(sync_status == 200 and isinstance(sync_body, dict) and len(sync_body.get("images", [])) == 1,
          "and that request, which was not cancelled, completes normally")

    # 6. Jobs that are finished, unknown, or asked about wrongly.
    status, body, _ = client.call("POST", "/sdcpp/v1/jobs/%s/cancel" % job_id)
    check(status == 200 and body["status"] == "completed" and body["result"] is not None,
          "cancelling a job that already completed leaves it completed")
    status, body, _ = client.call("POST", "/sdcpp/v1/jobs/job_does_not_exist/cancel")
    check(status == 404, "an unknown job is 404")
    for wrong in ("abc", "-1", "1.5", ""):
        status, body, _ = client.call("POST", "/sdcpp/v1/jobs/%s/cancel?wait=%s" % (job_id, wrong))
        check(status == 400, "wait=%r is rejected with 400" % wrong)
    status, body, seconds = client.call("POST", "/sdcpp/v1/jobs/%s/cancel?wait=999" % job_id)
    check(status == 200 and seconds < 5, "a long wait is accepted and costs nothing on a finished job")

    # 7. At the end the server is as healthy as at the start.
    health = client.health()
    now = health["counters"]
    check(health["status"] == start["status"] and health["busy"] is False, "the server is idle and as healthy as before")
    check(now["cancelled"] - before["cancelled"] == 3 and now["failed"] == before["failed"],
          "three jobs were stopped while generating (the queued one never reached the model), none failed")
    check(now["step_downs"] == before["step_downs"], "no step down happened")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="build/bin/sd-server", help="path to sd-server")
    parser.add_argument("--model", help="model for the server to load")
    parser.add_argument("--url", help="use a server that is already running instead of starting one")
    parser.add_argument("--api-key", help="API key of the server")
    parser.add_argument("--port", type=int, default=0, help="port for the server this test starts (default: a free one)")
    parser.add_argument("--extra", action="append", default=[], help="an extra option for sd-server (repeatable)")
    parser.add_argument("--size", type=int, default=64, help="width and height of the images (default 64, the minimum)")
    parser.add_argument("--steps", type=int, default=500, help="steps of the jobs that get stopped (they never run to the end)")
    parser.add_argument("--start-timeout", type=int, default=300, help="seconds to wait for the server to load the model")
    parser.add_argument("--warmup-timeout", type=int, default=900, help="seconds for any job that must complete")
    args = parser.parse_args()
    if not args.url and not args.model:
        parser.error("give --model (and --server), or --url of a running server")

    process = None
    log = None
    log_path = None
    if args.url:
        base_url = args.url
    else:
        handle, log_path = tempfile.mkstemp(prefix="sd-server-cancel-", suffix=".log")
        os.close(handle)
        log = open(log_path, "wb")
        process, base_url = start_server(args, log)

    offset = {"value": 0}

    def mark_log():
        # Whatever the server wrote while it started and warmed up is not what this test is about.
        if log is not None:
            time.sleep(0.5)
            log.flush()
            offset["value"] = os.path.getsize(log_path)

    client = Client(base_url, args.api_key)
    try:
        wait_ready(client, process, args.start_timeout)
        run_scenarios(client, args, mark_log)
    except Abort as error:
        check(False, str(error))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        if log is not None:
            log.close()

    if log_path is not None:
        with open(log_path, "rb") as handle:
            handle.seek(offset["value"])
            text = handle.read().decode(errors="replace")
        errors = re.findall(r"\[E\][^\r\n]*", text)
        check(not errors, "stopping jobs writes no error line to the server log" + (": " + errors[0] if errors else ""))
        if failures:
            print("server log kept at " + log_path)
        else:
            os.remove(log_path)

    print("\n%s (%d failures)" % ("TEST FAILED" if failures else "ALL TESTS PASSED", failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
