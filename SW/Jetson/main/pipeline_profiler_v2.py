"""Observational V51 profiler. No robot/camera/model calls or file writes.

Control threads only timestamp and enqueue. A daemon formats/writes to the
original console stream (never the synchronous AUTO-X file tee). Diagnostic
output may be dropped under backpressure; summaries have a separate queue.
All *_child_s values are inclusive diagnostics, NOT additive wall-clock bins.
"""
from __future__ import annotations

import functools
import os
import queue
import sys
import threading
import time
from collections import Counter
from urllib.parse import quote


_active = None
_LEGACY_EVENTS = {
    "standby_command_start": "STANDBY_COMMAND_BEGIN",
    "standby_command_end": "STANDBY_COMMAND_END",
    "snapshot_request": "SNAPSHOT_REQUEST",
    "burst_start": "BURST_START",
    "burst_complete": "BURST_COMPLETE",
    "segmentation_complete": "SEGMENTATION_COMPLETE",
    "pose_complete": "POSE_COMPLETE",
    "inference_start": "INFERENCE_MODEL_START",
    "inference_complete": "PERCEPTION_RESULT_READY",
    "heatmap_cca_start": "STRUCTURE_CCA_START",
}
_PHASES = {
    "STANDBY_COMMAND_BEGIN": "standby",
    "STANDBY_ARRIVAL_VERIFIED": "rejudge_dispatch",
    "REJUDGE_WORKER_START": "rejudge_dispatch",
    "SNAPSHOT_REQUEST": "camera",
    "BURST_COMPLETE": "perception",
    "PERCEPTION_START": "perception",
    "PERCEPTION_COMPLETE": "judgment",
    "FINAL_BINDING_COMPLETE": "judgment_finalize",
    "JUDGMENT_WORKER_END": "next_action_scheduling",
}
_BINS = ("cleanup", "standby", "rejudge_dispatch", "camera", "perception",
         "judgment", "judgment_finalize", "next_action_scheduling")
_RETAINED_KINDS = ("planner", "io", "serial", "camera", "observation")


def _union(intervals, lo, hi):
    intervals = sorted((max(lo, a), min(hi, b)) for a, b in intervals
                       if b > lo and a < hi)
    total, end = 0.0, lo
    for a, b in intervals:
        if b > max(a, end):
            total += b - max(a, end)
        end = max(end, b)
    return total


class PipelineProfilerV2:
    def __init__(self, stream=None, clock=time.monotonic, capacity=8192,
                 start_writer=True, record_sink=None):
        self.clock = clock
        self.stream = stream if stream is not None else sys.stdout
        self.record_sink = record_sink
        self.session = time.monotonic_ns()
        self.lock = threading.RLock()
        self.local = threading.local()
        self.records = queue.Queue(maxsize=capacity)
        self.summaries = queue.SimpleQueue()
        self.stop = threading.Event()
        self.dropped = 0
        self.counter = 0
        self.feedback_counter = 0
        self.transition = None
        self.background_io_spans = {}  # open I/O only; survives transition rollover/cancel, removed at end
        self.action = "NONE"
        self.pending_first_serial = None
        self.observation = 0
        self.inference_serial = None
        self.counts = Counter()
        self.thread = None
        if start_writer:
            self.thread = threading.Thread(target=self._writer, name="pipeline-v2-output", daemon=True)
            self.thread.start()

    def _enqueue(self, record, summary=False):
        if self.record_sink is not None:
            self.record_sink(record)
        elif summary:
            self.summaries.put(record)
        else:
            try:
                self.records.put_nowait(record)
            except queue.Full:
                self.dropped += 1

    def _record(self, event, t, **meta):
        record = {"prefix": "PIPELINE-V2", "event": event, "session": self.session,
                  "transition": self.counter, "action": self.action, "t": t,
                  "observation": self.observation, "inference_serial": self.inference_serial,
                  "thread_role": getattr(self.local, "role", "background")}
        tr = self.transition
        if tr and tr["effect"] is not None:
            record["elapsed_from_effect_end"] = max(0.0, t - tr["effect"])
        record.update(meta)
        self._enqueue(record)

    def event(self, name, *, t=None, **meta):
        with self.lock:
            now = self.clock() if t is None else t
            tr = self.transition
            if name == "SNAPSHOT_REQUEST":
                self.observation += 1
                self.inference_serial = None
                self.counts = Counter({key: count for key, count in self.counts.items()
                                       if key[0] >= self.observation - 1})
            if tr is not None:
                tr["events"][name] = now
                if name == "GARMENT_ACTION_EFFECT_END":
                    # Multi-pass actions: the last manipulation end is authoritative.
                    tr["effect"] = now
                    tr["evidence"] = meta.get("evidence", "unknown")
                    tr["edges"] = [(now, "cleanup")]
                    # End-at-boundary observations still contribute to the count.
                    tr["spans"] = [item for item in tr["spans"] if item[1] >= now]
                elif tr["effect"] is not None and name in _PHASES:
                    tr["edges"].append((now, _PHASES[name]))
            self._record(name, now, **meta)

    def arm_action(self, action, epoch=None):
        with self.lock:
            self.pending_first_serial = (str(action), threading.get_ident())
            self._record("ACTION_SERIAL_TRACKING_ARMED", self.clock(), action=str(action), epoch=epoch)

    def disarm_action(self, worker):
        with self.lock:
            pending = self.pending_first_serial
            if pending is not None and pending[1] == threading.get_ident():
                self._record("ACTION_SERIAL_TRACKING_CANCELLED", self.clock(),
                             attempted_action=pending[0], worker=worker,
                             reason="worker_ended_without_motion_write")
                self.pending_first_serial = None

    def serial_sent(self, arm, cmd, t=None):
        if cmd.get("T") == 105:
            with self.lock:
                now = self.clock() if t is None else t
                self.feedback_counter += 1
                sent = getattr(self.local, "feedback_sent", None)
                if sent is None:
                    sent = self.local.feedback_sent = {}
                sent[arm] = (now, self.feedback_counter)
                self._record("SERIAL_FEEDBACK_REQUEST_SENT", now, arm=arm,
                             request_id=self.feedback_counter, evidence="serial_write_return")
            return
        # Torque/feedback/config are not motion/gripper commands.
        if not (cmd.get("T") == 106 or all(k in cmd for k in ("x", "y", "z"))):
            return
        with self.lock:
            if self.pending_first_serial is None:
                return
            now = self.clock() if t is None else t
            next_action = self.pending_first_serial[0]
            self._record("NEXT_ACTION_FIRST_SERIAL_COMMAND", now, action=next_action,
                         arm=arm, command_type=cmd.get("T"), evidence="serial_write_return")
            if self.transition is not None:
                tr = self.transition
                frozen = dict(tr, events=dict(tr["events"]), edges=list(tr["edges"]),
                              spans=list(tr["spans"]), span_stages=set(tr["span_stages"]))
                frozen.pop("open_spans", None)
                open_spans = dict(tr["open_spans"])
                open_spans.update(self.background_io_spans)  # deduplicate spans that started in this transition
                for token in open_spans.values():
                    start, stage, kind, _, _, _, meta, thread_id = token
                    frozen["spans"].append((start, now, kind, stage, thread_id, meta.get("thread_role")))
                job = {"prefix": "PIPELINE-V2-SUMMARY-JOB", "transition_data": frozen,
                       "now": now, "next_action": next_action, "dropped": self.dropped}
                # Aggregation/sorting happens on the output daemon, after the write.
                if self.record_sink is not None:  # Deterministic, no-I/O self-tests only.
                    self._enqueue(self._summary(now, next_action, frozen), summary=True)
                else:
                    self._enqueue(job, summary=True)
            self.counter += 1
            self.action = next_action
            self.pending_first_serial = None
            self.transition = {"effect": None, "evidence": "unknown", "events": {},
                               "edges": [], "spans": [], "open_spans": {}, "span_stages": set(), "action": next_action,
                               "id": self.counter}

    def begin(self, stage, kind="diagnostic", **meta):
        with self.lock:
            now = self.clock()
            token = (now, stage, kind, self.transition, self.observation,
                     self.inference_serial, dict(meta, thread_role=getattr(self.local, "role", "background")),
                     threading.get_ident())
            if self.transition is not None and kind in _RETAINED_KINDS:
                self.transition["open_spans"][id(token)] = token
            if kind == "io" and token[6]["thread_role"] == "background":
                self.background_io_spans[id(token)] = token
            if kind in ("planner", "model", "target", "debug_save"):
                self.event(stage + "_START" if kind != "debug_save" else "DEBUG_SAVE_BEGIN", t=now, **meta)
            return token

    def end(self, token, status="ok"):
        if token is None:
            return
        with self.lock:
            start, stage, kind, tr, obs, serial, meta, thread_id = token
            now = self.clock()
            if tr is not None:
                tr["open_spans"].pop(id(token), None)
            retained_tr = tr
            if kind == "io" and meta.get("thread_role") == "background":
                self.background_io_spans.pop(id(token), None)
                retained_tr = self.transition  # aggregation follows actual overlap; diagnostics keep the original tr below
            if retained_tr is not None:
                if kind in ("observation", "io"):
                    retained_tr["span_stages"].add((kind, stage))  # Preserve zero-valued summary fields after pruning.
                if kind in _RETAINED_KINDS and (retained_tr["effect"] is None or now >= retained_tr["effect"]):
                    retained_tr["spans"].append((start, now, kind, stage, thread_id, meta.get("thread_role")))
            fields = dict(meta, stage=stage, elapsed_s=max(0.0, now - start),
                          start_t=start, status=status, scope="child", observation=obs,
                          inference_serial=serial, thread=thread_id)
            if tr is not None:
                fields.update(transition=tr["id"], action=tr["action"],
                              elapsed_from_effect_end=(max(0.0, now - tr["effect"])
                                                       if tr["effect"] is not None else None))
            if kind == "observation":
                key = (obs, stage, meta.get("function"))
                self.counts[key] += 1
                fields["count"] = self.counts[key]
                event_name = "OBSERVATION_WORK"
            elif kind == "io":
                event_name = "DEBUG_IO"
            elif kind == "serial":
                event_name = "SERIAL_FEEDBACK"
                sent = getattr(self.local, "feedback_sent", {}).get(meta.get("arm"))
                if sent is not None and sent[0] >= start:
                    fields.update(request_sent_t=sent[0], request_id=sent[1],
                                  response_after_write_s=max(0.0, now - sent[0]))
            elif kind == "model":
                event_name = stage + "_COMPLETE"
            elif kind in ("planner", "target"):
                event_name = stage + "_END"
            elif kind == "debug_save":
                event_name = "DEBUG_SAVE_END"
            else:
                event_name = stage
            if tr is not None and kind == "model":
                tr["events"][event_name] = now
            self._record(event_name, now, **fields)

    def cancel(self, reason):
        with self.lock:
            if self.transition is not None:
                self._record("TRANSITION_INCOMPLETE", self.clock(), reason=reason,
                             missing_next_first_serial=1)
            self.transition = None
            self.pending_first_serial = None

    def _summary(self, now, next_action, tr=None):
        tr = self.transition if tr is None else tr
        effect = tr["effect"]
        result = {"prefix": "PIPELINE-V2-SUMMARY", "session": self.session,
                  "transition": tr["id"], "prev_action": tr["action"],
                  "next_action": next_action, "t": now, "effect_evidence": tr["evidence"],
                  "dropped_diagnostic_records": self.dropped}
        events = tr["events"]
        result["missing_events"] = ",".join(name for name in (
            "GARMENT_ACTION_EFFECT_END", "RELEASE_COMPLETE", "VERTICAL_CLEAR_COMPLETE",
            "STANDBY_ARRIVAL_VERIFIED", "FINAL_ACTION_SELECTED", "FINAL_BINDING_COMPLETE",
            "JUDGMENT_WORKER_END", "PHYSICAL_APPROVAL_READY") if name not in events) or "NONE"
        def delta(a, b):
            start, end = events.get(a), events.get(b)
            return end - start if start is not None and end is not None and end >= start else None
        for key, a, b in (
            ("effect_end_to_release_s", "GARMENT_ACTION_EFFECT_END", "RELEASE_COMPLETE"),
            ("release_to_clear_s", "RELEASE_COMPLETE", "VERTICAL_CLEAR_COMPLETE"),
            ("clear_to_standby_verified_s", "VERTICAL_CLEAR_COMPLETE", "STANDBY_ARRIVAL_VERIFIED"),
            ("standby_to_snapshot_s", "STANDBY_ARRIVAL_VERIFIED", "SNAPSHOT_REQUEST"),
            ("stillness_s", "SNAPSHOT_REQUEST", "STILLNESS_PASS"),
            ("burst_s", "BURST_START", "BURST_COMPLETE"),
            ("perception_s", "PERCEPTION_START", "PERCEPTION_COMPLETE"),
            ("action_model_s", "ACTION_MODEL_START", "ACTION_MODEL_COMPLETE"),
            ("structure_cca_s", "STRUCTURE_CCA_START", "STRUCTURE_CCA_COMPLETE"),
        ):
            result[key] = delta(a, b)
        result["pending_to_first_serial_s"] = (
            now - events["FINAL_BINDING_COMPLETE"] if "FINAL_BINDING_COMPLETE" in events else None)
        result["total_effect_end_to_first_serial_s"] = None if effect is None else now - effect
        bins = dict.fromkeys(_BINS, 0.0 if effect is not None else None)
        if effect is not None:
            edges = sorted(tr["edges"] + [(now, None)], key=lambda item: item[0])
            for (a, phase), (b, _) in zip(edges, edges[1:]):
                bins[phase] += max(0.0, min(now, b) - max(effect, a))
        result.update({"wall_" + key + "_s": value for key, value in bins.items()})
        result["wall_partition_total_s"] = None if effect is None else sum(bins.values())
        lo = effect if effect is not None else now
        spans = tr["spans"]
        for key, kinds in (("planner_total_s", {"planner"}), ("debug_io_s", {"io"}),
                           ("serial_feedback_child_s", {"serial"}),
                           ("camera_read_wait_child_s", {"camera"})):
            result[key] = _union([(a, b) for a, b, kind, _, _, role in spans
                                  if kind in kinds and (key != "debug_io_s" or role != "background")], lo, now)
        result["debug_io_all_threads_child_s"] = _union(
            [(a, b) for a, b, kind, _, _, _ in spans if kind == "io"], lo, now)
        for stage in ({stage for kind, stage in tr["span_stages"] if kind == "observation"}
                      | {stage for _, _, kind, stage, _, _ in spans if kind == "observation"}):
            relevant = [(a, b) for a, b, kind, name, _, _ in spans
                        if kind == "observation" and name == stage and b >= lo and a <= now]
            result["obs_" + stage + "_count"] = len(relevant)
            result["obs_" + stage + "_child_s"] = _union(relevant, lo, now)
        for stage in ({stage for kind, stage in tr["span_stages"] if kind == "io"}
                      | {stage for _, _, kind, stage, _, _ in spans if kind == "io"}):
            relevant = [(a, b) for a, b, kind, name, _, role in spans
                        if kind == "io" and name == stage and role != "background" and b > lo and a < now]
            result["io_" + stage + "_count"] = len(relevant)
            result["io_" + stage + "_child_s"] = _union(relevant, lo, now)
        result["standby_command_child_s"] = delta("STANDBY_COMMAND_BEGIN", "STANDBY_COMMAND_END")
        result["standby_arrival_after_command_child_s"] = delta("STANDBY_COMMAND_END", "STANDBY_ARRIVAL_VERIFIED")
        result["detail_scope"] = "child_do_not_sum"
        return result

    @staticmethod
    def format_record(record):
        prefix = record["prefix"]
        def value(item):
            if item is None:
                return "NA"
            if isinstance(item, float):
                return f"{item:.6f}"
            return quote(str(item), safe="_-.,:/")
        return "[" + prefix + "] " + " ".join(
            key + "=" + value(item) for key, item in record.items() if key != "prefix") + "\n"

    def _writer(self):
        while not self.stop.is_set() or not self.records.empty() or not self.summaries.empty():
            try:
                try:
                    record = self.summaries.get_nowait()
                except queue.Empty:
                    record = self.records.get(timeout=0.05)
                if record["prefix"] == "PIPELINE-V2-SUMMARY-JOB":
                    job = record
                    record = self._summary(job["now"], job["next_action"], job["transition_data"])
                    record["dropped_diagnostic_records"] = job["dropped"]
                # An existing print can be between its text and newline while its
                # AUTO-X tee flushes. Keep our prefix on its own parseable line.
                self.stream.write("\n" + self.format_record(record))
            except queue.Empty:
                pass
            except Exception:
                # Console failures never propagate into robot control.
                pass

    def close(self):
        self.stop.set()  # No join/flush/blocking I/O on a control thread.


def configure(stream=None):
    global _active
    try:
        if _active is not None:
            _active.close()
        # Main can create multiple V51 sessions; unwrap ALL older AUTO-X tees.
        # Otherwise profiler output would itself generate DEBUG_IO indefinitely.
        seen = set()
        while stream is not None and type(stream).__name__ == "_AutoXTeeStream" and id(stream) not in seen:
            seen.add(id(stream))
            stream = stream.original_stream
        _active = None if os.environ.get("PIPELINE_V2", "1") == "0" else PipelineProfilerV2(stream)
        thread_role("camera")
    except Exception:
        _active = None


def _safe(method, *args, **kwargs):
    try:
        profiler = _active
        if profiler is not None:
            return getattr(profiler, method)(*args, **kwargs)
    except Exception:
        pass
    return None


def event(name, **meta):
    _safe("event", name, **meta)


def legacy_mark(name, **meta):
    if name in _LEGACY_EVENTS:
        event(_LEGACY_EVENTS[name], **meta)


def arm_action(action, epoch=None):
    _safe("arm_action", action, epoch)


def serial_sent(arm, cmd, **meta):
    _safe("serial_sent", arm, cmd, **meta)


def observation_serial(serial):
    try:
        if _active is not None:
            with _active.lock:
                _active.inference_serial = serial
    except Exception:
        pass


class span:
    def __init__(self, stage, kind="diagnostic", **meta):
        self.stage, self.kind, self.meta = stage, kind, meta
        self.profiler, self.token = None, None
        self.status = "ok"

    def __enter__(self):
        try:
            self.profiler = _active
            if self.profiler is not None:
                self.token = self.profiler.begin(self.stage, self.kind, **self.meta)
        except Exception:
            pass
        return self

    def __exit__(self, typ, value, traceback):
        try:
            if self.profiler is not None:
                self.profiler.end(self.token, "error" if typ is not None else self.status)
        except Exception:
            pass
        return False


def timed(stage, kind="diagnostic", arm_attr=None, asset_arg=None):
    def decorate(fn):
        @functools.wraps(fn)
        def measured(*args, **kwargs):
            if _active is None:
                return fn(*args, **kwargs)
            meta = {"function": fn.__name__}
            if arm_attr:
                meta["arm"] = getattr(args[0], arm_attr, "UNKNOWN")
            if asset_arg is not None and len(args) > asset_arg:
                meta["asset"] = os.path.basename(str(args[asset_arg]))
            if kind == "observation" and args:
                meta["input_id"] = id(args[0])
            with span(stage, kind, **meta) as measurement:
                result = fn(*args, **kwargs)
                if kind == "serial" and result is None:
                    measurement.status = "no_response"
                return result
        return measured
    return decorate


def observe_call(stage, fn, *args, **kwargs):
    if _active is None:
        return fn(*args, **kwargs)
    with span(stage, "observation"):
        return fn(*args, **kwargs)


def planner_call(stage, fn, *args, **kwargs):
    if _active is None:
        return fn(*args, **kwargs)
    with span(stage, "planner"):
        return fn(*args, **kwargs)


def io_call(stage, fn, *args, **kwargs):
    if _active is None:
        return fn(*args, **kwargs)
    with span(stage, "io") as measurement:
        result = fn(*args, **kwargs)
        if stage == "debug_io_read_barrier":
            try:
                # Writer failures are cumulative evidence; the original flush result is unchanged.
                failures = getattr(getattr(fn, "__self__", None), "failures", 0)
                measurement.status = "timeout" if result is False else "write_failure" if failures else "ok"
            except Exception:
                pass  # classification must never affect control or the barrier result
        return result


def note_release(label):
    event("RELEASE_COMPLETE", label=label, evidence="command_and_existing_wait", verified=0)


def note_clear(label):
    if label in ("VLA-MICRO-V14-BOTH-ARMS-SAFE-CLEAR", "VLA-MICRO-V14-ERROR-SEQUENTIAL-SAFE-CLEAR"):
        event("VERTICAL_CLEAR_COMPLETE", label=label, evidence="feedback", verified=1)
    else:
        event("VERTICAL_CLEAR_DECLARED", label=label, evidence="existing_wait_no_feedback", verified=0)


def motion_group_end(label, ok):
    if ok and label in ("COARSE-PARALLEL-LIFT", "COARSE-STRETCH-LIFT", "COARSE-STRETCH-PLACE",
                         "POSE-ROT-PLACE-HOLD", "A133-DUAL-SWEEP", "A93-NOACTION-SOFT-SWEEP"):
        event("GARMENT_ACTION_EFFECT_END", label=label, evidence="existing_fixed_wait", verified=0)


def clear_arrival_result(fn):
    @functools.wraps(fn)
    def measured(*args, **kwargs):
        if _active is None:
            return fn(*args, **kwargs)
        result = fn(*args, **kwargs)
        try:
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                diag = result[1]
                # A disabled/no-send barrier returns success without physical proof.
                if diag.get("result") in ("arrived", "safe_clear_fallback"):
                    event("VERTICAL_CLEAR_COMPLETE", label=fn.__name__, evidence="feedback",
                          verified=1, result=diag.get("result"))
        except Exception:
            pass
        return result
    return measured


def close():
    global _active
    if _active is not None:
        _active.close()
        _active = None


def cancel(reason):
    _safe("cancel", reason)


def thread_role(role):
    try:
        if _active is not None:
            _active.local.role = role
    except Exception:
        pass


def worker_start(name, action=None, epoch=None):
    judge = str(name).startswith("x-auto-judge")
    thread_role("judgment" if judge else "action")
    if judge:
        event("REJUDGE_WORKER_START", worker=name, epoch=epoch)
    elif name == "auto-x-pending-execute":
        event("NEXT_ACTION_WORKER_START", action=action, epoch=epoch)


def worker_end(name, epoch=None):
    if str(name).startswith("x-auto-judge"):
        event("JUDGMENT_WORKER_END", worker=name, epoch=epoch)
    elif name == "auto-x-pending-execute":
        _safe("disarm_action", name)


def standby_verified(label, physically_verified):
    event("STANDBY_ARRIVAL_VERIFIED" if physically_verified else "STANDBY_ARRIVAL_DECLARED",
          label=label, evidence="feedback" if physically_verified else "barrier_disabled_or_dry_run",
          verified=int(physically_verified))


def targeted_stage_end(stage, index, total):
    if stage == "sweep":
        event("GARMENT_ACTION_EFFECT_END", pass_index=index, pass_total=total,
              evidence="existing_fixed_wait", verified=0)


def standby_sample(stable, required, label):
    if stable >= required:
        standby_verified(label, True)


def pose_sample(stable, required, label):
    if stable >= required and label == "PLACE-HOLD":
        event("GARMENT_ACTION_EFFECT_END", label=label, evidence="feedback", verified=1)


def scheduled_result(fn):
    @functools.wraps(fn)
    def measured(*args, **kwargs):
        if _active is None:
            return fn(*args, **kwargs)
        result = fn(*args, **kwargs)
        if result:
            event("REJUDGE_SCHEDULED", origin="TARGET_INVALIDATED")
        return result
    return measured
