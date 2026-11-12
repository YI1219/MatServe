"""Oracle feedback ordering, adapter publication state, arrivals and energy sampling."""
from __future__ import annotations
import copy
import csv
import hashlib
import math
import json
from pathlib import Path
import threading
import time


class OracleEpisode:
    def __init__(self, source_id, article, target, date):
        self.source_id, self.article, self.date = source_id, article, date
        self._target, self._responded = target, False

    def public(self):
        return dict(source_id=self.source_id, article=self.article, date=self.date)

    @property
    def gold(self):
        if not self._responded:
            raise RuntimeError("oracle title is sealed until generation completes")
        return self._target

    def record_prediction(self, text, token_ids):
        if self._responded:
            raise RuntimeError("episode already answered")
        self.prediction, self.generated_ids = text, list(token_ids)
        self._responded = True


class OnlineState:
    def __init__(self, total_queries):
        self.total_queries, self.served = total_queries, 0
        self.edits, self.phi = [], 0.
        self.committed_frontier, self.committed_edits, self.version = 0, 0, 0
        self.active, self.refresh_count = None, 0

    def append_feedback(self, source_index, pair, charge):
        if source_index != self.served or self.served >= self.total_queries:
            raise RuntimeError("feedback must follow actual generation in query order")
        if not math.isfinite(charge) or charge < 0:
            raise ValueError("invalid maintenance charge")
        if pair is not None:
            item = copy.deepcopy(pair)
            item["source_index"] = source_index
            self.edits.append(item)
        self.served += 1
        self.phi += charge

    def unflushed(self):
        return [e for e in self.edits if e["source_index"] >= self.committed_frontier]

    def launch(self, selected):
        if self.active is not None or self.served >= self.total_queries or not selected:
            raise RuntimeError("refresh unavailable, empty, or stream already ended")
        if len(set(selected)) != len(selected) or any(i<0 or i>=len(self.edits) for i in selected):
            raise ValueError("invalid selected log indices")
        self.refresh_count += 1
        self.active = dict(refresh_id=self.refresh_count, source_frontier=self.served,
            edit_frontier=len(self.edits), selected_source_ids=[self.edits[i]["source_id"] for i in selected],
            pairs=copy.deepcopy([self.edits[i] for i in selected]), consumed_phi=self.phi)
        self.phi = 0.  # Only pre-launch potential is consumed, never callback feedback.
        return copy.deepcopy(self.active)

    def publish(self, ticket):
        if self.active is None or ticket["refresh_id"] != self.active["refresh_id"]:
            raise RuntimeError("wrong or duplicate publication")
        self.committed_frontier = self.active["source_frontier"]
        self.committed_edits = self.active["edit_frontier"]
        self.version += 1
        self.active = None


class ArrivalDriver:
    def __init__(self, offsets, start, clock=time.perf_counter):
        if any(b<a for a,b in zip(offsets,offsets[1:])):
            raise ValueError("arrivals must be ordered")
        self.offsets, self.start, self.clock, self.next_index = list(offsets), start, clock, 0

    @property
    def done(self):
        return self.next_index == len(self.offsets)

    def drain(self, serve):
        while not self.done and self.clock() >= self.start+self.offsets[self.next_index]:
            index = self.next_index
            serve(index, self.start+self.offsets[index])
            self.next_index += 1

    def next_time(self):
        return None if self.done else self.start+self.offsets[self.next_index]


class EventJournal:
    """Exclusive append-only records; a later failure cannot erase prior events."""
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open("x"); self.count=0

    def append(self,record):
        self.handle.write(json.dumps(record,allow_nan=False,separators=(",",":"))+"\n")
        self.handle.flush(); self.count+=1

    def metadata(self):
        if not self.handle.closed: self.handle.flush()
        return dict(path=str(self.path.resolve()),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    records=self.count,format="append-only JSONL; flushed per completed event")

    def close(self):
        if not self.handle.closed: self.handle.close()


def integrate_samples(samples, start, stop):
    """Trapezoidal watts×monotonic-seconds, interpolated at window boundaries."""
    if len(samples)<2 or stop<start or samples[0][0]>start or samples[-1][0]<stop:
        return None
    energy=0.
    for (t0,p0),(t1,p1) in zip(samples,samples[1:]):
        lo,hi=max(start,t0),min(stop,t1)
        if hi<=lo or t1<=t0:
            continue
        w0=p0+(p1-p0)*(lo-t0)/(t1-t0)
        w1=p0+(p1-p0)*(hi-t0)/(t1-t0)
        energy += .5*(w0+w1)*(hi-lo)
    return energy


class SysfsEnergy:
    def __init__(self, sensor_path, csv_path, interval=.2):
        self.path, self.csv_path, self.interval = Path(sensor_path), Path(csv_path), interval
        self.samples, self.errors = [], []
        self.lock, self.stopping = threading.Lock(), threading.Event()
        self.result = None

    def sample(self):
        # Timestamp, device read, append and CSV write form one ordered operation.
        with self.lock:
            stamp=time.perf_counter()
            try:
                raw=int(self.path.read_text().strip())
                watts=raw/1_000_000.
                if not 0 < watts <= 1000:
                    raise ValueError("INA238 power1_input outside declared uW range")
                self.samples.append((stamp,watts)); self.writer.writerow([stamp,raw,watts,""])
                self.handle.flush()
            except Exception as exc:
                self.errors.append(repr(exc)); self.writer.writerow([stamp,"","",repr(exc)])
                self.handle.flush()
                self.stopping.set()

    def _run(self):
        while not self.stopping.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.csv_path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.csv_path.open("x",newline="")
        self.writer=csv.writer(self.handle); self.writer.writerow(["monotonic_s","power_uW","power_W","error"])
        self.sample()
        self.started=self.samples[0][0] if self.samples else time.perf_counter()
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()
        return self

    def energy_between(self,start,stop):
        if self.errors:
            return None
        self.sample()
        with self.lock:
            if self.errors: return None
            return integrate_samples(list(self.samples),start,stop)

    def __exit__(self,*_):
        self.stopping.set(); self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.errors.append("sampler thread failed to stop")
        if not self.errors:
            self.sample()
        self.handle.close()
        stop=self.samples[-1][0] if self.samples else time.perf_counter()
        self.result=dict(sensor_path=str(self.path),unit="uW",samples=len(self.samples),
            start_monotonic_s=self.started,stop_monotonic_s=stop,
            energy_j=None if self.errors else integrate_samples(self.samples,self.started,stop),
            errors=list(self.errors),csv_path=str(self.csv_path.resolve()),
            csv_sha256=hashlib.sha256(self.csv_path.read_bytes()).hexdigest(),
            boundary="complete replay; model-load/shared-calibration excluded")
