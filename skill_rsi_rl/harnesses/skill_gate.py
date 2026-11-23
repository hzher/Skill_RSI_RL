"""Decision-time triggers for step-scoped Skill injection in the two-task pilot."""

from dataclasses import dataclass
from statistics import median
import math
import re


class SkillGate:
    def decide(self, query):
        raise NotImplementedError

    def submitted(self, instance_id, action):
        pass

    def observed(self, instance_id, observation):
        pass


class CohortGate(SkillGate):
    _ESTIMATE_TOOLS = {"estimate_survival_by_group", "predict_cohort_survival"}
    _GAP = re.compile(r"^Unobservable cohorts \(([1-9][0-9]*),", re.MULTILINE)

    def __init__(self):
        self.instance_id = None
        self.pending_tool = None
        self.unresolved = False

    def decide(self, query):
        if query.instance_id != self.instance_id:
            self.instance_id = query.instance_id
            self.pending_tool = None
            self.unresolved = False
        if (query.metadata or {}).get("step") != "submission_extraction":
            return False, "outside_submission"
        if self.unresolved:
            return True, "local_estimate_has_unobservable_groups"
        return False, "no_observed_target_coverage_gap"

    def submitted(self, instance_id, action):
        if instance_id == self.instance_id:
            tool = action.get("tool_call") if isinstance(action, dict) else None
            self.pending_tool = tool.get("tool") if isinstance(tool, dict) else None

    def observed(self, instance_id, observation):
        tool, self.pending_tool = self.pending_tool, None
        if instance_id != self.instance_id or tool not in self._ESTIMATE_TOOLS:
            return
        content = observation.content
        if re.search(r"^=== Per-Group Survival ===$", content, re.MULTILINE):
            self.unresolved = bool(self._GAP.search(content))


@dataclass(frozen=True)
class Peak:
    center: float
    width: float


@dataclass(frozen=True)
class Scan:
    scan_id: str
    scope: tuple[str, str, str, str]
    peaks: tuple[Peak, ...]


def parse_scan(query):
    fields = dict(re.findall(r"^\s*(scan_id|sensor_id|detector_version|integration_time_ms|Band):\s*([^\n]+)$",
                             query.prompt, re.MULTILINE))
    if (fields.keys() != {"scan_id", "sensor_id", "detector_version", "integration_time_ms", "Band"}
            or fields["scan_id"] != (query.metadata or {}).get("scan_id")):
        return None
    number = r"\d+(?:\.\d+)?"
    matches = re.findall(rf"^\s*- peak_id: \S+ \| freq: ({number}) MHz \| power: -?{number} dBm \| width: ({number}) MHz$",
                         query.prompt, re.MULTILINE)
    peaks = tuple(Peak(float(center), float(width)) for center, width in matches)
    if len(peaks) != query.prompt.count("- peak_id:") or (
            not peaks and "(no peaks detected)" not in query.prompt):
        return None
    scope = tuple(fields[key] for key in ("sensor_id", "detector_version", "integration_time_ms", "Band"))
    return Scan(fields["scan_id"], scope, peaks)


def nearby(peak, peers):
    # Match by center, while leaving wider shifts and ambiguous identities to the normal task.
    radius = max(0.5, min(1.5, peak.width * 0.20))
    return [peer for peer in peers if abs(peak.center - peer.center) <= radius]


def supporting_observations(peak, history):
    matches = [(scan.scan_id, nearby(peak, scan.peaks)) for scan in history]
    return [(scan_id, peers[0]) for scan_id, peers in matches if len(peers) == 1]


class SpectrumGate(SkillGate):
    def __init__(self):
        self.scans = {}
        self.current = None
        self.current_instance_id = None
        self.prior_report = ()
        self.report_scope = None
        self.checked_silent = set()

    def decide(self, query):
        scan = parse_scan(query)
        self.current = scan
        self.current_instance_id = query.instance_id
        if scan is None:
            return False, "unrecognized_current_scan"
        if scan.scan_id in self.scans:
            return False, "duplicate_scan"
        history = [past for past in self.scans.values() if scan.scope == past.scope]
        self.scans[scan.scan_id] = scan
        for peak in scan.peaks:
            widths = [peer.width for _, peer in supporting_observations(peak, history)]
            if len(widths) < 2 or len(nearby(peak, scan.peaks)) != 1:
                continue
            typical = median(widths)
            spread = median(abs(width - typical) for width in widths)
            if abs(peak.width - typical) > max(1.0, 0.25 * typical, 2 * spread):
                return True, "current_width_conflicts_with_comparable_scans"
        if scan.scope == self.report_scope:
            unchecked = set()
            for peak in self.prior_report:
                if nearby(peak, scan.peaks):
                    continue
                support = supporting_observations(peak, history)
                if len(support) == 1 and support[0] not in self.checked_silent:
                    unchecked.add(support[0])
            if unchecked:
                self.checked_silent.update(unchecked)
                return True, "newly_disputed_silent_candidate_has_one_supporting_scan"
        return False, "no_observed_interval_conflict"

    def submitted(self, instance_id, action):
        if self.current is None or instance_id != self.current_instance_id or not isinstance(action, dict):
            return
        peaks = []
        for row in action.get("transmitters", []):
            center, width = row["center_freq"], row["bandwidth"]
            if math.isfinite(center) and math.isfinite(width) and width > 0:
                peaks.append(Peak(center, width))
        self.prior_report = tuple(peaks)
        self.report_scope = self.current.scope


GATES = {"cohort_studies": CohortGate, "blind_spectrum_monitoring": SpectrumGate}
