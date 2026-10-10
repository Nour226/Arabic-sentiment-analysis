"""Simple Prometheus-compatible metric exporter for monitoring samples."""

from __future__ import annotations

from collections import defaultdict


class PrometheusMetricsExporter:
    """Generate a small exporter payload without requiring the full Prometheus client."""

    def __init__(self, namespace: str = "arabic_sentiment") -> None:
        self.namespace = namespace
        self.http_request_ms: dict[str, float] = {}
        self.drift_scores: dict[str, float] = {}
        self.counters: dict[str, float] = defaultdict(float)

    def record_http_request(self, route: str, duration_ms: float) -> None:
        self.http_request_ms[route] = float(duration_ms)

    def record_drift(self, metric_name: str, value: float) -> None:
        self.drift_scores[metric_name] = float(value)

    def increment_counter(self, name: str, value: float = 1.0) -> None:
        self.counters[name] += float(value)

    def render(self) -> str:
        lines: list[str] = []
        if self.http_request_ms:
            lines.append(f"# HELP {self.namespace}_http_request_ms HTTP latency in milliseconds.")
            lines.append(f"# TYPE {self.namespace}_http_request_ms gauge")
            for route, latency in sorted(self.http_request_ms.items()):
                lines.append(f'{self.namespace}_http_request_ms{{route="{route}"}} {latency}')

        if self.drift_scores:
            lines.append(f"# HELP {self.namespace}_drift_score Drift score for monitored signal.")
            lines.append(f"# TYPE {self.namespace}_drift_score gauge")
            for metric, value in sorted(self.drift_scores.items()):
                lines.append(f'{self.namespace}_drift_score{{metric="{metric}"}} {value}')

        if self.counters:
            lines.append(f"# HELP {self.namespace}_events Total cumulative events.")
            lines.append(f"# TYPE {self.namespace}_events counter")
            for name, value in sorted(self.counters.items()):
                lines.append(f'{self.namespace}_events{{name="{name}"}} {value}')

        return "\n".join(lines) + ("\n" if lines else "")

    def __str__(self) -> str:
        return self.render()
