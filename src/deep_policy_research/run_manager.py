from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactVersion, RunManifest
from .spec import TaskSpec


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    version_prefix: str
    alias_filename: str
    versions_attr: str
    current_attr: str
    file_extension: str = ".json"


ARTIFACTS: dict[str, ArtifactDescriptor] = {
    "policy": ArtifactDescriptor("policy", "policy.json", "policy_versions", "current_policy_version"),
    "example_pool": ArtifactDescriptor(
        "example_pool",
        "example_pool.json",
        "example_pool_versions",
        "current_example_pool_version",
    ),
    "prompt": ArtifactDescriptor(
        "reader_prompts",
        "reader_prompts.json",
        "prompt_versions",
        "current_prompt_version",
    ),
    "metrics": ArtifactDescriptor("metrics", "metrics.json", "metrics_versions", "current_metrics_version"),
    "research": ArtifactDescriptor(
        "research",
        "research.json",
        "research_versions",
        "current_research_version",
    ),
    "prompt_optimization": ArtifactDescriptor(
        "prompt_optimization",
        "prompt_optimization.json",
        "prompt_optimization_versions",
        "current_prompt_optimization_version",
    ),
    "redteam": ArtifactDescriptor(
        "redteam",
        "redteam.json",
        "redteam_versions",
        "current_redteam_version",
    ),
    "sources": ArtifactDescriptor(
        "sources",
        "sources.jsonl",
        "sources_versions",
        "current_sources_version",
        file_extension=".jsonl",
    ),
    "chunks": ArtifactDescriptor(
        "chunks",
        "chunks.jsonl",
        "chunks_versions",
        "current_chunks_version",
        file_extension=".jsonl",
    ),
    "candidate_rules": ArtifactDescriptor(
        "candidate_rules",
        "candidate_rules.jsonl",
        "candidate_rules_versions",
        "current_candidate_rules_version",
        file_extension=".jsonl",
    ),
    "filtered_rules": ArtifactDescriptor(
        "filtered_rules",
        "filtered_rules.jsonl",
        "filtered_rules_versions",
        "current_filtered_rules_version",
        file_extension=".jsonl",
    ),
}


class RunManager:
    def __init__(self, root_dir: Path, manifest: RunManifest):
        self.root_dir = root_dir
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        output_dir: str | Path,
        spec: TaskSpec,
        run_config: dict[str, Any],
        *,
        current_step: str = "eval",
    ) -> "RunManager":
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        now = _utc_now()
        run_id = now.strftime("run_%Y%m%d_%H%M%S")
        root_dir = output_root / run_id
        suffix = 1
        while root_dir.exists():
            root_dir = output_root / f"{run_id}_{suffix:02d}"
            suffix += 1
        run_id = root_dir.name
        root_dir.mkdir(parents=False, exist_ok=False)
        (root_dir / "logs").mkdir(parents=False, exist_ok=False)

        spec_path = root_dir / "spec.json"
        spec_path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

        run_config_path = root_dir / "run_config.json"
        run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

        created_at = now.isoformat().replace("+00:00", "Z")
        manifest = RunManifest(
            run_id=run_id,
            task_id=spec.task_id,
            created_at=created_at,
            updated_at=created_at,
            status="running",
            current_round=0,
            current_iteration=0,
            current_step=current_step,
            spec_path=spec_path.name,
            run_config_path=run_config_path.name,
        )

        manager = cls(root_dir=root_dir, manifest=manifest)
        manager.save_manifest()
        return manager

    @classmethod
    def load(cls, run_dir: str | Path) -> "RunManager":
        root_dir = Path(run_dir).expanduser().resolve()
        manifest_path = root_dir / "run_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = RunManifest.from_dict(json.load(handle))
        return cls(root_dir=root_dir, manifest=manifest)

    @property
    def spec_path(self) -> Path:
        return self.root_dir / self.manifest.spec_path

    @property
    def run_config_path(self) -> Path:
        return self.root_dir / self.manifest.run_config_path

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / "run_manifest.json"

    def save_manifest(self) -> None:
        self.manifest.updated_at = _utc_now().isoformat().replace("+00:00", "Z")
        self.manifest_path.write_text(
            json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def update_status(
        self,
        status: str,
        *,
        current_step: str | None = None,
        current_round: int | None = None,
        current_iteration: int | None = None,
    ) -> None:
        self.manifest.status = status
        if current_step is not None:
            self.manifest.current_step = current_step
        if current_round is not None:
            self.manifest.current_round = current_round
        if current_iteration is not None:
            self.manifest.current_iteration = current_iteration
        self.save_manifest()

    def new_version(self, kind: str) -> str:
        descriptor = _artifact_descriptor(kind)
        seq = len(getattr(self.manifest, descriptor.versions_attr)) + 1
        return f"{self.manifest.run_id}__{descriptor.version_prefix}__{seq:03d}"

    def write_artifact(self, kind: str, artifact: Any) -> ArtifactVersion:
        return self.record_artifact(kind, artifact, accept=True)

    def record_artifact(self, kind: str, artifact: Any, *, accept: bool) -> ArtifactVersion:
        descriptor = _artifact_descriptor(kind)
        version = artifact.version
        filename = f"{descriptor.version_prefix}_{version}{descriptor.file_extension}"
        created_at = _utc_now().isoformat().replace("+00:00", "Z")

        self._write_json(self.root_dir / filename, artifact.to_dict())

        version_entry = ArtifactVersion(version=version, path=filename, created_at=created_at)
        versions = getattr(self.manifest, descriptor.versions_attr)
        versions.append(version_entry)
        if accept:
            self._write_json(self.root_dir / descriptor.alias_filename, artifact.to_dict())
            setattr(self.manifest, descriptor.current_attr, version)
        self.save_manifest()
        return version_entry

    def load_artifact(self, kind: str, cls_type: Any, version: str | None = None) -> Any:
        descriptor = _artifact_descriptor(kind)
        if version is None:
            version = getattr(self.manifest, descriptor.current_attr)
        if version is None:
            raise ValueError(f"no current {kind} artifact is available")

        versions = getattr(self.manifest, descriptor.versions_attr)
        for entry in versions:
            if entry.version == version:
                path = self.root_dir / entry.path
                with path.open("r", encoding="utf-8") as handle:
                    return cls_type.from_dict(json.load(handle))
        raise ValueError(f"unable to find {kind} artifact version {version!r}")

    def accept_artifact_version(self, kind: str, version: str) -> None:
        descriptor = _artifact_descriptor(kind)
        versions = getattr(self.manifest, descriptor.versions_attr)
        for entry in versions:
            if entry.version == version:
                source_path = self.root_dir / entry.path
                if descriptor.file_extension == ".json":
                    payload = json.loads(source_path.read_text(encoding="utf-8"))
                    self._write_json(self.root_dir / descriptor.alias_filename, payload)
                else:
                    (self.root_dir / descriptor.alias_filename).write_text(
                        source_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                setattr(self.manifest, descriptor.current_attr, version)
                self.save_manifest()
                return
        raise ValueError(f"unable to find {kind} artifact version {version!r}")

    def write_rows_artifact(self, kind: str, version: str, rows: list[Any]) -> ArtifactVersion:
        return self.record_rows_artifact(kind, version, rows, accept=True)

    def record_rows_artifact(
        self,
        kind: str,
        version: str,
        rows: list[Any],
        *,
        accept: bool,
    ) -> ArtifactVersion:
        descriptor = _artifact_descriptor(kind)
        if descriptor.file_extension != ".jsonl":
            raise ValueError(f"artifact kind {kind!r} is not configured as a JSONL artifact")

        filename = f"{descriptor.version_prefix}_{version}{descriptor.file_extension}"
        created_at = _utc_now().isoformat().replace("+00:00", "Z")
        self._write_jsonl(self.root_dir / filename, rows)

        version_entry = ArtifactVersion(version=version, path=filename, created_at=created_at)
        versions = getattr(self.manifest, descriptor.versions_attr)
        versions.append(version_entry)
        if accept:
            self._write_jsonl(self.root_dir / descriptor.alias_filename, rows)
            setattr(self.manifest, descriptor.current_attr, version)
        self.save_manifest()
        return version_entry

    def load_rows_artifact(self, kind: str, row_cls: Any, version: str | None = None) -> list[Any]:
        descriptor = _artifact_descriptor(kind)
        if descriptor.file_extension != ".jsonl":
            raise ValueError(f"artifact kind {kind!r} is not configured as a JSONL artifact")
        if version is None:
            version = getattr(self.manifest, descriptor.current_attr)
        if version is None:
            raise ValueError(f"no current {kind} artifact is available")

        versions = getattr(self.manifest, descriptor.versions_attr)
        for entry in versions:
            if entry.version == version:
                rows: list[Any] = []
                with (self.root_dir / entry.path).open("r", encoding="utf-8") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        rows.append(row_cls.from_dict(json.loads(stripped)))
                return rows
        raise ValueError(f"unable to find {kind} artifact version {version!r}")

    def describe(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.root_dir),
            "manifest": self.manifest.to_dict(),
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), sort_keys=True))
                handle.write("\n")


def _artifact_descriptor(kind: str) -> ArtifactDescriptor:
    try:
        return ARTIFACTS[kind]
    except KeyError as exc:
        supported = ", ".join(sorted(ARTIFACTS))
        raise ValueError(f"unsupported artifact kind {kind!r}; expected one of {supported}") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
