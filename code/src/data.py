"""Session-loading helpers for Dynamic Routing analyses."""

from pathlib import Path

import pandas as pd
import pynwb

DEFAULT_DATA_ROOT = Path("/root/capsule/data")
DEFAULT_METADATA_PATH = DEFAULT_DATA_ROOT / "dynamic_routing_metadata.csv"
DEFAULT_CUBE_PATH = DEFAULT_DATA_ROOT / "dynamicrouting_datacube"
TASK_STIMULI = ("vis1", "vis2", "sound1", "sound2")
TASK_CONTEXTS = ("vis", "aud")


def resolve_session(session_ref, metadata_path=DEFAULT_METADATA_PATH):
    """Resolve a metadata row index, numeric session ID, or full session name."""
    metadata = pd.read_csv(metadata_path)
    value = str(session_ref)
    if value.isdigit() and int(value) < len(metadata):
        session_name = str(metadata.name.iloc[int(value)])
    else:
        matches = metadata.loc[
            metadata["name"].astype(str).str.contains(value, regex=False), "name"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one metadata match for {session_ref!r}; found {len(matches)}"
            )
        session_name = str(matches.iloc[0])
    session_id = "_".join(session_name.split("_")[1:3])
    return session_name, session_id


def load_session(
    session_ref,
    metadata_path=DEFAULT_METADATA_PATH,
    cube_path=DEFAULT_CUBE_PATH,
):
    """Load one NWB session and return it with its stable identifiers."""
    session_name, session_id = resolve_session(session_ref, metadata_path)
    path = Path(cube_path) / session_name / f"{session_id}.nwb.zarr"
    return pynwb.read_nwb(str(path)), session_name, session_id


def task_trials(session):
    """Return the recognized stimulus trials from the two rewarded contexts."""
    trials = session.trials.to_dataframe()
    mask = trials["stim_name"].isin(TASK_STIMULI) & trials[
        "rewarded_modality"
    ].isin(TASK_CONTEXTS)
    trials = trials.loc[mask].copy().reset_index(drop=True)
    trials["stimulus_context"] = (
        trials["stim_name"].astype(str)
        + " | "
        + trials["rewarded_modality"].astype(str)
    )
    return trials


def select_units(units, area):
    """Select default-QC single units assigned to one structure."""
    return units.loc[
        units["default_qc"].astype(bool)
        & units["decoder_label"].eq("sua")
        & units["structure"].eq(area)
    ].copy()
