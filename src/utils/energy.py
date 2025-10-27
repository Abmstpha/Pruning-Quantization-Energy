"""Energy tracking utilities using CodeCarbon."""

import os
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from codecarbon import EmissionsTracker


def count_csv_rows(csv_path: Path) -> int:
    """Count rows in CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Number of rows in CSV
    """
    if not csv_path.exists():
        return 0
    try:
        return len(pd.read_csv(csv_path))
    except Exception:
        return 0


def get_last_emission_data(csv_path: Path, start_index: int) -> Tuple[Optional[float], Optional[float], Dict]:
    """Get the last emission data from CSV.
    
    Args:
        csv_path: Path to emissions CSV
        start_index: Starting index to look from
        
    Returns:
        Tuple of (energy_kwh, emissions_kg, metadata_dict)
    """
    if not csv_path.exists():
        return None, None, {}
    
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None, {}
    
    # Get data from start_index onwards
    df_subset = df.iloc[start_index:] if len(df) > start_index else df
    if df_subset.empty:
        return None, None, {}
    
    row = df_subset.iloc[-1].to_dict()
    
    energy_kwh = float(row.get("energy_consumed")) if pd.notna(row.get("energy_consumed")) else None
    emissions_kg = float(row.get("emissions")) if pd.notna(row.get("emissions")) else None
    
    metadata = {
        "country_name": row.get("country_name"),
        "country_iso": row.get("country_iso"),
        "region": row.get("region"),
        "cloud": row.get("cloud_provider"),
        "cloud_region": row.get("cloud_region"),
    }
    
    return energy_kwh, emissions_kg, metadata


@contextmanager
def track_emissions(
    section_name: str, 
    output_dir: Path, 
    country_iso: str = None
):
    """Context manager for tracking emissions during code execution.
    
    Args:
        section_name: Name of the section being tracked
        output_dir: Directory to save emissions data
        country_iso: Country ISO code for carbon intensity
        
    Yields:
        None
        
    Example:
        with track_emissions("training", Path("./results")) as tracker:
            # Training code here
            pass
        # Access results via track_emissions.last_results
    """
    if country_iso is None:
        country_iso = os.getenv("CC_ISO", "FRA")
    
    # Prepare tracker arguments
    tracker_kwargs = {
        "project_name": f"pruning-quantization::{section_name}",
        "output_dir": str(output_dir),
        "save_to_file": True,
        "measure_power_secs": 1,
        "log_level": "warning",
    }
    
    # Handle different CodeCarbon versions
    sig = inspect.signature(EmissionsTracker.__init__)
    if "country_iso" in sig.parameters:
        tracker_kwargs["country_iso"] = country_iso
    elif "country_iso_code" in sig.parameters:
        tracker_kwargs["country_iso_code"] = country_iso
    
    emissions_csv = output_dir / "emissions.csv"
    before_count = count_csv_rows(emissions_csv)
    
    tracker = EmissionsTracker(**tracker_kwargs)
    tracker.start()
    
    try:
        yield
    finally:
        tracker.stop()
        
        # Extract results
        energy_kwh, emissions_kg, metadata = get_last_emission_data(emissions_csv, before_count)
        
        # Store results for access
        track_emissions.last_results = {
            "energy_kwh": energy_kwh,
            "emissions_kg": emissions_kg,
            **metadata
        }


# Initialize last_results attribute
track_emissions.last_results = {}
