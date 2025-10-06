# DATA Directory Structure & Tracking Policy

This repository separates data assets by lifecycle stage to keep Git lean while preserving essential reference datasets.

## Folders

- raw/        : Original source files (small CSV/XLSX or authoritative shapefile set). Minimal/manual edits only.
- external/   : Third-party supplemental reference (codes, lookup tables). Treat as read-only snapshots.
- interim/    : In-progress transformation outputs (joins, normalization, temp merges). Auto-regenerated.
- processed/  : Clean, analysis-ready curated datasets for modeling/visualization.
- cache/      : Ephemeral geocoding caches, API responses, derived indices.
- temp/       : (Optional) One-off scratch exports not meant for versioning.
- exports/    : Final published deliverables (charts, packaged CSVs) – usually large, ignored.

## Git Tracking Rules

- Keep: raw/, external/ small stable files; canonical boundary shapefile folder (BND_ADM_DONG_PG) whitelisted.
- Ignore: interim/, processed/, cache/, temp/, exports/ (patterned in .gitignore).
- Shapefiles: All *.shp/.shx/.dbf/... ignored except explicit whitelist path(s).
- Large notebook artifacts (.png/.html) under notebooks/outputs are ignored; *.ipynb kept.

## Naming Conventions

```
<topic>_geocoded_<YYYYMMDD_HHMM>.csv       # time-stamped
<topic>_geocoded_enriched.csv              # canonical enriched version
<topic>_geocoded_debug.csv                 # optional debug diagnostics
<topic>_syn_filtered.csv                   # subset (신속통합기획)
```

## Reproducibility Notes

1. Any file under ignored folders must be regenerable via scripts (document command in README or script --help).
2. Geocoding cache JSON lives in cache/ and can be optionally archived if reproducibility across API changes is critical.
3. Do not manually edit processed/ outputs – adjust upstream script logic instead.

## Adding New Authoritative Data

1. Place original file in raw/ (or external/ if third-party reference).
2. Record source URL/date in a header comment at top of README or separate SOURCES.md.
3. Run transformation pipeline → outputs land in interim/ then curated copy optionally promoted to processed/.

## Whitelisting Additional Shapefiles

If a new boundary/reference geometry must be versioned:
- Create a dedicated folder under DATA/ (e.g., DATA/BND_<NAME>_PG)
- Add an exception line to .gitignore: `!DATA/BND_<NAME>_PG/**`
- Keep only essential files (.shp, .shx, .dbf, .prj, .cpg) and remove huge duplicates.

## Cleanup Guidance

- Periodically purge cache/, interim/, temp/ to avoid stale artifacts.
- Before committing, ensure no large binary slipped past ignore patterns.

## Future Enhancements

- Optional DVC or Git LFS integration for very large stable assets.
- Hash manifest for raw/ integrity verification.

