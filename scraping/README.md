# Scraping Tools

This folder contains standalone data-ingestion scripts used to refresh local
metadata caches for the moodboard analysis pipeline.

## Tools

- `refresh_database.py`: unified CLI for refreshing local databases in
  `database/`.
- `scrape_color_names.py`: refreshes `database/color_names.json` from Name That
  Color metadata, with attribution kept in the generated JSON. Pantone names
  are intentionally not redistributed.
- `aesthetic_sources.py`: contains the metadata-only Aesthetics Wiki and CARI
  scrapers used by `/api/aesthetics` and `refresh_database.py`.
