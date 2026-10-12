from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

PROJECT_DIRS = {
    'base_dir': BASE_DIR,
    'data_dir': BASE_DIR / 'data',
    'raw_data_dir': BASE_DIR / 'data' / 'raw',
    'staging_data_dir': BASE_DIR / 'data' / 'staging',
    'mart_data_dir': BASE_DIR / 'data' / 'mart',
    'assets_dir': BASE_DIR / 'assets',
    'notebooks_dir': BASE_DIR / 'notebook',
}


def config_paths() -> dict[str, Path]:
    """Return validated project directories."""
    missing = [
        f"{name}: {path}"
        for name, path in PROJECT_DIRS.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            'Missing required project directories:\n- ' + '\n- '.join(missing)
        )

    return PROJECT_DIRS.copy()


if __name__ == '__main__':
    for key, path in config_paths().items():
        print(f'{key}: {path}')

