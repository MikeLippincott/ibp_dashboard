"""Copy the standalone static dashboard into a loadable index.html.

Run:
    python export_static.py

This copies static/index.html to the dashboard folder root so the page loads as the default document when hosted.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / 'static' / 'index.html'
DST = HERE / 'index.html'
ALT_DST = HERE / 'static_site_export.html'

def main():
    if not SRC.exists():
        print('static/index.html is missing')
        return 1
    shutil.copy(SRC, DST)
    shutil.copy(SRC, ALT_DST)
    print('Exported static site to', DST)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
