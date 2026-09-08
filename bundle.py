"""Build an offline source bundle."""
from pathlib import Path
def main():
    root=Path(__file__).parent; Path("kaggle_inference_bundle.py").write_text("\n\n".join(p.read_text() for p in sorted((root/"src").rglob("*.py"))))
if __name__=="__main__": main()
