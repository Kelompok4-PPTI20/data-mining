"""Run all four KDD phases in notebook order."""

import os


# The CLI pipeline must also run in headless/CI environments. Notebook kernels
# still render figures inline; this only supplies a non-GUI fallback before the
# phase modules import matplotlib.pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    """Execute the four phase modules as one end-to-end pipeline."""

    if __package__:
        from .phase1_preprocessing import run_phase1
        from .phase2_clustering import run_phase2
        from .phase3_association_rules import run_phase3
        from .phase4_anomaly_detection import run_phase4
    else:
        # Support direct execution with `python src/pipeline.py`.
        from phase1_preprocessing import run_phase1
        from phase2_clustering import run_phase2
        from phase3_association_rules import run_phase3
        from phase4_anomaly_detection import run_phase4

    run_phase1()
    run_phase2()
    run_phase3()
    run_phase4()


if __name__ == "__main__":
    main()
