"""Display saved MOFI figures in notebooks."""
from contextlib import contextmanager
from pathlib import Path


def show_paper_panel(path):
    """Display a saved figure."""
    from IPython.display import Image, display
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Figure not found: {path}. Saved results can be viewed in the notebook.")
    display(Image(data=path.read_bytes(), format="png"))


def setup_notebook(*args, **kwargs):
    """Require the full implementation for model inference."""
    raise RuntimeError("The full MOFI implementation will be released after publication. The saved notebook results are available for viewing.")


@contextmanager
def suppress_live_plot_output():
    yield
