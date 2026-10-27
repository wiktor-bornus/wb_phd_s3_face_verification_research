from pathlib import Path
import random

def get_random_face_path(
    person_id: str,
    dataset_root: str = "/home/wiktor/Doktorat/repositories/mavceleb_baseline/dataset/mavceleb_v1_train/faces",
    exclude_path: str = None
) -> str:
    """
    Return the path to a random face image belonging to a given identity.

    Parameters
    ----------
    person_id : str
        Identity folder name, e.g. "id0002".
    dataset_root : str
        Root directory containing identity folders.
    exclude_path : str, optional
        Image path to exclude from selection.

    Returns
    -------
    str
        Absolute path to a randomly selected image.
    """
    person_dir = Path(dataset_root) / person_id

    if not person_dir.is_dir():
        raise FileNotFoundError(f"Identity directory not found: {person_dir}")

    extensions = {".jpg", ".jpeg", ".png"}

    image_paths = [
        path for path in person_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]

    if exclude_path is not None:
        exclude_path = Path(exclude_path).resolve()
        image_paths = [
            path for path in image_paths
            if path.resolve() != exclude_path
        ]

    if not image_paths:
        raise FileNotFoundError(
            f"No eligible images found for identity {person_id}"
        )

    return str(random.choice(image_paths))