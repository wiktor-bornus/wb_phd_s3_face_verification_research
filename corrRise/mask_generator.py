import numpy as np
import matplotlib.pyplot as plt


def generate_random_mask(
    height: int,
    width: int,
    patch_size: int,
    num_patches: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate one CorrRISE-style random mask.

    Parameters
    ----------
    height : int
        Height of the mask.
    width : int
        Width of the mask.
    patch_size : int
        Side length of each square patch in pixels.
    num_patches : int
        Number of square patches to place in the mask.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    mask : np.ndarray
        Float32 array of shape (height, width), with values in [0, 1].
        The background is 1.0, and each patch receives a random value.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    if patch_size > min(height, width):
        raise ValueError("patch_size cannot exceed the image dimensions.")

    if num_patches < 0:
        raise ValueError("num_patches cannot be negative.")

    if rng is None:
        rng = np.random.default_rng()

    # Initialize the mask to preserve the original image.
    mask = np.ones((height, width), dtype=np.float32)

    # Place square patches at random locations.
    for _ in range(num_patches):
        # Sample the top-left corner of the patch.
        y = rng.integers(0, height - patch_size + 1)
        x = rng.integers(0, width - patch_size + 1)

        # Sample a random patch value in [0, 1).
        value = rng.random()

        # Assign the same value to every pixel in this square patch.
        mask[
            y:y + patch_size,
            x:x + patch_size
        ] = value

    return mask


def generate_mask_set(
    height: int,
    width: int,
    patch_size: int,
    num_patches: int,
    num_masks: int,
    seed: int = None
) -> np.ndarray:
    """
    Generate a set of CorrRISE-style random masks.

    Returns
    -------
    masks : np.ndarray
        Float32 array of shape (num_masks, height, width).
    """
    if num_masks <= 0:
        raise ValueError("num_masks must be positive.")

    rng = np.random.default_rng(seed)

    masks = np.empty(
        (num_masks, height, width),
        dtype=np.float32
    )

    for i in range(num_masks):
        masks[i] = generate_random_mask(
            height=height,
            width=width,
            patch_size=patch_size,
            num_patches=num_patches,
            rng=rng
        )

    return masks



if __name__ == "__main__":
    # Mask generation parameters
    HEIGHT = 112
    WIDTH = 112
    PATCH_SIZE = 30
    NUM_PATCHES = 10
    NUM_MASKS = 1000

    # Visualization parameters
    ROWS = 4
    COLS = 5

    # Generate masks
    masks = generate_mask_set(
        height=HEIGHT,
        width=WIDTH,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        num_masks=NUM_MASKS,
        seed=42
    )

    print("Mask array shape:", masks.shape)
    print("Minimum mask value:", masks.min())
    print("Maximum mask value:", masks.max())

    # Plot a grid of example masks
    num_to_plot = min(ROWS * COLS, NUM_MASKS)

    fig, axes = plt.subplots(
        ROWS,
        COLS,
        figsize=(COLS * 2.5, ROWS * 2.5)
    )

    # Flatten axes to make indexing easier for any grid size
    axes = np.array(axes).reshape(-1)

    for i in range(ROWS * COLS):
        ax = axes[i]

        if i < num_to_plot:
            ax.imshow(
                masks[i],
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )
            ax.set_title(f"Mask {i + 1}")
        else:
            ax.axis("off")

        ax.axis("off")

    plt.tight_layout()
    plt.show()
