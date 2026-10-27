import numpy as np
import torch
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from matplotlib import pyplot as plt

from corrRise.mask_generator import generate_mask_set


def cosine_similarity(embedding_a, embedding_b):
    """Calculate cosine similarity between two embeddings."""
    a = np.asarray(embedding_a).ravel()
    b = np.asarray(embedding_b).ravel()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        raise ValueError("Cannot calculate cosine similarity for zero embeddings.")

    return float(np.dot(a, b) / denominator)


def pixelwise_pearson_correlation(masks, scores):
    """
    Calculate Pearson correlation between scores and mask values
    independently for every pixel.

    Parameters
    ----------
    masks : np.ndarray
        Shape (N, H, W).
    scores : np.ndarray
        Shape (N,).

    Returns
    -------
    saliency : np.ndarray
        Shape (H, W), with correlation coefficients in [-1, 1].
        Constant variables produce a correlation of zero.
    """
    n, height, width = masks.shape

    # Flatten spatial dimensions: (N, H*W).
    mask_matrix = masks.reshape(n, -1).astype(np.float64)

    # Center the mask values and similarity scores.
    centered_masks = mask_matrix - mask_matrix.mean(axis=0, keepdims=True)

    scores = np.asarray(scores, dtype=np.float64)
    centered_scores = scores - scores.mean()

    # Pearson correlation numerator for every pixel.
    numerator = centered_masks.T @ centered_scores

    # Pearson correlation denominator.
    mask_norm = np.sqrt(np.sum(centered_masks ** 2, axis=0))
    score_norm = np.sqrt(np.sum(centered_scores ** 2))

    denominator = mask_norm * score_norm

    # Define correlations as zero when either variable is constant.
    correlations = np.zeros(height * width, dtype=np.float64)

    valid = denominator > 0
    correlations[valid] = numerator[valid] / denominator[valid]

    # Protect against small floating-point errors.
    correlations = np.clip(correlations, -1.0, 1.0)

    return correlations.reshape(height, width)


def corrRISE(
    image_a,
    image_b,
    extract_embedding,
    extractor,
    masks
):
    """
    Generate CorrRISE saliency maps for two face images.

    Parameters
    ----------
    image_a, image_b : np.ndarray
        Images of identical shape (H, W, C).

    extract_embedding : callable
        User-provided function:
            extract_embedding(image) -> embedding

    masks : np.ndarray
        Pre-generated masks of shape (N, H, W).

    Returns
    -------
    dict containing:
        scores_a: Similarity scores with image A masked.
        scores_b: Similarity scores with image B masked.
        saliency_a: Signed correlation map for image A.
        saliency_b: Signed correlation map for image B.
        positive_a: Nonnegative correlations for image A.
        negative_a: Negative correlations for image A.
        positive_b: Nonnegative correlations for image B.
        negative_b: Negative correlations for image B.
    """
    image_a = np.asarray(image_a, dtype=np.float32)
    image_b = np.asarray(image_b, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)

    if image_a.shape != image_b.shape:
        raise ValueError("Images A and B must have identical shapes.")

    if image_a.ndim != 3:
        raise ValueError("Images must have shape (H, W, C).")

    height, width, _ = image_a.shape

    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError("Masks must have shape (N, H, W).")

    if masks.shape[0] < 2:
        raise ValueError("At least two masks are required.")

    # --------------------------------------------------
    # 1. Extract embeddings of the original images.
    # --------------------------------------------------
    embedding_a = np.asarray(
        extract_embedding(image_a, extractor)
    ).ravel()

    embedding_b = np.asarray(
        extract_embedding(image_b, extractor)
    ).ravel()

    # --------------------------------------------------
    # 2. Evaluate the masked images.
    # --------------------------------------------------
    num_masks = masks.shape[0]

    scores_a = np.empty(num_masks, dtype=np.float64)
    scores_b = np.empty(num_masks, dtype=np.float64)

    for k, mask in enumerate(masks):

        # Mask A; keep B unchanged.
        masked_a = image_a * mask[:, :, None]

        embedding_masked_a = np.asarray(
            extract_embedding(masked_a, extractor)
        ).ravel()

        scores_a[k] = cosine_similarity(
            embedding_masked_a,
            embedding_b
        )

        # Mask B; keep A unchanged.
        masked_b = image_b * mask[:, :, None]

        embedding_masked_b = np.asarray(
            extract_embedding(masked_b, extractor)
        ).ravel()

        scores_b[k] = cosine_similarity(
            embedding_masked_b,
            embedding_a
        )

    # --------------------------------------------------
    # 3. Compute pixel-wise Pearson correlations.
    # --------------------------------------------------
    saliency_a = pixelwise_pearson_correlation(
        masks, scores_a
    )

    saliency_b = pixelwise_pearson_correlation(
        masks, scores_b
    )

    # --------------------------------------------------
    # 4. Separate positive and negative correlations.
    # --------------------------------------------------
    # Preserve the spatial structure of the maps.
    positive_a = np.where(saliency_a >= 0, saliency_a, 0)
    negative_a = np.where(saliency_a < 0, saliency_a, 0)

    positive_b = np.where(saliency_b >= 0, saliency_b, 0)
    negative_b = np.where(saliency_b < 0, saliency_b, 0)

    return {
        "scores_a": scores_a,
        "scores_b": scores_b,
        "saliency_a": saliency_a,
        "saliency_b": saliency_b,
        "positive_a": positive_a,
        "negative_a": negative_a,
        "positive_b": positive_b,
        "negative_b": negative_b,
    }

def import_and_process_image(path, mtcnn):
    img = Image.open(path)
    print(img.size)
    face = np.array(mtcnn(img))
    face = np.moveaxis(face, 0, -1)
    return face

def extract_embedding(face, extractor, tolist=True):

    face = torch.Tensor(np.moveaxis(face, -1, 0))
    face = face.unsqueeze(0).to('cuda')

    with torch.no_grad():
        embedding = extractor(face)

    embedding = embedding / embedding.norm(dim=1, keepdim=True)

    if tolist:
        return embedding.squeeze().cpu().tolist()
    else:
        return embedding.squeeze()

if __name__ == "__main__":

    # image_a and image_b: NumPy arrays of shape (H, W, 3)
    # extract_embedding(image): returns a face embedding

    mtcnn = MTCNN(image_size=160, thresholds=[0.9, 0.9, 0.9], device=torch.device("cuda"))
    extractor = InceptionResnetV1(pretrained="vggface2", device=torch.device("cuda")).eval()

    face_1_path = "/home/wiktor/Doktorat/repositories/mavceleb_baseline/dataset/mavceleb_v1_train/faces/id0002/English/a8YwLpyHArc/0000100.jpg"
    face_2_path = "/home/wiktor/Doktorat/repositories/mavceleb_baseline/dataset/mavceleb_v1_train/faces/id0001/English/HpzYIwAaNlE/0001225.jpg"

    image_a = import_and_process_image(face_1_path, mtcnn)
    image_b = import_and_process_image(face_2_path, mtcnn)

    # plt.subplot(121), plt.imshow((image_a + 1) / 2)
    # plt.subplot(122), plt.imshow((image_b + 1) / 2)
    # plt.show()


    N = 1000

    masks = generate_mask_set(
        height=image_a.shape[0],
        width=image_a.shape[1],
        patch_size=16,
        num_patches=10,
        num_masks=N,
        seed=42
    )

    results = corrRISE(
        image_a=image_a,
        image_b=image_b,
        extract_embedding=extract_embedding,
        extractor=extractor,
        masks=masks
    )


    alpha = 0.4
    image_a_display = np.clip((image_a + 1.0) / 2.0, 0, 1)
    image_b_display = np.clip((image_b + 1.0) / 2.0, 0, 1)


    heatmap_a = np.abs(results["saliency_a"])
    heatmap_b = np.abs(results["saliency_b"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))


    axes[0, 0].imshow(image_a_display)
    im1 = axes[0, 0].imshow(
        results["positive_a"],
        cmap="jet",
        alpha=alpha
    )
    axes[0, 0].set_title("Image A: Similar Regions")
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04,
                 label="Positive correlation")


    axes[0, 1].imshow(image_a_display)
    im2 = axes[0, 1].imshow(
        np.abs(results["negative_a"]),
        cmap="jet",
        alpha=alpha
    )
    axes[0, 1].set_title("Image A: Dissimilar Regions")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04,
                 label="Absolute negative correlation")

    axes[1, 0].imshow(image_b_display)
    im3 = axes[1, 0].imshow(
        results["positive_b"],
        cmap="jet",
        alpha=alpha
    )
    axes[1, 0].set_title("Image B: Similar Regions")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04,
                 label="Positive correlation")


    axes[1, 1].imshow(image_b_display)
    im4 = axes[1, 1].imshow(
        np.abs(results["negative_b"]),
        cmap="jet",
        alpha=alpha
    )
    axes[1, 1].set_title("Image B: Dissimilar Regions")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04,
                 label="Absolute negative correlation")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()