import os
import hashlib
import numpy as np
import cv2
import typing
import io
import validators
import urllib.request
import urllib.parse
from scipy import spatial
from shapely import geometry
import matplotlib.pyplot as plt
import typing_extensions as tx


def read(filepath_or_buffer: typing.Union[str, io.BytesIO, np.ndarray]):
    """Read a file into an image object

    Args:
        filepath_or_buffer: The path to the file, a URL, or any object
            with a `read` method (such as `io.BytesIO`)
    """
    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer
    if hasattr(filepath_or_buffer, "read"):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)  # type: ignore
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(filepath_or_buffer, str):
        if validators.url(filepath_or_buffer):
            return read(urllib.request.urlopen(filepath_or_buffer))
        assert os.path.isfile(filepath_or_buffer), (
            "Could not find image at path: " + filepath_or_buffer
        )
        image = cv2.imread(filepath_or_buffer)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sha256sum(filename):
    """Compute the sha256 hash for a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):  # type: ignore
            h.update(mv[:n])
    return h.hexdigest()


def get_default_cache_dir():
    return os.environ.get(
        "KERAS_OCR_CACHE_DIR", os.path.expanduser(os.path.join("~", ".keras-ocr"))
    )


def download_and_verify(url, sha256=None, cache_dir=None, verbose=True, filename=None):
    """Download a file to a cache directory and verify it with a sha256
    hash.

    Args:
        url: The file to download
        sha256: The sha256 hash to check. If the file already exists and the hash
            matches, we don't download it again.
        cache_dir: The directory in which to cache the file. The default is
            `~/.keras-ocr`.
        verbose: Whether to log progress
        filename: The filename to use for the file. By default, the filename is
            derived from the URL.
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    if filename is None:
        filename = os.path.basename(urllib.parse.urlparse(url).path)
    filepath = os.path.join(cache_dir, filename)
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    if verbose:
        print("Looking for " + filepath)
    if not os.path.isfile(filepath) or (sha256 and sha256sum(filepath) != sha256):
        if verbose:
            print("Downloading " + filepath)
        urllib.request.urlretrieve(url, filepath)
    assert sha256 is None or sha256 == sha256sum(
        filepath
    ), "Error occurred verifying sha256."
    return filepath


def pad(image, width: int, height: int, cval: int = 255):
    """Pad an image to a desired size. Raises an exception if image
    is larger than desired size.

    Args:
        image: The input image
        width: The output width
        height: The output height
        cval: The value to use for filling the image.
    """
    output_shape: typing.Union[typing.Tuple[int, int, int], typing.Tuple[int, int]]
    if len(image.shape) == 3:
        output_shape = (height, width, image.shape[-1])
    else:
        output_shape = (height, width)
    assert height >= output_shape[0], "Input height must be less than output height."
    assert width >= output_shape[1], "Input width must be less than output width."
    padded = np.zeros(output_shape, dtype=image.dtype) + cval
    padded[: image.shape[0], : image.shape[1]] = image
    return padded


def resize_image(image, max_scale, max_size):
    """Obtain the optimal resized image subject to a maximum scale
    and maximum size.

    Args:
        image: The input image
        max_scale: The maximum scale to apply
        max_size: The maximum size to return
    """
    if max(image.shape) * max_scale > max_size:
        # We are constrained by the maximum size
        scale = max_size / max(image.shape)
    else:
        # We are contrained by scale
        scale = max_scale
    return (
        cv2.resize(
            image, dsize=(int(image.shape[1] * scale), int(image.shape[0] * scale))
        ),
        scale,
    )


# pylint:disable=too-many-locals
def warpBox(
    image,
    box,
    target_height=None,
    target_width=None,
    margin=0,
    cval=None,
    return_transform=False,
    skip_rotate=False,
):
    """Warp a boxed region in an image given by a set of four points into
    a rectangle with a specified width and height. Useful for taking crops
    of distorted or rotated text.

    Args:
        image: The image from which to take the box
        box: A list of four points starting in the top left
            corner and moving clockwise.
        target_height: The height of the output rectangle
        target_width: The width of the output rectangle
        return_transform: Whether to return the transformation
            matrix with the image.
    """
    if cval is None:
        cval = (0, 0, 0) if len(image.shape) == 3 else 0
    if not skip_rotate:
        box, _ = get_rotated_box(box)
    w, h = get_rotated_width_height(box)
    assert (target_width is None and target_height is None) or (
        target_width is not None and target_height is not None
    ), "Either both or neither of target width and height must be provided."
    if target_width is None and target_height is None:
        target_width = w
        target_height = h
    scale = min(target_width / w, target_height / h)
    M = cv2.getPerspectiveTransform(
        src=box,
        dst=np.array(
            [
                [margin, margin],
                [scale * w - margin, margin],
                [scale * w - margin, scale * h - margin],
                [margin, scale * h - margin],
            ]
        ).astype("float32"),
    )
    crop = cv2.warpPerspective(image, M, dsize=(int(scale * w), int(scale * h)))
    target_shape = (
        (target_height, target_width, 3)
        if len(image.shape) == 3
        else (target_height, target_width)
    )
    full = (np.zeros(target_shape) + cval).astype("uint8")
    full[: crop.shape[0], : crop.shape[1]] = crop
    if return_transform:
        return full, M
    return full


def drawAnnotations(image, predictions, ax=None):
    """Draw text annotations onto image.

    Args:
        image: The image on which to draw
        predictions: The predictions as provided by `pipeline.recognize`.
        ax: A matplotlib axis on which to draw.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(drawBoxes(image=image, boxes=predictions, boxes_format="predictions"))
    predictions = sorted(predictions, key=lambda p: p[1][:, 1].min())
    left = []
    right = []
    for word, box in predictions:
        if box[:, 0].min() < image.shape[1] / 2:
            left.append((word, box))
        else:
            right.append((word, box))
    ax.set_yticks([])
    ax.set_xticks([])
    for side, group in zip(["left", "right"], [left, right]):
        for index, (text, box) in enumerate(group):
            y = 1 - (index / len(group))
            xy = box[0] / np.array([image.shape[1], image.shape[0]])
            xy[1] = 1 - xy[1]
            ax.annotate(
                text=text,
                xy=xy,
                xytext=(-0.05 if side == "left" else 1.05, y),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "r"},
                color="r",
                fontsize=14,
                horizontalalignment="right" if side == "left" else "left",
            )
    return ax


def drawBoxes(image, boxes, color=(255, 0, 0), thickness=5, boxes_format="boxes"):
    """Draw boxes onto an image.

    Args:
        image: The image on which to draw the boxes.
        boxes: The boxes to draw.
        color: The color for each box.
        thickness: The thickness for each box.
        boxes_format: The format used for providing the boxes. Options are
            "boxes" which indicates an array with shape(N, 4, 2) where N is the
            number of boxes and each box is a list of four points) as provided
            by `keras_ocr.detection.Detector.detect`, "lines" (a list of
            lines where each line itself is a list of (box, character) tuples) as
            provided by `keras_ocr.data_generation.get_image_generator`,
            or "predictions" where boxes is by itself a list of (word, box) tuples
            as provided by `keras_ocr.pipeline.Pipeline.recognize` or
            `keras_ocr.recognition.Recognizer.recognize_from_boxes`.
    """
    if len(boxes) == 0:
        return image
    canvas = image.copy()
    if boxes_format == "lines":
        revised_boxes = []
        for line in boxes:
            for box, _ in line:
                revised_boxes.append(box)
        boxes = revised_boxes
    if boxes_format == "predictions":
        revised_boxes = []
        for _, box in boxes:
            revised_boxes.append(box)
        boxes = revised_boxes
    for box in boxes:
        cv2.polylines(
            img=canvas,
            pts=box[np.newaxis].astype("int32"),
            color=color,
            thickness=thickness,
            isClosed=True,
        )
    return canvas


def get_rotated_width_height(box):
    """
    Returns the width and height of a rotated rectangle

    Args:
        box: A list of four points starting in the top left
        corner and moving clockwise.
    """
    w = (
        spatial.distance.cdist(box[0][np.newaxis], box[1][np.newaxis], "euclidean")
        + spatial.distance.cdist(box[2][np.newaxis], box[3][np.newaxis], "euclidean")
    ) / 2
    h = (
        spatial.distance.cdist(box[0][np.newaxis], box[3][np.newaxis], "euclidean")
        + spatial.distance.cdist(box[1][np.newaxis], box[2][np.newaxis], "euclidean")
    ) / 2
    return int(w[0][0]), int(h[0][0])


def get_rotated_box(
    points,
) -> typing.Tuple[np.ndarray, float,]:
    """Obtain the parameters of a rotated box.

    Returns:
        The vertices of the rotated box in top-left,
        top-right, bottom-right, bottom-left order along
        with the angle of rotation about the bottom left corner.
    """
    try:
        mp = geometry.MultiPoint(points=points)
        pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[
            :-1
        ]  # noqa: E501
    except AttributeError:
        # There weren't enough points for the minimum rotated rectangle function
        pts = points
    # The code below is taken from
    # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    pts = np.array([tl, tr, br, bl], dtype="float32")

    rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
    return pts, rotation


def adjust_boxes(
    boxes,
    scale=1,
    boxes_format: tx.Literal["boxes", "predictions", "lines"] = "boxes",
) -> typing.Union[
    np.ndarray,
    typing.List[typing.List[typing.Tuple[np.ndarray, str]]],
    typing.List[typing.Tuple[str, np.ndarray]],
]:
    """Adjust boxes using a given scale and offset.

    Args:
        boxes: The boxes to adjust
        boxes_format: The format for the boxes. See the `drawBoxes` function
            for an explanation on the options.
        scale: The scale to apply
    """
    if scale == 1:
        return boxes
    if boxes_format == "boxes":
        return np.array(boxes) * scale
    if boxes_format == "lines":
        return [
            [(np.array(box) * scale, character) for box, character in line]
            for line in boxes
        ]
    if boxes_format == "predictions":
        return [(word, np.array(box) * scale) for word, box in boxes]
    raise NotImplementedError(f"Unsupported boxes format: {boxes_format}")
