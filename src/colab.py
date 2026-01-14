import cv2

try:
    colab = __import__('google.colab.patches')
    is_colab = True
except ModuleNotFoundError:
    is_colab = False

def _cv2_imshow_colab(
    _: str,
    mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat
) -> None:
    colab.cv2_imshow(mat)

cv2_imshow = _cv2_imshow_colab if is_colab else cv2.imshow
