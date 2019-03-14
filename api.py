import cv2
import numpy as np

from closed_form_matting import closed_form_matting_with_trimap


def do_matting(image, trimap, max_size=2000, confidence=100.0):
    assert image.shape[0:2] == trimap.shape[0:2]
    ori_h, ori_w = image.shape[0:2]
    scale = max_size / max(ori_h, ori_w)
    new_h, new_w = int(scale * ori_h), int(scale * ori_w)
    
    image = cv2.resize(image, (new_w, new_h))
    trimap = cv2.resize(trimap, (new_w, new_h))
    
    image = image / 255.0
    trimap = trimap / 255.0

    alpha = closed_form_matting_with_trimap(image, trimap, trimap_confidence=confidence)
    alpha *= 255.0
    alpha = alpha.astype(np.uint8)
    alpha = cv2.resize(alpha, (ori_w, ori_h))
    return alpha


if __name__ == '__main__':
    image = cv2.imread('image.png', cv2.IMREAD_COLOR)
    trimap = cv2.imread('trimap.png', cv2.IMREAD_GRAYSCALE)
    alpha = do_matting(image, trimap)
    alpha = np.where(alpha > 20, 255, 0)
    cv2.imwrite('out2.png', alpha)