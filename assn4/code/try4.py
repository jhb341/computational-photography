import argparse
import os
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Develop a JPEG image from a RAW TIFF image (basic ISP: AWB, CFA interp, Gamma)")
    parser.add_argument('tiff', help='Input RAW TIFF file')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='Gamma value for correction (default: 2.2)')
    parser.add_argument('--contrast', type=float, default=None,
                        help='Optional contrast multiplier (e.g., 1.2)')
    parser.add_argument('--brightness', type=float, default=None,
                        help='Optional brightness offset (e.g., 10)')
    parser.add_argument('--sharpen', action='store_true',
                        help='Apply unsharp mask sharpening')
    parser.add_argument('--stretch', action='store_true',
                        help='Apply histogram stretching')
    parser.add_argument('-o', '--output', default=None,
                        help='Output JPEG filename (default: same as input with .jpg)')
    return parser.parse_args()




def automatic_white_balance(img):
    # Split channels
    B, G, R = cv2.split(img)
    # Compute per-channel averages
    B_avg = np.mean(B)
    G_avg = np.mean(G)
    R_avg = np.mean(R)
    # Compute gains
    gain_B = G_avg / (B_avg + 1e-8)
    gain_G = 1.0
    gain_R = G_avg / (R_avg + 1e-8)
    # Apply gains
    B = B * gain_B
    G = G * gain_G
    R = R * gain_R
    awb_img = cv2.merge([B, G, R])
    return awb_img    



def demosaic_bilinear(raw):
    """
    Simple bilinear CFA interpolation for GRBG pattern.
    raw: 2D float32 array
    returns: float32 BGR image
    """
    h, w = raw.shape
    # Allocate empty channels
    R = np.zeros((h, w), dtype=np.float32)
    G = np.zeros((h, w), dtype=np.float32)
    B = np.zeros((h, w), dtype=np.float32)
    # GRBG pattern:
    # G R
    # B G
    # Fill known samples
    G[0::2, 0::2] = raw[0::2, 0::2]
    R[0::2, 1::2] = raw[0::2, 1::2]
    B[1::2, 0::2] = raw[1::2, 0::2]
    G[1::2, 1::2] = raw[1::2, 1::2]
    
    # 아이폰
    #R[0::2, 0::2] = raw[0::2, 0::2]    # Row even, Col even = R
    #G[0::2, 1::2] = raw[0::2, 1::2]    # Row even, Col odd  = G
    #G[1::2, 0::2] = raw[1::2, 0::2]    # Row odd,  Col even = G
    #B[1::2, 1::2] = raw[1::2, 1::2]    # Row odd,  Col odd  = B

    # Define bilinear kernel for averaging 4 neighbors
    kernel = np.array([[0.25, 0.5, 0.25],
                       [0.5,  1.0, 0.5],
                       [0.25, 0.5, 0.25]], dtype=np.float32)

    # Interpolate missing values
    R = cv2.filter2D(R, -1, kernel)
    G = cv2.filter2D(G, -1, kernel)
    B = cv2.filter2D(B, -1, kernel)

    # Merge into BGR
    bgr = cv2.merge([B, G, R])
    return bgr


def gamma_correction(img, gamma):
    """
    Apply gamma correction: I_out = I_in^(1/gamma)
    img: float32 image, assumed normalized to [0,1]
    """
    # Avoid divide-by-zero
    img = np.clip(img, 0.0, 1.0)
    out = np.power(img, 1.0 / gamma)
    return out


def apply_optional_ops(img, contrast=None, brightness=None, sharpen=False, stretch=False):
    """
    Apply optional operations: contrast, brightness, histogram stretch, unsharp mask.
    img: uint8 BGR image
    """
    out = img.astype(np.float32)
    # Contrast and brightness
    if contrast is not None:
        out = out * contrast
    if brightness is not None:
        out = out + brightness
    # Clip after contrast/brightness
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    # Histogram stretching
    if stretch:
        chans = cv2.split(out)
        stretched = []
        for c in chans:
            c_min, c_max = np.min(c), np.max(c)
            if c_max > c_min:
                c = (c - c_min) / (c_max - c_min) * 255.0
            stretched.append(c)
        out = cv2.merge([c.astype(np.uint8) for c in stretched])

    # Sharpen (unsharp mask)
    if sharpen:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
        out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)

    return out


def main():
    args = parse_args()

    # Read RAW TIFF (single-channel)
    raw = cv2.imread(args.tiff, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    #raw = cv2.imread(args.tiff,
    #             cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH).astype(np.float32)

    if raw is None:
        print("Error: failed to load single-channel TIFF mosaic.")
        return
    
    if raw.ndim == 3:
        raw = raw[..., 0]
        
    raw = raw.astype(np.float32)

    # 1) CFA interpolation (demosaic)
    bgr = demosaic_bilinear(raw)

    # 2) Automatic white balance (gray-world)
    awb = automatic_white_balance(bgr)

    # 3) Normalize to [0,1]
    max_val = np.max(awb)
    if max_val <= 0:
        max_val = 1.0
    norm = awb / max_val


    # 4) Gamma correction
    gamma_img = gamma_correction(norm, args.gamma)

    # 5) Convert to 8-bit
    img8 = np.clip(gamma_img * 255.0, 0, 255).astype(np.uint8)

    # 6) Optional operations
    final = apply_optional_ops(
        img8,
        contrast=args.contrast,
        brightness=args.brightness,
        sharpen=args.sharpen,
        stretch=args.stretch
    )

    # Write output JPEG
    out_path = args.output if args.output else os.path.splitext(args.tiff)[0] + '.jpg'
    cv2.imwrite(out_path, final, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved JPEG to {out_path}")


if __name__ == '__main__':
    main()