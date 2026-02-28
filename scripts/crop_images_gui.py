#!/usr/bin/env python3
import os
import glob
import cv2
import argparse

DEFAULT_DIR = "/home/haziq/datasets/gimo/my_scripts/paper/ours_vs_dimop3d/new/"


class Cropper:
    def __init__(self, img_path):
        self.img_path = img_path
        self.orig = cv2.imread(img_path)
        if self.orig is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        h, w = self.orig.shape[:2]
        self.scale = 1.0
        max_w = 1400
        if w > max_w:
            self.scale = max_w / w
            self.disp = cv2.resize(self.orig, (int(w * self.scale), int(h * self.scale)))
        else:
            self.disp = self.orig.copy()

        self.points = []
        self.window = 'cropper'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse_cb)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # map display coords back to original
            ox = int(x / self.scale)
            oy = int(y / self.scale)
            if len(self.points) < 2:
                self.points.append((ox, oy))

    def draw(self):
        img = self.disp.copy()
        if len(self.points) >= 1:
            x1, y1 = self.points[0]
            x1d, y1d = int(x1 * self.scale), int(y1 * self.scale)
            cv2.circle(img, (x1d, y1d), 4, (0, 255, 0), -1)
        if len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            x1d, y1d = int(x1 * self.scale), int(y1 * self.scale)
            x2d, y2d = int(x2 * self.scale), int(y2 * self.scale)
            cv2.rectangle(img, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)
        return img

    def get_bbox(self):
        if len(self.points) != 2:
            return None
        (x1, y1), (x2, y2) = self.points
        x1, x2 = sorted([max(0, x1), max(0, x2)])
        y1, y2 = sorted([max(0, y1), max(0, y2)])
        h, w = self.orig.shape[:2]
        x1 = min(x1, w - 1)
        x2 = min(x2, w - 1)
        y1 = min(y1, h - 1)
        y2 = min(y2, h - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)


def crop_all_images(img_dir, bbox):
    pattern = os.path.join(img_dir, '*.png')
    files = sorted(glob.glob(pattern))
    if not files:
        print('No PNG images found in', img_dir)
        return
    x1, y1, x2, y2 = bbox
    print(f'Cropping {len(files)} images to bbox: {(x1,y1,x2,y2)}')
    for f in files:
        im = cv2.imread(f)
        if im is None:
            print('Failed to read', f)
            continue
        h, w = im.shape[:2]
        # clamp bbox to image
        sx1 = max(0, min(x1, w - 1))
        sx2 = max(0, min(x2, w))
        sy1 = max(0, min(y1, h - 1))
        sy2 = max(0, min(y2, h))
        if sx2 <= sx1 or sy2 <= sy1:
            print('Invalid bbox for', f)
            continue
        crop = im[sy1:sy2, sx1:sx2]
        base, ext = os.path.splitext(f)
        out = base + '_cropped.png'
        cv2.imwrite(out, crop)
    print('Done.')


def main():
    parser = argparse.ArgumentParser(description='Select bbox by clicking two corners and crop all PNGs')
    parser.add_argument('--dir', '-d', default=DEFAULT_DIR, help='Directory with PNG images')
    args = parser.parse_args()
    img_dir = args.dir
    if not os.path.isdir(img_dir):
        print('Directory not found:', img_dir)
        return
    pattern = os.path.join(img_dir, '*.png')
    files = sorted(glob.glob(pattern))
    if not files:
        print('No PNG images in', img_dir)
        return

    first = files[0]
    cropper = Cropper(first)

    print('Instructions:')
    print('- Click top-left then bottom-right corner of the bbox.')
    print("- Press Enter to confirm and crop all images.")
    print("- Press 'r' to reset selection, 'q' to quit without saving.")

    while True:
        disp = cropper.draw()
        cv2.imshow(cropper.window, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            print('Quitting without cropping.')
            break
        if k == ord('r'):
            cropper.points = []
            print('Selection reset.')
        # Enter keys: 13 (CR) or 10 (LF)
        if k in (13, 10):
            bbox = cropper.get_bbox()
            if bbox is None:
                print('Please select two valid points first.')
                continue
            crop_all_images(img_dir, bbox)
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
