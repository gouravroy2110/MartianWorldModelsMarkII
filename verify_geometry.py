import cv2
import numpy as np
import glob
import os

def compute_stereo_depth(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    scale = 0.5
    grayL = cv2.resize(grayL, (0,0), fx=scale, fy=scale)
    grayR = cv2.resize(grayR, (0,0), fx=scale, fy=scale)
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5, P1=8*3*25, P2=32*3*25, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp = cv2.medianBlur(disp, 5)
    disp = cv2.resize(disp, (imgL.shape[1], imgL.shape[0]))
    disp = np.clip(disp, 1.0, None)
    return 1000.0 / disp 

def compute_smooth_normals(depth):
    depth_smooth = cv2.GaussianBlur(depth, (31, 31), 0)
    dzdx = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=5)
    dzdy = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=5)
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    return normals / (norm + 1e-8)

def render_novel_view(img, depth, shift_x=30):
    H, W, _ = img.shape
    y, x = np.indices((H, W))
    shift_magnitude = (depth.max() - depth) / (depth.max() - depth.min() + 1e-5)
    new_x = np.clip(x + shift_x * shift_magnitude, 0, W-1).astype(int)
    warped = np.zeros_like(img)
    warped[y, new_x] = img[y, x]
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)

def virtual_flashlight(normals, light_dir):
    intensity = np.clip(np.dot(normals, light_dir), 0, 1)
    return (intensity * 255).astype(np.uint8)

def main():
    data_dir = '/kaggle/working/experiment_pairs'
    zl_files = sorted(glob.glob(os.path.join(data_dir, 'ZL*.png')))
    if not zl_files: return
    imgL = cv2.imread(zl_files[0])
    imgR = cv2.imread(zl_files[0].replace('ZL', 'ZR'))
    if imgR is None: return
    depth = compute_stereo_depth(imgL, imgR)
    normals = compute_smooth_normals(depth)
    cv2.imwrite(os.path.join(data_dir, 'testA_novel_view.png'), render_novel_view(imgL, depth, shift_x=40))
    l1 = np.array([0.8, -0.2, 0.5]); l1 = l1 / np.linalg.norm(l1)
    l2 = np.array([-0.8, -0.2, 0.5]); l2 = l2 / np.linalg.norm(l2)
    cv2.imwrite(os.path.join(data_dir, 'testB_flashlight_right.png'), virtual_flashlight(normals, l1))
    cv2.imwrite(os.path.join(data_dir, 'testB_flashlight_left.png'), virtual_flashlight(normals, l2))
    print('Done.')

if __name__ == '__main__': main()
