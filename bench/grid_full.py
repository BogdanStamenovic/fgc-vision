"""Full 1920x1080 frame, field region y 150-900, absolute 50 px grid, labels every 100 px on
BOTH edges and mid-frame, no resizing (so displayed px = absolute px - (0,150))."""
import sys, cv2
im = cv2.imread(sys.argv[1])
for x in range(0, 1920, 50):
    cv2.line(im, (x, 0), (x, 1080), (0, 220, 255) if x % 100 == 0 else (0, 100, 130), 1)
    if x % 100 == 0:
        for yy in (165, 520, 890): cv2.putText(im, str(x), (x + 2, yy), 0, 0.5, (0, 255, 255), 1)
for y in range(150, 900, 50):
    cv2.line(im, (0, y), (1920, y), (0, 220, 255) if y % 100 == 0 else (0, 100, 130), 1)
    if y % 100 == 0:
        for xx in (2, 960, 1870): cv2.putText(im, str(y), (xx, y - 2), 0, 0.5, (0, 255, 255), 1)
cv2.imwrite(sys.argv[2], im[150:900, 150:1800], [cv2.IMWRITE_JPEG_QUALITY, 92])
