"""Clean labelling view: frame crop with a 50 px grid (labels every 100 px), no model output."""
import sys, cv2
f, out, x0, y0, x1, y1 = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
im = cv2.imread(f)
for x in range(0, 1920, 50):
    cv2.line(im, (x, 0), (x, 1080), (0, 220, 255) if x % 100 == 0 else (0, 110, 140), 1)
    if x % 100 == 0: cv2.putText(im, str(x), (x + 2, y0 + 14), 0, 0.45, (0, 255, 255), 1)
for y in range(0, 1080, 50):
    cv2.line(im, (0, y), (1920, y), (0, 220, 255) if y % 100 == 0 else (0, 110, 140), 1)
    if y % 100 == 0: cv2.putText(im, str(y), (x0 + 2, y - 2), 0, 0.45, (0, 255, 255), 1)
cv2.imwrite(out, im[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
