from imagedt.decorator import time_cost
import cv2 as cv


def brg2rgb(image):
    (r, g, b) = cv.split(image)
    return cv.merge([b, g, r])


def orb_detect(image_source, image_aim):
    orb = cv.ORB_create()

    kp1, des1 = orb.detectAndCompute(image_source, None)
    kp2, des2 = orb.detectAndCompute(image_aim, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    image_result = cv.drawMatches(image_source, kp1, image_aim, kp2, matches[:100], None)

    return brg2rgb(image_result)


@time_cost
def sift_detect(image_1, image_2, detector='surf'):
    if detector.startswith('si'):
        print("sift detector....")
        surf = cv.xfeatures2d.SURF_create()
    else:
        print("surf detector.......")
        surf = cv.xfeatures2d.SURF_create()

    kp1, des1 = surf.detectAndCompute(image_1, None)
    kp2, des2 = surf.detectAndCompute(image_2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]

    image_3 = cv.drawMatchesKnn(image_1, kp1, image_2, kp2, good, None, flags=2)

    return image_3
    # return brg2rgb(image_3)


if __name__ == '__main__':
    image_a = cv.imread("image/source_image.jpg")
    image_b = cv.imread("image/aim_image2.jpg")

    image = sift_detect(image_a, image_b)

    image = cv.resize(image, (1280, 720), interpolation=cv.INTER_CUBIC)

    cv.imshow("result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # plt.imshow(image)
    # plt.show()
