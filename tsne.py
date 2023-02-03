# # bbox = [
# # [603.0, 556.0, 8.0, 10.0],
# # [465.0, 64.0, 12.0, 34.0],
# # [595.0, 550.0, 8.0, 13.0],
# # [636.0, 557.0, 4.0, 11.0],
# # [583.0, 557.0, 7.0, 13.0],
# # [632.0, 555.0, 5.0, 10.0],
# # [585.0, 530.0, 46.0, 13.0],
# # [566.0, 536.0, 12.0, 7.0],
# # [561.0, 536.0, 6.0, 12.0],
# # [466.0, 105.0, 12.0, 30.0],
# # [471.0, 134.0, 13.0, 34.0],
# # [531.0, 421.0, 53.0, 22.0],
# # [759.0, 400.0, 29.0, 25.0],
# # ]
#
# bbox = [
# [337.0, 140.0, 24.0, 19.0],
# [387.0, 157.0, 10.0, 13.0],
# [377.0, 154.0, 10.0, 16.0],
# [557.0, 34.0, 13.0, 20.0],
# [624.0, 180.0, 13.0, 4.0],
# [491.0, 187.0, 11.0, 16.0],
# [315.0, 126.0, 27.0, 22.0],
# [468.0, 188.0, 12.0, 11.0],
# ]
#
# import cv2 as cv
#
# img = cv.imread("15.jpg")
# for i in range(len(bbox)):
#     cv.rectangle(img, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3])), (0, 255, 0), 1)
# cv.imwrite("15_1.jpg", img)

from time import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)
X = []
for i in range(100):
    img = cv.imread('png_opt/' + str(i) + '_2.jpg')
    img = cv.resize(img, (600, 600))
    X.append(img)
for i in range(100):
    img = cv.imread('png/' + str(i) + '_2.jpg')
    img = cv.resize(img, (600, 600))
    X.append(img)
X = np.array(X)
X = X.reshape(200,-1)
# digits = datasets.load_digits(n_class=10) #����������
# X = digits.data
# y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(100):
        plt.scatter(X[i, 0], X[i, 1], c='g')
    for i in range(100, 200):
        plt.scatter(X[i, 0], X[i, 1], c='b')
    # if hasattr(offsetbox, 'AnnotationBbox'): #(������������������������������)
    #     shown_images = np.array([[1., 1.]])
    # for i in range(digits.data.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]] #����������������������������������������������������
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    # ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    # if title is not None:
    #     plt.title(title)

#----------------------------------------------------------------------

# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')
# # ����PCA
# print("Computing PCA projection")
# t0 = time()
# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
# plot_embedding(X_pca,
#                "Principal Components projection of the digits (time %.2fs)" %
#                (time() - t0))
# ����t-SNE
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()
print('ok')