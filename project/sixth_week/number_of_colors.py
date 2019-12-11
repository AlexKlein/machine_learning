"""The task of machine learning - check the reaction of the molecule
================================================

Contents
--------
Script checks the reaction of the molecule.

Task
----
0. Download the picture parrots.jpg. Transform the image by casting all the values ​​in the interval from 0 to 1.
1. Create a matrix of features-objects: characterize each pixel with three coordinates.
2. Run the K-Means algorithm with the parameters init = 'k-means ++' and random_state = 241.
3. Measure the quality of the resulting segmentation using the PSNR metric.
4. Find the minimum number of clusters at which the PSNR value is above 20.

"""
from copy import deepcopy

from numpy import shape, reshape
from pandas import DataFrame
from skimage.feature.corner_cy import img_as_float
from skimage.io import imread, imsave
from sklearn.cluster import KMeans
from skimage.measure import compare_psnr


if __name__ == '__main__':
    output_file = open('output.txt', 'w', encoding='ANSI')
    image_matrix = img_as_float(
        imread(
            'parrots.jpg'
        )
    )

    length, height, width = shape(image_matrix)
    X = DataFrame(reshape(image_matrix, (length * height, width)))

    for i in range(1, 21):
        clf = KMeans(
            n_clusters=i,
            init='k-means++',
            random_state=241
        )
        X_temp = deepcopy(X)
        X_temp['cluster'] = clf.fit_predict(X)

        mean_groups = X_temp.groupby('cluster').mean().values
        mean_temp = [mean_groups[c] for c in X_temp['cluster'].values]
        mean_image = reshape(mean_temp, (length, height, width))
        imsave('parrots_mean_' + str(i) + '.jpg', mean_image)

        medians = X_temp.groupby('cluster').median().values
        median_pixels = [medians[c] for c in X_temp['cluster'].values]
        median_image = reshape(median_pixels, (length, height, width))
        imsave('parrots_median_' + str(i) + '.jpg', median_image)

        psnr_mean = compare_psnr(image_matrix, mean_image)
        psnr_median = compare_psnr(image_matrix, median_image)

        if psnr_mean >= 20:
            print(
                i,
                psnr_mean,
                'mean'
            )
            print(
                i,
                file=output_file
            )
            break

        if psnr_median >= 20:
            print(
                i,
                psnr_median,
                'median'
            )
            print(
                i,
                file=output_file
            )
            break

    output_file.close()
