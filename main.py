import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
from itertools import combinations_with_replacement
import seaborn as sns
from multiprocessing.pool import Pool
import h5py

def plot_image(pixels):
    plt.imshow(pixels.reshape((28, 28)), cmap = 'gray')
    plt.show()

def bar_count(train_labels):
    numbers, counts = np.unique(train_labels, return_counts=True)
    fig = plt.figure(figsize = (4, 4))
    ax = fig.add_subplot()
    ax.set_xlabel("Цифры")
    ax.set_ylabel("Кол-во")
    ax.set_xticks(numbers)
    ax.bar(numbers,counts)
    for index, value in enumerate(counts):
        ax.text(index, value,
             str(value), color = "black")
    fig.savefig(fname = 'labelcount.svg', format = "svg")

def pixel_density(train_images, train_labels):
    images_overall = np.zeros((10, 784))
    for num in np.unique(train_labels):
        images_overall[num] = np.mean(train_images[train_labels == num], axis = 0)
    return images_overall
    
def plot_density(images_overall, name):
    x = np.arange(0, 784, 1)
    fig, axes = plt.subplots(10, figsize = (15, 25), sharey=True, sharex = True)
    fig.suptitle("Закрашиваемость клеток по цифрам")
    fig.supylabel("Закрашиваемость")

    for num, image in enumerate(images_overall):
        axes[num].set_xticks([k * 28 for k in range(1, 29)])
        axes[num].set_title(str(num))
        axes[num].bar(x, image, width = 20)
    
    fig.savefig(fname = name, format = "svg")


def distance_all(train_images1, train_images2, ident):
    print(f"вычисляю...{ident[0]}_{ident[1]}")
    dist_cos = distance.cdist(train_images1, train_images2, "cosine")
    dist_eucl = distance.cdist(train_images1, train_images2, "euclidean")

    with h5py.File(f"dist_{ident[0]}_{ident[1]}", 'w') as f:
            f.create_dataset('dist_cos', data=dist_cos)
            f.create_dataset('dist_eucl', data=dist_eucl)

def plot_all_distance(distances, name):
    n = 10
    fig, axes = plt.subplots(n, n, figsize = (50, 50))
    fig.suptitle('Гистограммы распределения расстояний')
    num = 0
    percentiles_difference = np.zeros((n, n))
    for j in range(0, n):
        for i in range(j, n):
            counts, _= np.histogram(distances[num], bins = 100)
            p75 = np.percentile(distances[num], 75)
            p25 = np.percentile(distances[num], 25)
            percentiles_difference[i][j] = p75 - p25
            axes[i][j].set_xlabel('Расстояние')
            axes[i][j].set_ylabel('Количество')
            axes[i][j].set_xlim(0, max(distances[num]))
            axes[i][j].set_ylim(0, max(counts))
            axes[i][j].set_title(f"{j} и {i}")
            axes[i][j].text(p75, 100000, str(round(np.percentile(distances[num], 75), 3)))
            axes[i][j].text(p25, 100000 * 1.75, str(round(np.percentile(distances[num], 25), 3)))
            axes[i][j].text(np.median(distances[num]), 100000 * 2.5, str(round(np.median(distances[num]), 3)))
            axes[i][j].text(np.mean(distances[num]), 100000 * 3.25, str(round(np.mean(distances[num]), 3)))
            axes[i][j].hist(distances[num], bins = 100)
            axes[i][j].vlines(x = p75, ymin = 0, ymax = max(counts), color = 'r', label = "75-й перцентиль")
            axes[i][j].vlines(x = p25, ymin = 0, ymax = max(counts), color = 'y', label = "25-й перцентиль")
            axes[i][j].vlines(x = np.median(distances[num]), ymin = 0, ymax = max(counts), color = (0.6, 0.2, 0.0), label = "Медиана")
            axes[i][j].vlines(x = np.mean(distances[num]), ymin = 0, ymax = max(counts), color = 'm', label = "Среднее")
            axes[i][j].legend()
            num += 1
    print("finished")
    fig.savefig(fname = name, format = "svg")

def distance_of_means(images_overall):
    distance_overall_cosine = distance.cdist(images_overall, images_overall, "cosine")
    distance_overall_eucl = distance.cdist(images_overall, images_overall, "euclidean")
    return distance_overall_cosine, distance_overall_eucl

def plot_distance(distance, name, titl, size):
    fig, axes = plt.subplots(1, 2, figsize = size)
    fig.suptitle(titl)
    names = ["Косинусовое расстояние", "L2-расстояние"]
    for num, d in enumerate(distance):
        axes[num].set(xlabel="Номер класса", ylabel="Номер класса")
        axes[num].set_title(names[num])
        sns.heatmap(d, ax = axes[num], annot=True)
    fig.savefig(fname = name, format = "svg")

def read_dist(id):
    with h5py.File(f"dist_{id[0]}_{id[1]}", 'r') as f:
        d1 = f['dist_cos']
        d2 = f['dist_eucl']
        dist_cos = d1[:]
        dist_eucl = d2[:]
    return dist_cos, dist_eucl

if __name__ == "__main__":
    import os
    print(os.listdir())
    with open('train-labels.idx1-ubyte', 'rb') as f:
        train_labels_data = f.read()

    train_labels = np.frombuffer(train_labels_data, dtype = 'uint8', offset = 8)

    train_images = []
    with open('train-images.idx3-ubyte', 'rb') as f:
        f.read(4 * 4)
        for _ in range (len(train_labels)):
            image = f.read(28 * 28)
            image_np = np.frombuffer(image, dtype = 'uint8') / 255      # делим на 255, чтобы значения были от 0 до 1
            train_images.append(image_np)

    train_images = np.array(train_images)

    print(np.shape(train_images))

    bar_count(train_labels)

    start = time.time()
    images_overall = pixel_density(train_images, train_labels)
    end = time.time()
    print("Время вычислений средних векторов: ", end - start)

    start = time.time()
    dist_cos, dist_eucl = distance_of_means(images_overall)
    end = time.time()
    print("Время вычислений расстояния между средними векторами: ", end - start)

    plot_distance([dist_cos, dist_eucl], "dist.svg", "Расстояния между средними векторами классов", (18,7))
"""
    start = time.time()

    images_classed = list(train_images[train_labels == i] for i in np.unique(train_labels))

    items = combinations_with_replacement(np.unique(train_labels), 2)

    slices_to_process = []
    for item in items:
        slices_to_process.append((train_images[train_labels==item[0]], train_images[train_labels==item[1]], item))

    start = time.time()
    #with Pool() as pool:
    #    pool.starmap(distance_all, slices_to_process)
    end = time.time()

    print("Время вычислений средних расстояний между сэмплами классов: ", end - start)
    items = combinations_with_replacement(np.unique(train_labels), 2)

    distances_cos = []
    counts_cos = []
    distances_eucl = []
    counts_eucl = []

    for item in list(items):
        dist_cos, dist_eucl = read_dist(item)
        #dist_cos, count_cos = np.unique(np.tril(dist_cos), return_counts=True)
        #dist_eucl, count_eucl = np.unique(np.tril(dist_eucl), return_counts=True)
        dist_cos = dist_cos[np.tril_indices_from(dist_cos)]
        dist_eucl = dist_eucl[np.tril_indices_from(dist_eucl)]
        distances_cos.append(dist_cos)
        distances_eucl.append(dist_eucl)
        #counts_cos.append(count_cos)
        #counts_eucl.append(count_eucl)
    
   # print(dist_cos)
   # print(np.shape(dist_cos))
   # print(count_cos)
   # plot_all_distance(distances_cos, counts_cos, "distances_cos.svg")
    
    plot_all_distance(distances_cos, "distances_between_samples_cos.svg")
    plot_all_distance(distances_eucl, "distances_between_samples_eucl.svg")
"""

    
    

