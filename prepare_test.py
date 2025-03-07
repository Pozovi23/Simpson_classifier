import os
import csv
import matplotlib.pyplot as plt
from check_data import check_file_extension


main_path = 'data/kaggle_simpson_testset'


def read_labels():
    labels = []
    with open('labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            labels.append(line)

    return labels


def prepare_test_csv():
    files_with_wrong_extension = check_file_extension(main_path)
    labels = read_labels()
    with open("./simpson_test_set.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        index_of_current_label = 0
        for folder in sorted(os.listdir(main_path)):
            for file in sorted(os.listdir(main_path + '/' + folder)):
                current_path = main_path + '/' + folder + '/' + file
                if current_path not in files_with_wrong_extension:
                    if labels[index_of_current_label][1] in file:
                        writer.writerow([labels[index_of_current_label][0], current_path])
                    else:
                        while labels[index_of_current_label][1] not in file:
                            index_of_current_label += 1
                        writer.writerow([labels[index_of_current_label][0], current_path])


def show_difference():
    classes_and_amount_of_pictures = {}
    for i in range(42):
        classes_and_amount_of_pictures[i] = 0

    with open("./simpson_test_set.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            classes_and_amount_of_pictures[int(line[0])] += 1

    all_classes = sorted(classes_and_amount_of_pictures.keys())
    image_counts = []
    for class_ in all_classes:
        image_counts.append(classes_and_amount_of_pictures[class_])

    plt.figure(figsize=(12, 11))
    bars = plt.bar(all_classes, image_counts, width=0.8)
    plt.xlabel('Классы')
    plt.ylabel('Количество картинок')
    plt.title('Количество картинок в каждом классе')
    plt.xticks(all_classes, labels=[str(c) for c in all_classes], rotation=0)

    for i in range(len(image_counts)):
        yval = bars[i].get_height()
        plt.text(bars[i].get_x() + bars[i].get_width() / 2, yval + 0.5, str(image_counts[i]), ha='center', va='bottom', rotation=90)

    plt.savefig('hist.png')


show_difference()