import os
import csv
import matplotlib.pyplot as plt

from check_data import find_duplicates
from check_data import check_file_extension


main_path = 'data/simpsons_dataset'


def make_labels_dict(labels):
    number_of_class = 0
    for folder in sorted(os.listdir(main_path)):
        labels[folder] = number_of_class
        number_of_class += 1

    with open("./labels.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key in labels.keys():
            writer.writerow([labels[key], key])


def prepare_train_val_csv():
    files_with_wrong_extension = check_file_extension(main_path)
    duplicates = find_duplicates(main_path)
    labels = {}
    make_labels_dict(labels)
    with open("./simpson_train_set.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Number_of_class", "Path_to_img"])
        for folder in sorted(os.listdir(main_path)):
            files = sorted(os.listdir(main_path + '/' + folder))
            for i in range(int(len(files) * 0.85)):
                current_path = main_path + '/' + folder + '/' + files[i]
                if (current_path not in duplicates) and (current_path not in files_with_wrong_extension):
                    writer.writerow([labels[folder], current_path])

    with open("./simpson_validation_set.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Number_of_class", "Path_to_img"])
        for folder in sorted(os.listdir(main_path)):
            files = sorted(os.listdir(main_path + '/' + folder))
            for i in range(int(len(files) * 0.85), len(files)):
                current_path = main_path + '/' + folder + '/' + files[i]
                if (current_path not in duplicates) and (current_path not in files_with_wrong_extension):
                    writer.writerow([labels[folder], current_path])


def show_difference():
    classes_and_amount_of_pictures = {}
    with open("./simpson_train_set.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        head = next(reader)
        for line in reader:
            if classes_and_amount_of_pictures.get(int(line[0])) is None:
                classes_and_amount_of_pictures[int(line[0])] = 1
            else:
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
