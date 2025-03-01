import os
import csv
from check_data import check_file_extension


main_path = 'data/kaggle_simpson_testset'


def read_labels():
    labels = []
    with open('labels.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            labels.append(line)

    return labels


def prepare_train_csv():
    files_with_wrong_extension = check_file_extension(main_path)
    labels = read_labels()
    with open("./simpson_testset.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Number_of_class", "Path_to_img"])
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


prepare_train_csv()