import os
import hashlib


def hash_image(image_path):
    with open(image_path, 'rb') as image_file:
        hasher = hashlib.md5()
        image = image_file.read()
        hasher.update(image)
        digest = hasher.digest()

    return digest


def find_duplicates(main_path):
    duplicates = []
    for folder in os.listdir(main_path):
        hashes_and_their_photos = {}
        for file in os.listdir(main_path + '/' + folder):
            current_path = main_path + '/' + folder + '/' + file
            hashed_img = hash_image(current_path)
            if hashes_and_their_photos.get(hashed_img) is None:
                hashes_and_their_photos[hashed_img] = [current_path]
            else:
                duplicates.append(current_path)

    return duplicates


def check_file_extension(main_path):
    files_with_wrong_extension = []
    for folder in os.listdir(main_path):
        for file in os.listdir(main_path + '/' + folder):
            all_path = main_path + '/' + folder + '/' + file
            if not all_path.endswith((".png", ".jpg", ".jpeg")):
                files_with_wrong_extension.append(all_path)

    return files_with_wrong_extension


# print(len(find_duplicates()))
# print(1)
# print(check_file_extension())
