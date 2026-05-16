import os

def organize_test_folder(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)

        if not os.path.isfile(full_path):
            continue

        label = file.split('.')[0]
        class_dir = os.path.join(path, label)

        os.makedirs(class_dir, exist_ok=True)
        os.rename(
            full_path,
            os.path.join(class_dir, file)
        )
