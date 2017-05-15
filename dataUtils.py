import os
import numpy as np

file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    return np.asarray(classes)

def create_map_file_from_folder(root_folder, class_mapping, include_unknown=False):
    map_file_name = os.path.join(root_folder, "map.txt")
    with open(map_file_name , 'w') as map_file:
        for class_id in range(0, len(class_mapping)):
            folder = os.path.join(root_folder, class_mapping[class_id])
            if os.path.exists(folder):
                for entry in os.listdir(folder):
                    filename = os.path.join(folder, entry)
                    if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                        map_file.write("{0}\t{1}\n".format(filename, class_id))

        if include_unknown:
            for entry in os.listdir(root_folder):
                filename = os.path.join(root_folder, entry)
                if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                    map_file.write("{0}\t-1\n".format(filename))

    return map_file_name

train_data = { }
training_folder = "data//training"
train_data['class_mapping'] = create_class_mapping_from_folder(training_folder)
train_data['training_map'] = create_map_file_from_folder(training_folder, train_data['class_mapping'])

print("done!")