import os
import numpy as np
from PIL import Image

file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    return np.asarray(classes)

# create a map file from a flat folder, ie don't traverse subdirectories and don't add it to the path
def create_map_file_from_flatfolder(folder):
    map_file_name = os.path.join(folder, "map.txt")
    with open(map_file_name , 'w') as map_file:
        for entry in os.listdir(folder):
            filename = os.path.join(folder, entry)
            if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                tempName = '\\\\'.join(filename.split('\\'))
                tempName = '\\\\'.join(tempName.split('//'))
                tempName = '\\\\'.join(tempName.split('/'))
                map_file.write("{0}\t0\n".format(tempName))


    return map_file_name


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

def nparray_file_from_folder(root_folder, class_mapping, include_unknown=False):
    map_file_name = os.path.join(root_folder, "npArray_map.txt")
    labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
    indx = 0
    with open(map_file_name , 'w') as map_file:
        for class_id in range(0, len(class_mapping)):
            folder = os.path.join(root_folder, class_mapping[class_id])
            if os.path.exists(folder):
                for entry in os.listdir(folder):
                    filename = os.path.join(folder, entry)
                    if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                        #map_file.write("{0}\t{1}\n".format(filename, class_id))
                        img = Image.open(filename)
                        arr = np.array(img)
                        flat_arr = arr.ravel()
                        row_str = np.array_str(flat_arr)
                        feature_str = row_str[1:-1]
                        feature_str = feature_str.replace("\n", "")
                        label_str = labels[class_id]
                        res = '|labels {} |features {}\n'.format(label_str, feature_str)
                        print(res)
                        map_file.writelines(res)
                        indx = indx + 1
                        print("Processed file #{0} path {1}".format(indx, filename))

    return map_file_name

train_data = { }
training_folder1 = "data//summer2winter_yosemite//trainA"
training_folder2 = "data//summer2winter_yosemite//trainB"
train_data['training_map'] = create_map_file_from_flatfolder(training_folder1)
train_data['training_map'] = create_map_file_from_flatfolder(training_folder2)
#train_data['training_map'] = create_map_file_from_flatfolder(training_folder2)
#train_data['class_mapping'] = create_class_mapping_from_folder(training_folder1)
#train_data['training_map'] = create_map_file_from_folder(training_folder, train_data['class_mapping'])
#train_data['npArray_map'] = nparray_file_from_folder(training_folder, train_data['class_mapping'])

print("done!")