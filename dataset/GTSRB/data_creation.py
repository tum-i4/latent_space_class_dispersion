import zipfile
import os
import shutil
import csv

import torchvision.transforms as transforms


def initialize_data(folder):
    # train_zip = folder + '/train_images.zip'
    # test_zip = folder + '/test_images.zip'
    # if not os.path.exists(train_zip) or not os.path.exists(test_zip):
    #     raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
    #           + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2017/data '))

    # extract train_data.zip to train_data
    # train_folder = folder + '/train_images'
    # if not os.path.isdir(train_folder):
    #     print(train_folder + ' not found, extracting ' + train_zip)
    #     zip_ref = zipfile.ZipFile(train_zip, 'r')
    #     zip_ref.extractall(folder)
    #     zip_ref.close()

    # # extract test_data.zip to test_data
    # test_folder = folder + '/test_images'
    # if not os.path.isdir(test_folder):
    #     print(test_folder + ' not found, extracting ' + test_zip)
    #     zip_ref = zipfile.ZipFile(test_zip, 'r')
    #     zip_ref.extractall(folder)
    #     zip_ref.close()
        
    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val'

    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(os.path.join(dataset_dir, 'Train')):
            os.mkdir(val_folder + '/' + dirs)
            for f in os.listdir(os.path.join(dataset_dir, 'Train') + '/' + dirs):
                if f.split('_')[1].startswith('00000') or f.split('_')[1].startswith('00001') or f.split('_')[1].startswith('00002'):
                    # move file to validation folder
                    os.rename(os.path.join(dataset_dir, 'Train') + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)

def write_csv_files(train_csv, val_csv):
    with open(train_csv, 'r', newline='') as ip_file:
        reader = csv.reader(ip_file)
        org_data = list(reader)
        # train_data = list(reader)
        # val_data = list(reader)
    
    index_to_remove = []
    for i in range(1, len(org_data)):
        #png_name = row['Path'].split('/')[-1]
        row = org_data[i]
        png_name = row[-1].split('/')[-1]

        if png_name.split('_')[1].startswith('00000') or png_name.split('_')[1].startswith('00001') or png_name.split('_')[1].startswith('00002'):
            index_to_remove.append(i)   

    if index_to_remove is not None:
        # del org_data[index_to_remove]

        val_data_filtered = [org_data[i] for i in index_to_remove]
        train_data_filtered = [org_data[i] for i in range(len(org_data)) if i not in index_to_remove]

        print('Total data: {} \n Train data: {} \n Validation data: {}'.format(len(org_data), len(train_data_filtered), len(val_data_filtered)))
        # Write the modified data back to the CSV file
        with open(train_csv, 'w', newline='') as op_file:
            print('Writing new train splits to:', train_csv)
            writer = csv.writer(op_file)
            writer.writerows(train_data_filtered)

        with open(val_csv, 'w', newline='') as op_file:
            print('Writing new val splits to:', val_csv)
            writer = csv.writer(op_file)
            writer.writerows(val_data_filtered)
    else:
        print("Entry not found in the CSV file.")


if __name__ == '__main__':
    dataset_dir = "/data/disk2/gtsrb_dataset/"

    train_csv = os.path.join(dataset_dir, 'Train.csv')
    train_csv_copied = os.path.join(dataset_dir, 'train_org.csv')
    shutil.copy(train_csv, train_csv_copied)

    val_csv = os.path.join(dataset_dir, 'val.csv')

    initialize_data(dataset_dir)
    write_csv_files(train_csv, val_csv)
