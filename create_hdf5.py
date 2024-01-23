import h5py
import torch
import os

def create_hdf5_dataset(source_folder, output_file):
    # Check if the output directory exists, if not, create it
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Modify the output file name to add 'E' at the end for efficiency
    base_name, ext = os.path.splitext(output_file)
    efficient_output_file = base_name + 'E' + ext

    with h5py.File(efficient_output_file, 'w') as hdf5_file:
        # Iterate over files in the source folder and add them to the HDF5 file
        for file_name in os.listdir(source_folder):
            file_path = os.path.join(source_folder, file_name)
            if file_path.endswith('.pt'):
                data = torch.load(file_path)
                image = data['image'].numpy()
                label = data['label'].numpy()

                hdf5_file.create_dataset(file_name, data=image)
                hdf5_file.create_dataset(file_name + '_label', data=label)

# Example usage for train, val, and test datasets
create_hdf5_dataset('E:\\facial_recognition_datasets\\train_set_augmented', 'E:\\facial_recognition_datasets\\train_datasetE.hdf5')
create_hdf5_dataset('E:\\facial_recognition_datasets\\val_set_augmented', 'E:\\facial_recognition_datasets\\val_datasetE.hdf5')
create_hdf5_dataset('E:\\facial_recognition_datasets\\test_set_augmented', 'E:\\facial_recognition_datasets\\test_datasetE.hdf5')
