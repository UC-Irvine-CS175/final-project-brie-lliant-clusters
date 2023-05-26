"""
This module provides a set of utility functions to retrieve data from AWS S3 buckets.

Functions:
- get_bytesio_from_s3(s3_client:boto3.client, bucket_name:str,file_path:str) -> BytesIO:
    Retrieves individual files from a specific aws s3 bucket blob/file path as a
    BytesIO object to enable the user to not have to save the file to their local machine.

- get_file_from_s3(
    s3_client:boto3.client,
    bucket_name:str,
  s3_file_path:str, local_file_path:str) -> str:
    Retrieves and individual file from a specific aws s3 bucket blob/file path and saves
    the files of interest to a local filepath on the user's machine.

- save_tiffs_local_from_s3(
    s3_client:boto3.client,
    bucket_name:str,
    s3_path:str,
    local_fnames_meta_path:str,
    save_file_path:str,) -> None:
    Retrieves tiff file names from a locally stored csv file specific to the aws s3 bucket
    blob/path.

- export_subset_meta_dose_hr(
    dose_Gy_specifier: str,
    hr_post_exposure_val: int,
    in_csv_path_local: str) -> (str, int):
        Opens a csv file that contains the filepaths of the bps microscopy data from the s3 
        bucket saved either locally or as a file_buffer object as a pandas dataframe. The 
        dataframe is then sliced over the attributes of interest and written to another csv
        file for data versioning.
    
Notes:
- The functions in this module are designed to be used with the AWS open source registry for the
  bps microscopy data. The data is stored in a public s3 bucket and can be accessed without
  authentication. The data is stored in s3://nasa-bps-training-data/Microscopy

- Some functions require that the s3 client be configured for open UNSIGNED signature. This can be
  done prior to calling the functions by passing the following config to the boto3.client:
    
    config = Config(signature_version=UNSIGNED)
    s3_client = boto3.client('s3', config=config)

"""
import boto3
import csv
from botocore import UNSIGNED
from botocore.config import Config
import io
from io import BytesIO
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pyprojroot
import sys
sys.path.append(str(pyprojroot.here()))


def get_bytesio_from_s3(
    s3_client: boto3.client, bucket_name: str, file_path: str
) -> BytesIO:
    """
    This function retrieves individual files from a specific aws s3 bucket blob/file path as
    a BytesIO object to enable the user to not have to save the file to their local machine.

    args:
        s3_client (boto3.client): s3 client should be configured for open UNSIGNED signature.
        bucket_name (str): name of bucket from AWS open source registry.
        file_path (str): blob/file path name from aws including file name and extension.

    returns:
        BytesIO: BytesIO object from the file contents
    """
   # Download file contents from S3
    file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_path)['Body'].read()

    # Wrap file contents in a BytesIO object
    bytes_io = BytesIO(file_obj)

    return bytes_io


def get_file_from_s3(
    s3_client: boto3.client, bucket_name: str, s3_file_path: str, local_file_path: str
) -> str:
    """
    This function retrieves individual files from a specific aws s3 bucket blob/file path and
    saves the files of interest to a local filepath on the user's machine.

    args:
      s3_client (boto3.client): s3 client should be configured for open UNSIGNED signature.
      bucket_name (str): name of bucket from AWS open source registry.
      s3_file_path (str): full blob/file path name from aws including file name and extension.
      local_file_path (str): user's local directory.

    returns:
      str: local file path with naming convention of the file that was downloaded from s3 bucket
    """
    
    # Create the directory if it does not exist
    # Create path with local directory provided by the userfile and the name of the s3 file of interest
    # derived from the s3_file_path
    # Download file
    #raise NotImplementedError
    
     # Create the directory if it does not exist
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    
    # Create path with local directory provided by the user and the name of the s3 file of interest
    local_file_path = os.path.join(local_file_path, os.path.basename(s3_file_path))
    
    # Download file
    s3_client.download_file(bucket_name, s3_file_path, local_file_path)
    
    return local_file_path


def save_tiffs_local_from_s3(
    s3_client: boto3.client,
    bucket_name: str,
    s3_path: str,
    local_fnames_meta_path: str,
    save_file_path: str,
) -> None:
    """
    This function retrieves tiff files from a locally stored csv file containing specific aws s3 bucket
    filenames, constructs the appropriate paths to retrieve the files of interest locally to the user's 
    machine following the same naming convention as the files from s3.

    args:
      s3_client (boto3.client): s3 client should be configured for open UNSIGNED signature.
      bucket_name (str): name of bucket from AWS open source registry.
      s3_path (str): blob/file directory where files of interest reside in s3 from AWS
      local_fnames_meta_path (str): file path for user's local directory containing the csv file containing the filenames
      save_file_path (str): file path for user's local directory where files of interest will be saved
    returns:
      None
    """
   # Read the CSV file with the S3 filenames
    with open(local_fnames_meta_path, "r") as f:
        reader = csv.reader(f)
        filenames = [row[0] for row in reader if row[0] != "filename"]

    # Download the files from S3 and save them locally
    for filename in filenames:
        get_file_from_s3(s3_client, bucket_name, os.path.join(s3_path,filename), save_file_path)
 


def export_subset_meta_dose_hr(
    dose_Gy_specifier: str,
    hr_post_exposure_val: int,
    in_csv_path_local: str,             # path includes name of file w/ extension
    out_dir_csv: str
) -> tuple:
    """
    This function opens a csv file that contains the filenames of the bps microscopy data from the 
    s3 bucket saved either locally or as a file_buffer object as a pandas dataframe. The dataframe
    is then sliced over the attributes of interest and written to another csv file for data 
    versioning.

    args:
      dose_Gy (str): dose_Gy is a string corresponding to the dose of interest ['hi', 'med', 'low']
      hr_post_exposure_val (int): hr_post_exposure_val is an integer corresponding to the hour post 
      exposure of interest [4, 24, 48]
      in_csv_path_local (str): a string of input original csv file
      out_dir_csv (str): a string of the output directory you would like to write the subset_meta file to

    returns:
      Tuple[str, int]: a tuple of the output csv file path and the number of rows in the output csv 
      file
    """
    # Create output directory out_dir_csv if it does not exist
    if not (os.path.exists(out_dir_csv)):
        os.makedirs(out_dir_csv)
    # Load csv file into pandas DataFrame
    # if not os.path.exists(in_csv_path_local):
    #     return ([None, None])
    df = pd.read_csv(in_csv_path_local)

    dose_specifier_array = ['low', 'med', 'hi']
    # Check that dose_Gy and hr_post_exposure_val are valid
    if dose_Gy_specifier not in dose_specifier_array:
        return ([None, None])
    else:
        dose_Gy_specifier = dose_specifier_array.index(dose_Gy_specifier)
    if hr_post_exposure_val not in [4,24,48]:
        return ([None, None])
    #               low, med, hi
    Fe_dose_Gy = [0.0, 0.3, 0.82]
    Xray_dose_Gy = [0.0, 0.1, 1.0]


    # Slice DataFrame by attributes of interest
    df_Fe = df[(df['hr_post_exposure'] == hr_post_exposure_val) & (df['particle_type'] == "Fe") & (df['dose_Gy'] == Fe_dose_Gy[dose_Gy_specifier])]

    df_ray = df[(df['hr_post_exposure'] == hr_post_exposure_val) & (df['particle_type'] == "X-ray") & (df['dose_Gy'] == Xray_dose_Gy[dose_Gy_specifier])]

    
    combined_df = pd.concat([df_Fe,df_ray])

    # Write sliced DataFrame to output csv file with same name as input csv file with 
    # _dose_hr_post_exposure.csv appended
    name, ext = os.path.splitext(in_csv_path_local)
    first_file = name + "_dose_hr_post_exposure" + ext

    f = open(first_file, "w")
    df.to_csv(first_file, index=False)
    f.close()
    # Construct output csv file path using out_dir_csv and the name of the input csv file
    # with the dose_Gy and hr_post_exposure_val appended to the name of the input csv file
    # for data versioning. 

    # Write sliced DataFrame to output csv file with name constructed above
    name, ext = os.path.splitext(in_csv_path_local)
    f_name = name + "_dose_" + dose_specifier_array[dose_Gy_specifier] + "_hr_" + str(hr_post_exposure_val) + "_post_exposure"  + ext
    
    t = open(f_name, "w")

    combined_df.to_csv(t, index=False)
    t.close()
    return (f_name, len(combined_df))
    
def train_test_split_subset_meta_dose_hr(
        subset_meta_dose_hr_csv_path: str,
        test_size: float,
        out_dir_csv: str,
        random_state: int = None,
        stratify_col: str = None
        ) -> tuple:
    """
    This function reads in a csv file containing the filenames of the bps microscopy data for
    a subset selected by the dose_Gy and hr_post_exposure attributes. The function then opens
    the file as a pandas dataframe and splits the dataframe into train and test sets using
    sklearn.model_selection.train_test_split. The train and test dataframes are then exported
    to csv files in the same directory as the input csv file.

    args:
        subset_meta_dose_hr_csv_path (str): a string of the input csv file path (full path includes filename)
        test_size (float or int): a float between 0 and 1 corresponding to the proportion of the data
        that should be in the test set. If int, represents the absolute number of test samples.
        out_dir_csv (str): a string of the output directory you would like to write the train and test
        random_state (int, RandomState instance or None, optional): controls the shuffling
        applied to the data before applying the split. Pass an int for reproducible output
        across multiple function calls.
        stratify (array-like or None, optional): array containing the labels for stratification. 
        Default: None.
    returns:
        Tuple[str, str]: a tuple of the output csv file paths for the train and test sets
    """
     # Create output directory out_dir_csv if it does not exist
    os.makedirs(out_dir_csv, exist_ok=True)

    # Load csv file into pandas DataFrame and use the train_test_split function to split the
    # DataFrame into train and test sets
    df = pd.read_csv(subset_meta_dose_hr_csv_path)
    X = df.drop(['filename'], axis=1)
    y = df['filename']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=X[stratify_col] if stratify_col else None)

    # Rewrite index numbers for both train and test sets to conform to order in new dataframe
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Write train and test DataFrames to output csv files with same name as input csv file with
    # _train.csv or _test.csv appended
    train_csv_path = os.path.join(out_dir_csv, os.path.splitext(os.path.basename(subset_meta_dose_hr_csv_path))[0] + '_train.csv')
    test_csv_path = os.path.join(out_dir_csv, os.path.splitext(os.path.basename(subset_meta_dose_hr_csv_path))[0] + '_test.csv')
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    # return the train and test csv paths
    return train_csv_path, test_csv_path

def main():
    """
    A driver function for testing the functions in this module. Use if you like.
    """

    output_dir = '../data/processed'

    # s3 bucket info
    bucket_name = 'nasa-bps-training-data'
    s3_path = 'Microscopy/train'
    s3_meta_csv_path = f'{s3_path}/meta.csv'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # local file path info
    local_file_path = "../data/raw"

    local_new_path_fname = get_file_from_s3(
        s3_client=s3_client,
        bucket_name=bucket_name,
        s3_file_path=s3_meta_csv_path,
        local_file_path=local_file_path)

    subset_new_path_fname, subset_size = export_subset_meta_dose_hr(
        dose_Gy_specifier='hi',
        hr_post_exposure_val=4,
        in_csv_path_local=local_new_path_fname,
        out_dir_csv=output_dir)

    print(subset_size)

    subset_new_path_fname, subset_size = export_subset_meta_dose_hr(
        dose_Gy_specifier='med',
        hr_post_exposure_val=4,
        in_csv_path_local=local_new_path_fname,
        out_dir_csv=output_dir)
    
    print(subset_size)

    subset_new_path_fname, subset_size = export_subset_meta_dose_hr(
        dose_Gy_specifier='low',
        hr_post_exposure_val=4,
        in_csv_path_local=local_new_path_fname,
        out_dir_csv=output_dir)
    

    train_test_split_subset_meta_dose_hr(
        subset_meta_dose_hr_csv_path=subset_new_path_fname,
        test_size=0.2,
        out_dir_csv=output_dir,
        random_state=42,
        stratify_col="particle_type")

    
    ## save tiffs locally from s3 using boto3
    # save_tiffs_local_from_s3(
    # s3_client=s3_client,
    # bucket_name=bucket_name,
    # s3_path=s3_path,
    # local_fnames_meta_path=subset_new_path_fname,
    # save_file_path=local_file_path)


if __name__ == "__main__":
    main()