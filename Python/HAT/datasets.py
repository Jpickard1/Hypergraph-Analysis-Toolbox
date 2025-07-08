"""
HIF Dataset Downloader Module

This module provides functionality to download Hypergraph Interchange Format (HIF) 
datasets from the HIF-datasets GitHub repository. It handles both single-file and 
multi-file datasets, validates the downloaded files against the HIF schema, and 
provides a clean interface for accessing hypergraph datasets.

Example:
    >>> download("BioCarta_2013", "./datasets")
    >>> # Downloads BioCarta_2013.hif to ./datasets/ directory
"""

import requests
import os
import re
from pathlib import Path
import fastjsonschema
import json
from HAT import Hypergraph

def load(dataset_name, datapath=None):
    """
    Load a HIF dataset as a Hypergraph object.
    
    This function first checks if the dataset exists locally in the specified
    data directory. If found, it loads the local file directly. If not found,
    it downloads the dataset from the GitHub repository and then loads it.
    
    Args:
        dataset_name (str): Name of the dataset to load (without .hif extension).
                           Examples: "BioCarta_2013", "KEGG_2018"
        datapath (str, optional): Directory path where datasets are stored.
                                 Defaults to "data" if not specified.
    
    Returns:
        Hypergraph: A Hypergraph object loaded from the HIF file
    
    Raises:
        FileNotFoundError: If dataset cannot be found locally or downloaded
        ValueError: If dataset_name is invalid or empty
        Exception: If HIF file cannot be parsed or loaded
    
    Example:
        >>> # Load from local cache if available, otherwise download
        >>> H = load("BioCarta_2013")
        >>> print(f"Loaded hypergraph with {len(H.nodes)} nodes")
        
        >>> # Load to custom directory
        >>> H = load("KEGG_2018", datapath="./my_datasets")
        >>> print(f"Hypergraph loaded from ./my_datasets/")
    
    Note:
        - The function automatically handles both single-file and multi-file datasets
        - Downloaded files are cached locally for future use
        - The function validates the HIF file against the official schema
    """
    # Input validation
    if not dataset_name or not isinstance(dataset_name, str):
        raise ValueError("dataset_name must be a non-empty string")
    
    # Set default data path if not provided
    if datapath is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        datapath = os.path.join(current_file_dir, "data")
    
    # Ensure data directory exists
    os.makedirs(datapath, exist_ok=True)
    
    # Construct the expected file path
    hif_file_path = os.path.join(datapath, f"{dataset_name}.hif")
    
    # Check if dataset exists locally
    if os.path.exists(hif_file_path):
        print(f"Loading {dataset_name} from local cache: {hif_file_path}")
        try:
            with open(hif_file_path, 'r') as f:
                hif_data = json.load(f)
            H = Hypergraph.from_hif(hif_data)
            return H
        except Exception as e:
            print(f"Error loading local file {hif_file_path}: {e}")
            print("Attempting to re-download dataset...")
            # If local file is corrupted, remove it and re-download
            os.remove(hif_file_path)
    
    # Dataset not found locally or local file was corrupted - download it
    print(f"Dataset {dataset_name} not found locally. Downloading...")
    try:
        download(dataset_name, datapath)
        
        # Verify the downloaded file exists
        if not os.path.exists(hif_file_path):
            raise FileNotFoundError(f"Download completed but file not found: {hif_file_path}")
        
        # Load the downloaded dataset
        with open(hif_file_path, 'r') as f:
            hif_data = json.load(f)
        H = Hypergraph.from_hif(hif_data)
        print(f"Successfully loaded {dataset_name} from downloaded file")
        return H
        
    except Exception as e:
        raise Exception(f"Failed to download or load dataset '{dataset_name}': {e}")

def download(dataset_name, download_path):
    """
    Download a HIF dataset from the GitHub repository.
    
    This function handles the complete download process including:
    - Checking if the dataset exists
    - Downloading single or multiple files
    - Joining multi-file datasets
    - Validating the final file against HIF schema
    
    Args:
        dataset_name (str): Name of the dataset to download (without .hif extension)
        download_path (str): Local directory path where the dataset will be saved
    
    Returns:
        datasetfile (str): Path to the dataset as a .hif file
    
    Raises:
        Exception: If dataset is not found or download fails
    
    Example:
        >>> download("BioCarta_2013", "./my_datasets")
        File BioCarta_2013.hif downloaded successfully
        HIF-Compliant JSON.
    """
    # Step 1: get list of all HIF dataset names
    hif_dataset_files = get_hif_datasets()
    
    # Step 2: Check if dataset exists and determine file pattern
    dataset_files = find_dataset_files(hif_dataset_files, dataset_name)
    
    # Step 3: Download each file separately
    for file_name in dataset_files:
        download_hif_file(file_name, download_path=download_path)
    
    # Step 4: Join the files together
    if len(dataset_files) > 1:
        rejoin_files(dataset_name, directory=download_path)
    dataset_file = os.path.join(download_path, dataset_name + ".hif")
    
    # Step 5: Validate the final file to the HIF schma
    validate_hif_schema(dataset_file)
    return dataset_file

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def get_hif_datasets():
    """
    Retrieve list of all HIF dataset files from the GitHub repository.
    
    Connects to the GitHub API to fetch the contents of the datasets directory
    and returns a list of all available HIF files.
    
    Returns:
        list: List of filenames (str) available in the repository
    
    Raises:
        requests.RequestException: If API request fails
    
    Example:
        >>> files = get_hif_datasets()
        >>> print(f"Found {len(files)} datasets")
        >>> print(files[:3])  # Show first 3 files
    """
    api_url = "https://api.github.com/repos/Jpickard1/HIF-datasets/contents/datasets"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        files = response.json()
        file_list = [f['name'] for f in files if f['type'] == 'file']
    else:
        file_list = []
    return file_list

def find_dataset_files(hif_dataset_files, dataset_name):
    """
    Find all files belonging to a specific dataset.
    
    Searches for dataset files in two patterns:
    1. Single file: dataset_name.hif
    2. Multi-file: dataset_name_1of5.hif, dataset_name_2of5.hif, etc.
    
    Args:
        hif_dataset_files (list): List of all available HIF files
        dataset_name (str): Name of the dataset to search for
    
    Returns:
        list: List of filenames that belong to the dataset, or None if not found
    
    Example:
        >>> files = get_hif_datasets()
        >>> dataset_files = find_dataset_files(files, "BioCarta_2013")
        >>> print(dataset_files)
        ['BioCarta_2013.hif']
    """
    # Pattern 1: Single file (dataset_name.hif)
    single_file_pattern = f"{dataset_name}.hif"
    
    # Pattern 2: Multi-file (dataset_name_1of5.hif, dataset_name_2of5.hif, etc.)
    multi_file_pattern = re.compile(rf"{re.escape(dataset_name)}_(\d+)of(\d+)\.hif")
    
    # Check for single file
    single_file_found = None
    for file in hif_dataset_files:
        if file == single_file_pattern:
            return [file]
    
    # Check for multi-file pattern
    multi_files = []
    for file in hif_dataset_files:
        match = multi_file_pattern.match(file)
        if match:
            multi_files.append(file)

    if len(multi_files) > 0:
        return multi_files
    else:
        print(f"No HIF dataset found matching {dataset_name}. Please check the dataset name and spelling.")
        return None

def download_hif_file(file, download_path):
    """
    Download a single HIF file from the GitHub repository.
    
    Downloads a file from the raw GitHub URL and saves it to the specified path.
    
    Args:
        file (str): Name of the file to download
        download_path (str): Local directory where the file will be saved
    
    Returns:
        None
    
    Raises:
        requests.RequestException: If download fails
        IOError: If file cannot be written to disk
    
    Example:
        >>> download_hif_file("BioCarta_2013.hif", "./datasets")
        File BioCarta_2013.hif downloaded successfully
    """
    url = "https://raw.githubusercontent.com/Jpickard1/HIF-datasets/main/datasets/" + file
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(os.path.join(download_path, file), "wb") as file:
            file.write(response.content)
        print(f"File {file} downloaded successfully")
    else:
        print(f"Failed to download file: {response.status_code}")

def rejoin_files(base_name, directory="."):
    """
    Rejoin split HIF files back into a single file and clean up chunks.
    
    Takes multiple files with pattern base_name_1of3.hif, base_name_2of3.hif, etc.
    and joins them into a single base_name.hif file. After successful joining,
    the individual chunk files are deleted.
    
    Args:
        base_name (str): Base name of the dataset (without .hif extension)
        directory (str, optional): Directory containing the chunk files. Defaults to ".".
    
    Returns:
        None
    
    Example:
        >>> rejoin_files("LargeDataset", "./datasets")
        Rejoining 3 chunks for LargeDataset
          Adding: LargeDataset_1of3.hif
          Adding: LargeDataset_2of3.hif
          Adding: LargeDataset_3of3.hif
          Rejoined file created: ./datasets/LargeDataset.hif
          Final size: 15.23MB
          Cleanup complete - removed 3 chunk files
    """
    directory = Path(directory)
    
    # Find all chunks for this base name
    pattern = f"{base_name}_*of*.hif"
    chunk_files = list(directory.glob(pattern))
    
    if not chunk_files:
        print(f"No chunk files found for {base_name}")
        return
    
    # Sort chunks by number
    def get_chunk_number(filename):
        # Extract number from filename like "name_1of3.hif"
        stem = filename.stem
        if "_" in stem and "of" in stem:
            try:
                number_part = stem.split("_")[-1]  # Get "1of3"
                return int(number_part.split("of")[0])  # Get "1"
            except:
                return 0
        return 0
    
    chunk_files.sort(key=get_chunk_number)
    
    print(f"Rejoining {len(chunk_files)} chunks for {base_name}")
    
    # Rejoin files
    output_file = directory / f"{base_name}.hif"
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for chunk_file in chunk_files:
            print(f"  Adding: {chunk_file.name}")
            with open(chunk_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
    
    print(f"  Rejoined file created: {output_file}")
    final_size = get_file_size_mb(output_file)
    print(f"  Final size: {final_size:.2f}MB")

    # Delete the individual chunk files
    print("  Cleaning up chunk files:")
    for chunk_file in chunk_files:
        try:
            chunk_file.unlink()
            print(f"    Deleted: {chunk_file.name}")
        except Exception as e:
            print(f"    Error deleting {chunk_file.name}: {e}")
    
    print(f"  Cleanup complete - removed {len(chunk_files)} chunk files")

def validate_hif_schema(file_path):
    """
    Validate a HIF file against the official HIF JSON schema.
    
    Downloads the official HIF schema from the HIF-standard repository and
    validates the provided file against it using fastjsonschema.
    
    Args:
        file_path (str): Path to the HIF file to validate
    
    Returns:
        None
    
    Raises:
        requests.RequestException: If schema cannot be downloaded
        json.JSONDecodeError: If file is not valid JSON
        fastjsonschema.JsonSchemaException: If file doesn't conform to HIF schema
    
    Example:
        >>> validate_hif_schema("./datasets/BioCarta_2013.hif")
        HIF-Compliant JSON.
        
        >>> validate_hif_schema("./datasets/invalid.hif")
        Invalid JSON: data must be object
    """
    url = "https://raw.githubusercontent.com/pszufe/HIF-standard/main/schemas/hif_schema.json"
    schema = requests.get(url).json()
    validator = fastjsonschema.compile(schema)
    hiftext = json.load(open(file_path,'r'))
    try:
        validator(hiftext)
        print("HIF-Compliant JSON.")
        return True
    except Exception as e:
        print(f"Invalid JSON: {e}")
        return False