
import sys
import uproot
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import awkward as ak

class RootFileProcessor:
    def __init__(self, file_path, list_entries: dict = {}):
        self.file_path = file_path
        self.dfs = {}
        self.list_entries = list_entries 
        self.load_file()

    def load_file(self):
        with uproot.open(self.file_path) as file:
            # Access the ticlDumper directory
            ticlDumper = file['ticlDumper']
            # Iterate over all TTrees in ticlDumper
            for key in ticlDumper.keys():
                tree_name = key.split(';')[0]  # Extract the tree name (removing the ';1' part)
                tree = ticlDumper[tree_name]
                # Convert TTree to DataFrame with specified entries
                if(tree_name in self.list_entries.keys()):
                  df = tree.arrays(self.list_entries[tree_name], library='pd')
                else:
                  df = tree.arrays(library='pd')
                # Convert columns containing Awkward arrays to lists or NumPy arrays
                for col in df.columns:
                    if isinstance(df[col][0], ak.Array):
                        df[col] = df[col].tolist()  # Convert Awkward array to list
                        # Alternatively, you can convert to NumPy array
                        # df[col] = ak.to_numpy(df[col])  # Convert Awkward array to NumPy array
                # Store the DataFrame in the dictionary
                self.dfs[tree_name] = df

    def get_dataframe(self, tree_name):
        return self.dfs.get(tree_name, None)

    def get_all_dataframes(self):
        return self.dfs

class MultipleRootFileProcessor:
    def __init__(self, file_paths, list_entries: dict = {}, number_of_threads=None):
        self.file_paths = file_paths
        self.dfs = {}
        self.list_entries = list_entries
        self.number_of_threads = number_of_threads
        self.process_files()

    def process_single_file(self, file_path):
        processor = RootFileProcessor(file_path, self.list_entries)
        return processor.get_all_dataframes()

    def process_files(self):
        with ThreadPoolExecutor(max_workers=self.number_of_threads) as executor:
            results = list(tqdm(executor.map(self.process_single_file, self.file_paths), total=len(self.file_paths), desc="Processing ROOT files"))

        temp_dfs = {} 
        for file_dfs in tqdm(results, desc = "Finished processing files, merging results"):
            for tree_name, df in file_dfs.items():
              if tree_name not in temp_dfs:
                  temp_dfs[tree_name] = [df]
              else:
                  temp_dfs[tree_name].append(df)
        # Concatenate all DataFrames for each tree_name once at the end
        for tree_name, df_list in temp_dfs.items():
            self.dfs[tree_name] = pd.concat(df_list, ignore_index=True)

    def get_dataframe(self, tree_name):
        return self.dfs.get(tree_name, None)

    def get_all_dataframes(self):
        return self.dfs

    def save_dataframe(self, tree_name, file_path, format='csv'):
        """
        Save a specific DataFrame to a file.

        Parameters:
            tree_name (str): The name of the DataFrame to save.
            file_path (str): The path to save the file.
            format (str, optional): The format to save the file in. Defaults to 'csv'.
                Supported formats: 'csv', 'hdf5'
        """
        df = self.get_dataframe(tree_name)
        if df is None:
            print(f"No DataFrame found with name '{tree_name}'")
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if format == 'csv':
            df.to_csv(file_path, index=False)
            print(f"DataFrame '{tree_name}' saved to '{file_path}' as CSV.")
        elif format == 'hdf5':
            df.to_hdf(file_path, key='data', mode='w')
            print(f"DataFrame '{tree_name}' saved to '{file_path}' as HDF5.")
        elif format == 'arrow':
            df.to_feather(file_path)
            print(f"DataFrame '{tree_name}' saved to '{file_path}' as Arrow.")
        else:
            print(f"Unsupported format '{format}'. Supported formats: 'csv', 'hdf5', 'arrow'")

    def save_all_dataframes(self, directory_path, format='csv'):
        """
        Save all DataFrames to files in the specified format.

        Parameters:
            directory_path (str): The directory path where the files will be saved.
            format (str, optional): The format to save the files in. Defaults to 'csv'.
                Supported formats: 'csv', 'hdf5'
        """
        if format not in ['csv', 'hdf5', 'arrow']:
            print(f"Unsupported format '{format}'. Supported formats: 'csv', 'hdf5'")
            return

        for tree_name, _ in self.dfs.items():
            file_path = f"{directory_path}/{tree_name}.{format}"
            self.save_dataframe(tree_name, file_path, format)
