import pandas as pd

def convert_to_parquet(input_csv_path, output_parquet_path):
    """
    Converts a deprivation index table from CSV to Parquet format.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_parquet_path (str): Path to save the output Parquet file.
    """
    # Load the deprivation index table from CSV
    df = pd.read_csv(input_csv_path)

    # Save the DataFrame to Parquet format
    df.to_parquet(output_parquet_path, index=False)

# Example usage
if __name__ == "__main__":
    input_csv = "../data/deprivation.csv" 
    output_parquet = "../processed_data/deprivation_index.parquet"  
    convert_to_parquet(input_csv, output_parquet)