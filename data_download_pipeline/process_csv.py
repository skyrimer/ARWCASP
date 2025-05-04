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

    # Only keep data from london boroughs
    london_boroughs = [
        "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
        "Camden", "Croydon", "Ealing", "Enfield", "Greenwich",
        "Hackney", "Hammersmith and Fulham", "Haringey", "Harrow",
        "Havering", "Hillingdon", "Hounslow", "Islington",
        "Kensington and Chelsea", "Kingston upon Thames",
        "Lambeth", "Lewisham", "Merton", "Newham",
        "Redbridge", "Richmond upon Thames", "Southwark",
        "Sutton", "Tower Hamlets", "Waltham Forest",
        "Wandsworth", "Westminster"
    ]

    df = df[df['Lower tier local authorities'].isin(london_boroughs)]

    # check if there is at least one row for each borough
    for borough in london_boroughs:
        if borough not in df['Lower tier local authorities'].values:
            print(f"Warning: No data for {borough}")
    

    # Save the DataFrame to Parquet format
    df.to_parquet(output_parquet_path, index=False)

# Example usage
if __name__ == "__main__":
    input_csv = "../data/deprivation.csv" 
    output_parquet = "../processed_data/deprivation_index.parquet"  
    convert_to_parquet(input_csv, output_parquet)

    input_csv = "../data/ethnicity.csv"
    output_parquet = "../processed_data/ethnicity_index.parquet"
    convert_to_parquet(input_csv, output_parquet)