import pandas as pd

def convert_to_parquet(input_csv_path, output_parquet_path, checklondon=False, seperator=','):
    """
    Converts a deprivation index table from CSV to Parquet format.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_parquet_path (str): Path to save the output Parquet file.
    """
    # Load the deprivation index table from CSV
    df = pd.read_csv(input_csv_path, sep=seperator, )

    if checklondon:
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

def agetable_to_parquet(man_csv, female_csv, total_csv, output_parquet_path):
    """
    Converts age tables from CSV to Parquet format.
    """
    
    # Load the age tables from CSV
    df_man = pd.read_csv(man_csv)
    df_female = pd.read_csv(female_csv)
    df_total = pd.read_csv(total_csv)

    # Add a column for sex to each DataFrame
    df_man['Sex'] = 'Male'
    df_female['Sex'] = 'Female'
    df_total['Sex'] = 'Total'

    # Combine the DataFrames
    combined_df = pd.concat([df_man, df_female, df_total], ignore_index=True)
    
    combined_df['%.23'] = combined_df['%.23'].astype(str)
    
    # For columns that should be numeric but contain some non-numeric values
    numeric_cols = [col for col in combined_df.columns 
                   if col not in ['Sex'] and '.' not in col and '%' not in col and 'Area' not in col and 'mnemonic' not in col]
    
    for col in numeric_cols:
        try:
            # Convert column to numeric, coercing errors to NaN
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            # Fill NaN values with 0 or another appropriate value
            combined_df[col] = combined_df[col].fillna(0)
        except:
            print(f"Could not convert column {col} to numeric")

    # Save the combined DataFrame to Parquet format
    combined_df.to_parquet(output_parquet_path, index=False)
    

# Example usage
if __name__ == "__main__":
    #input_csv = "../data/IMD_2019.csv" 
    #output_parquet = "../processed_data/deprivation_index_2019.parquet"  
    #convert_to_parquet(input_csv, output_parquet, False, seperator=';')
#
    #input_csv = "../data/ethnicity.csv"
    #output_parquet = "../processed_data/ethnicity_index.parquet"
    #convert_to_parquet(input_csv, output_parquet, True)

    #input_csv_man = "../data/age_man.csv"
    #input_csv_female = "../data/age_female.csv"
    #input_csv_total = "../data/age_total.csv"
    #output_parquet = "../processed_data/age_index.parquet"
    #agetable_to_parquet(input_csv_man, input_csv_female, input_csv_total, output_parquet)

    #input_csv = "../data/deprivation_2015.csv"
    #output_parquet = "../processed_data/deprivation_2015.parquet"
    #convert_to_parquet(input_csv, output_parquet, False, seperator=';')
    pass