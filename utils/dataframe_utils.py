import pandas as pd


def convert_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the columns of a pandas DataFrame to numeric types wherever possible.

    This function attempts to convert each column in the provided DataFrame to a numeric type.
    If a column cannot be converted (e.g., it contains non-numeric data), it is left unchanged.

    Args:
        df (pd.DataFrame): The input DataFrame whose columns will be evaluated and converted if possible.

    Returns:
        pd.DataFrame: The updated DataFrame with numeric conversions applied where possible.
    """
    # Iterate over each column in the DataFrame
    for col in df:
        try:
            # Attempt to convert the column to a numeric type
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            # If conversion fails (due to non-numeric values), leave the column unchanged
            pass
    return df


def df_to_dict(df, key_col, value_col):
    """
    Convert two columns of a DataFrame into a dictionary.

    Args:
        df (pd.DataFrame): Input DataFrame.
        key_col (str): Column name to use as keys.
        value_col (str): Column name to use as values.

    Returns:
        dict: Dictionary mapping key_col → value_col.
    """
    return df.set_index(key_col)[value_col].to_dict()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the column names of a DataFrame.

    This function converts all column names to uppercase and replaces
    spaces with underscores, making them consistent and suitable for
    downstream processing or database storage.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame whose column names will be normalized.

    Returns
    -------
    pd.DataFrame
        A DataFrame with normalized column names.
    """
    df.columns = [str(c).upper().replace(" ", "_") for c in df.columns]
    return df
