"""Cleaning Functions

These functions define standard text processing functions for cleaning.

"""

from html import unescape
import re
import emoji

def clean_text(text):
    """Cleans single data entry of text.

    Args:
        text (str): input text for cleaning.

    Returns:
        str: output cleaned text.
    """
    # convert HTML codes
    text = unescape(text)
    # replace mentions, URLs and emojis with special token
    text = re.sub(r"@[A-Za-z0-9_-]+",'[USER]',text)
    text = re.sub(r"http\S+",'[URL]',text)
    text = ''.join(' [EMOJI] ' if (emoji.is_emoji(char) is True) else char for char in text).strip()
    # remove newline and tab characters
    text = text.replace('\n',' ')
    text = text.replace('\t',' ')
    text = text.replace('<br /><br />',' ')
    text = text.replace('<br />',' ')
    # remove leading ">" (reddit artifact)
    text = text.lstrip('>')
    # collapse whitespace into single whitespace
    text = re.sub(r'\s+', ' ', text)
    # remove leading and trailing whitespaces
    text = text.strip()
    return text


def drop_nans(input_df, subset_col='text', verbose = False):
    """Removes posts with NaN values in given column.

    Args:
        input_df (pd.DataFrame): input dataframe.
        subset_col (str, optional): column for NaN removal. Defaults to 'text'.
        verbose (bool, optional): whether to print number of dropped values. Defaults to False.

    Returns:
        pd.DataFrame: output dataframe with modifications.
    """
    # Get original len
    orig_len = len(input_df)
    # remove NANs in place
    input_df.dropna(subset=[subset_col], inplace = True)
    # Get new len
    new_len = len(input_df)
    if verbose is True:
        print(f"""\nOrig len: {orig_len},
            Num of dropped values: {orig_len - new_len},
            New len: {new_len}""")
    return input_df


def drop_duplicates(input_df, subset_col = 'clean_text', verbose = False):
    """Removes duplicate values in given column. Should be run *after* text cleaning.

    Args:
        input_df (pd.DataFrame): input dataframe.
        subset_col (str, optional): column for de-duplication. Defaults to 'clean_text'.
        verbose (bool, optional): whether to print number of dropped values. Defaults to False.

    Returns:
        pd.DataFrame: output dataframe with modifications.
    """
    # Get original len
    orig_len = len(input_df)
    # remove duplicates in place
    input_df.drop_duplicates(subset=[subset_col], inplace = True)
    # Get new len
    new_len = len(input_df)
    if verbose is True:
        print(f"""\nOrig len: {orig_len},
            Num of dropped values: {orig_len - new_len},
            New len: {new_len}""")
    return input_df

def drop_empty_text(input_df, subset_col = 'clean_text', verbose = False):
    """Removes rows with empty text. Should be run *after* text cleaning.

    Args:
        input_df (pd.DataFrame): input dataframe.
        subset_col (str, optional): column for empty text removal. Defaults to 'clean_text'.
        verbose (bool, optional): whether to print number of dropped values. Defaults to False.

    Returns:
        pd.DataFrame: output dataframe with modifications.
    """
    # Get original len
    orig_len = len(input_df)
    # drop rows with empty text
    input_df = input_df[input_df[subset_col].values!=""]
    # Get new len
    new_len = len(input_df)
    if verbose is True:
        print(f"""\nOrig len: {orig_len},
            Num of dropped values: {orig_len - new_len},
            New len: {new_len}""")
    return input_df

def drop_url_emoji(input_df, subset_col = 'clean_text', verbose = False):
    """Removes rows with only [URL] or [EMOJI] tokens. Should be run *after* text cleaning.

    Args:
        input_df (pd.DataFrame): input dataframe.
        subset_col (str, optional): column for text removal. Defaults to 'clean_text'.
        verbose (bool, optional): whether to print number of dropped values. Defaults to False.

    Returns:
        pd.DataFrame: output dataframe with modifications.
    """
    # Get original len
    orig_len = len(input_df)
    # drop rows with text that is just [URL] or [EMOJI]
    input_df = input_df[(input_df[subset_col]!="[URL]") & (input_df[subset_col]!="[EMOJI]")]
    # Get new len
    new_len = len(input_df)
    if verbose is True:
        print(f"""\nOrig len: {orig_len},
            Num of dropped values: {orig_len - new_len},
            New len: {new_len}""")
    return input_df
