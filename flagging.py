"""
This file contains all necessary functions to create a DataFrame ready to use for
non-arms length transaction detection using statistical and heurstic methods.

"""

import pandas as pd
import numpy as np
import re

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA



def go(columns: list, permut: tuple, groups: tuple, labels, output_file: str):


    drop_cols = ['is_homestead_exemption', 'homestead_exemption_general_alternative',
                 'homestead_exemption_senior_citizens', 'homestead_exemption_senior_citizens_assessment_freeze',
                 'card', 'sqft', 'year_built', 'is installment contract_fufilled', 'is_multisale',
                  'num_parcels_sale', 'is_condemndation', ] 

    df = read_data('sale_sample_18-21.parquet', 'cards.csv', 'char_sample.csv')

    df = create_stats(df, groups)

    df = string_processing(df)

    df = create_labels(df, columns, labels, permut)

    df = iso_forest()

    df = drop_irrelevant(df, drop_cols)

    df.to_csv(output_file)


def drop_irrelevant(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops columns that aren't relevant to analsts in determining whether
    a transaction may be non-arms length.
    Inputs:
        df (pd.DataFrame): dataframe to have columsn dropped from
    Outputs:
        df(pd.DataFrame)
    """
    df.drop(columns, axis=1, inplace=True)

    return df


def read_data(sales_name: str, card_name: str, char_name: str):
    sales = pd.read_parquet(sales_name)
    cards = pd.read_csv(card_name)
    char_sample = pd.read_csv(char_name, dtype={'class': str}) # also has 'EX' in column

    cards.drop_duplicates(inplace=True)
    # some are duplicates other than sqft and year built, same year and other info but diff sqft, so we drop it cuz this look anamolous
    char_sample.drop_duplicates(subset=['year', 'pin','class'],inplace=True)

    sales['year'] = sales.year.astype(int)
    sales['pin'] = sales.pin.astype(int)
    sales['class'] = sales['class'].astype(str)
    sales['township_code'] = sales.township_code.astype(int)

    sales = pd.merge(sales, char_sample)
    sales = pd.merge(sales, cards)

    sales = sales[sales['is_multisale'] != '1']
    sales = sales[sales['card'] == 1]
    sales.rename(columns = {'township_code_x': 'township_code'}, inplace=True)

    df = sales
    return df


def create_stats(df: pd.DataFrame, groups: tuple   ):

    df = price_sqft(df)
    df = grouping_mean(df, groups)
    df = dup_stats(df, groups)
    df = transaction_days(df)

    return df


def create_labels(df: pd.DataFrame,
                  columns: list,
                  labels: dict,
                  permut: tuple,
                  group: str = 'township_code'):
    """
    Brings all functions together to create microdata and label columns.
    To be run after create_stats().
    Inputs:
        df (pd.DataFrame): dataframe to make microdata on
        columns (list): list of string, columns in df to make outliers from
        group(str): geographic grouping like 'township_code'.
        labels (dict): mapping of column names to outlier labels for primary_outlier().
        permut (tuple): std deviations on each, used to make outliers.
    Outputs:
        df (pd.DataFrame): dataframe with microdata columns and outlier labels.
    """

    df = z_normalize(df, columns)

    columns = [col + '_zscore' for col in columns]

    holds = get_thresh(df, columns, permut)

    outs = over_std(df, group, columns, permut)
    sale_outs = outs.sale_key.to_list()

    df['primary_outlier'] = df.apply(primary_outlier, args=(holds, sale_outs, columns, labels), axis=1)

    df['outlier_value'] = df.apply(outlier_value, args=(labels,), axis=1)
    df['outlier_value_std'] = df.apply(outlier_value_std, args=(holds, labels), axis=1)
    df['outlier_std_lower'] = df.apply(outlier_std_lower, args=(holds, labels), axis=1)
    df['outlier_std_upper'] = df.apply(outlier_std_upper, args=(holds, labels), axis=1)

    df = outlier_description(df)

    return df


def dup_stats(df: pd.DataFrame, groups: tuple) -> pd.DataFrame:
    """
    Stats that can only be calculated for PINs occuring more than once, such as sale volatiltiy,
    and growth rates.
    Helper for create_stats().
    Inputs:
        df (pd.DataFrame): dataframe with sales data
        groups (tuple): for get_movement groups
    Outputs:
        df(pd.DataFrame): dataframe with sale counts and town_class movement columns.
    """
    dups = df[df.pin.duplicated(keep=False)]
    dups = get_sale_counts(dups)
    dups = get_movement(dups, groups)

    df = pd.merge(df, dups, how='outer')

    return df


def is_outlier_groupby(s: pd.Series, lower_lim : int, upper_lim: int) -> pd.DataFrame:
    """
    Finds values outside of std deviation range.
    Function meant for use in groupby apply() only.
    Inputs:
        s: pandas row
        lower_lim (int): lower std limit
        upper_lim (int): upper std limit
    Outputs:
        dataframe with only entries between
        lower_limit and upper_limit
    """
    lower_limit = s.mean() - (s.std() * lower_lim)
    upper_limit = s.mean() + (s.std() * upper_lim)
    return ~s.between(lower_limit, upper_limit)


def price_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates price/sqft columns in DataFrame. Must contain 'sale_price',
    'sale_price_log10' and 'sqft' in the columns, where the first two names are
    self explanatory and 'sqft' is the properties square footage.
    Helperfor create_stats().
    Inputs:
        df (pd.DataFrame): pandas dataframe with required columns.
    Outputs:
        df (pd.DataFrame): pandas dataframe with _per_sqft columns.
    """
    df['price_per_sqft'] = df['sale_price'] / df['sqft']
    df['price_per_sqft_log10'] = df['sale_price_log10'] / df['sqft']
    df['price_per_sqft'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['price_per_sqft_log10'].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def grouping_mean(df: pd.DataFrame, groups: tuple) -> pd.DataFrame:
    """
    Gets sale_price mean by two groupings. Usually town + class.
    Helper for create_stats().
    Inputs:
        df (pd.DataFrame): dataframe with the grouping columns
        groups (tuple): tuple (len == 2) where each element is a column name to be grouped by.
    Outputs:
        df (pd.DataFrame): dataframe with grouped by mean column
    """
    group1 = groups[0]
    group2 = groups[1]

    df['pct'] = df.sort_values('sale_date').groupby('pin')['sale_price_log10'].pct_change()
    group_mean = df.groupby([group1, group2])['sale_price_log10'].mean()
    df.set_index([group1, group2], inplace=True)
    df[f'{group1}_{group2}_mean_sale_log10'] = group_mean
    df[f'diff_from_{group1}_{group2}_mean_sale_log10'] = abs(df[f'{group1}_{group2}_mean_sale_log10'] - df['sale_price_log10'])
    df.reset_index(inplace=True)

    return df


def get_sale_counts(dups: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates how many times transactions occured for a gieven property.
    Helper for dup_stats()
    Inputs:
        df (pd.DataFrame): pandsa dataframe
    """
    v_counts = dups.pin.value_counts().reset_index().rename(columns={'index':'pin', 'pin':'counts'})
    dups = pd.merge(dups, v_counts, how='outer')

    return dups


def get_movement(dups: pd.DataFrame, groups:tuple) -> pd.DataFrame:
    """
    Creates a coloumn that determines whether the price movement of the records is
    towards or away from the mean.
    Helper for dup_stats().
    Inputs:
        df (pd.DataFrame): duplicate records
        groups (tuple): groupby groups
    Outputs:
        df (pd.DataFrame): duplicate records with new column
    """
    group1 = groups[0]
    group2 = groups[1]

    temp = dups.sort_values('sale_date').groupby(['pin'])[f'diff_from_{group1}_{group2}_mean_sale_log10'].shift()
    dups['town_class_movement'] = dups[f'diff_from_{group1}_{group2}_mean_sale_log10'].lt(temp).astype(float) # 0 is moving away, 1 is moving towards

    return dups


def transaction_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each record, gets number of days since the last transaction.
    Inputs:
        df (pd.DataFrame): DataFrame with a sale_date column in datetime
    Outputs:
        df (pd.DataFrame): DataFrame with new column
    """

    df['days_since_last_transaction'] = \
    df.sort_values('sale_date').groupby('pin')['sale_date'].diff().apply(lambda x: x.days)

    return df


def get_thresh(df: pd.DataFrame, cols: list, permut: tuple) -> dict:
    """
    Creates a nested dictionary where the top level key is a column
    and the 2nd-level key is a (township_code, class) combo.
    Ex: stds['sale_price'][76, 203]
    Inputs:
        df (pd.DataFrame): Dataframe to create dictionary from.
        cols (list): list of columns to get standard deviations for.
        permit (tuple): standard deviation range for lower_limit and upper_limit
                        First term is how many stndard deviations away on the left
                        Second term is how many standard deviations away on the right.
    Outputs:
        stds (dict): nested dictionary of std deviations for all columns
                     from DataFrame.
    """
    stds = {}

    for col in cols:
        grouped = df.dropna(subset=['township_code', col]).groupby(['township_code', 'class'])[col]
        lower_limit = grouped.mean() - (grouped.std() * permut[0])
        upper_limit = grouped.mean() + (grouped.std() * permut[1])
        std = grouped.std()

        lower_limit = lower_limit.to_dict()
        upper_limit = upper_limit.to_dict()
        std = std.to_dict()

        limits =  {x: (std.get(x, 0), lower_limit.get(x, 0), upper_limit.get(x, 0))
                    for x in set(std).union(upper_limit, lower_limit)}
        stds[col] = limits

    return stds


def over_std(df: pd.DataFrame, group: str, cols: list, permuts: tuple) -> pd.DataFrame:
    """
    Returns a DataFrame that includes outliers for each column, for the given grouping
    and std permutation.
    Inputs:
        df (pd.DataFrame): Dataframe to take columsn from.
        group (str): How to group in groupby. Most likely 'township_code'.
        cols (list): columns to get outliers for
        permuts (tuple): std range
    Outputs:
        all_outs (pd.DataFrame): DataFrame containing all outliers for given parameters.
    """
    outties = []

    for col in cols:
        if col == 'pct':
            df = df[df.town_class_movement == 0]
        outties.append(
            df.dropna(subset=[group, col])[df.groupby([group, 'class'])[col].apply(
                is_outlier_groupby, permuts[0], permuts[1])])

    all_outs = pd.concat(outties).drop_duplicates()

    return all_outs


def z_normalize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Do zscore normalization on given column set so that
    we can compare them apples to apples.
    Inputs:
        df (pd.DataFrame):
        columns (list): columsn to be normalized
    Outputs:
        df (pd.DataFrame): dataframe with given columns normalized
                           as 'column_name_zscore'
    """
    for col in columns:
        df[col + '_zscore'] = zscore(df[col], nan_policy='omit')
    
    return df


def primary_outlier(row, thresholds: dict, outliers: pd.DataFrame, columns: list, labels: dict):
    """
    Meant to be used as an apply() function for a pd.DataFrame.
    Determines the primary outlier for a record by finding
    largest std deviation of relevant column.
    Inputs:
        row (object): the row of the dataframe to be evaluated (passed from .apply()).
        thresholds (dict): nested dictionary of thresholds for each class/township combo
                           containing (lower_limit, upper_limit, std)
                           Generated by get_thresh().
        outliers (list): list containing the sale_key of all outliers for this column-set
                         and std deviation permutation.
        columns (list): column-set to be used for this function
                        (should be same as used in get_thresh(), over_std()).
        labels (dict): labels for column. Key is the name of a column,
                       value is the label to be mapped to that column if it is an outlier.
    Outputs:
        value (str): Value of the primary_outlier column.
                     Chosen from labels.
    """

    if row['buyer_id'] == row['seller_id'] and row['buyer_id'] != 'Empty Name':
        value = 'Buyer ID and Seller ID match'
    elif row['sale_key'] not in outliers:
        value = 'Not Outlier'
    else:
        stds = {}
        for col in columns:
            if thresholds.get(col).get((row['township_code'], row['class'])) and pd.notnull(row[col]):
                std, *std_range = thresholds.get(col).get((row['township_code'], row['class']))
                stds[col] = abs(std)

        highest = max(stds, key=stds.get)
        value = labels[highest]

    return value


def outlier_description(df: pd.DataFrame):
    """
    Runs np.select that creates more detailed description of what the outlier is.
    Inputs:
        df (pd.DataFrame): dataframe with necessary columns created from previous apply() functions.
    Outputs:
        df (pd.DataFrame): dataframe with 'outlier_description' column.
    """
    conditions = [
    (df['primary_outlier'] == 'Price Change Outlier') & (df['outlier_value'] > df['outlier_std_upper']),
    (df['primary_outlier'] == 'Price Change Outlier') & (df['outlier_value'] < df['outlier_std_lower']),
    (df['primary_outlier'] == 'Raw Price Outlier') & (df['outlier_value'] > df['outlier_std_upper']),
    (df['primary_outlier'] == 'Raw Price Outlier') & (df['outlier_value'] < df['outlier_std_lower']),
    (df['primary_outlier'] == 'Price/SQFT Outlier') & (df['outlier_value'] > df['outlier_std_upper']),
    (df['primary_outlier'] == 'Price/SQFT Outlier') & (df['outlier_value'] < df['outlier_std_lower']),
    (df['primary_outlier'] == 'Days Since Last Transaction Outlier') & (df['outlier_value'] > df['outlier_std_upper']),
    (df['primary_outlier'] == 'Days Since Last Transaction Outlier') & (df['outlier_value'] < df['outlier_std_lower']),
    (df['primary_outlier'] == 'Transaction Volatility Outlier') & (df['outlier_value'] > df['outlier_std_upper']),
    (df['primary_outlier'] == 'Transaction Volatlity Outlier') & (df['outlier_value'] < df['outlier_std_lower'])]
    labels = ['Price change increasing from mean','Price change descreasing from mean',
              'Valuation Outlier above the mean', 'Valuation Outlier below the mean',
              'Per SQFT Outlier above the mean','Per SQFT Outlier below the mean',
              'Days since last transaction is above mean', 'Days since last transaction is below mean',
              'Number of transactions is above mean','Number of transactions is below mean']
    df["outlier_description"] = np.select(conditions, labels, default=np.nan)

    return df


def outlier_value(row: pd.Series, labels: dict):
    """
    Uses the mapping of column : primary_outlier label to retrieve
    the actual outlier value from the row passed from apply().
    Inputs:
        row: passed from apply()
        labels (dict): mapping of column: primary_outlier label
                       Reversed and used to retrieve actual value of
                       the outlier column.
    Outputs:
        value: Whatever the actual outlying value is.
               Most likely int or float.

    """
    reverse_labels = {v: k for k, v in labels.items()}
    if row['primary_outlier'] not in labels.values():
        value = None
    else:
        value = row[reverse_labels[row['primary_outlier']]]

    return value


def outlier_value_std(row: pd.Series, thresholds: dict, labels: dict):
    """
    Retrieves the standard deviation for the outlier column.
    Inputs:
        row: passed from apply()
        thresholds (dict): nested dictionary from get_thresh()
        labels (dict): mapping of columns: outlier mappings
    Ouputs:
        value (float): the std deviation for the value that was
        selected as having the greatest standard deviation.
    """
    reverse_labels = {v: k for k, v in labels.items()}

    if row['primary_outlier'] == 'Not Outlier':
        value = None
    elif row['primary_outlier'] == 'Buyer ID and Seller ID match':
        return row['seller_id']
    else:
        col = reverse_labels[row['primary_outlier']]
        std, *std_range = thresholds.get(col).get((row['township_code'], row['class']))
        value = std

    return value


def outlier_std_lower(row, thresholds: dict, labels: dict):
    """
    Fetches the lower_limit of the standard deviation range.
    Inputs:
        row: passed from apply().
        thresholds (dict): nested thersholds dictionary from get_thresh()
        labels (dict): columns: primary_outlier value mapping
    Outputs:
        value(float): lower_limit of the standard deviation threshold.
    """
    reverse_labels = {v: k for k, v in labels.items()}

    if reverse_labels.get(row['primary_outlier']):
        col = reverse_labels[row['primary_outlier']]
        if thresholds.get(col).get(row['township_code'], row['class']):
            std, *std_range = thresholds[col][row['township_code'], row['class']]
            value = tuple(std_range)
            value = value[0]
    else:
        value = None

    return value


def outlier_std_upper(row, thresholds: dict, labels: dict):
    """
    Fetches the upper_limit of the standard deviation range.
    Inputs:
        row: passed from apply().
        thresholds (dict): nested thersholds dictionary from get_thresh()
        labels (dict): columns: primary_outlier value mapping
        Outputs:
        value(float): upper_limit of the standard deviation threshold.
    """
    reverse_labels = {v: k for k, v in labels.items()}

    if reverse_labels.get(row['primary_outlier']):
        col = reverse_labels[row['primary_outlier']]
        if thresholds.get(col).get(row['township_code'], row['class']):
            std, *std_range = thresholds[col][row['township_code'], row['class']]
            value = tuple(std_range)
            value = value[1]
    else:
        value = None

    return value


# STRING CLEANUP

"""
    An outline of our overall approach:

    Tries to create an identifier from the buyer/seler name.
    Our appraoch is to try to identify if it is a legal identify of some sort,
    such as a bank, construction company, trust, LLC, or other and
    return the string as-is with some formatting applied if so. We also combine some
    spellings/mispellings of big entities.

    If we can't identify the string as a legal entity we assume the string contains a person's name.
    We then process these strings to determine if the person is a trustee, successor,
    or a successor trustee from the fragements of the strings.
    Once we do this, we determine the best place tosplit the string in split_logic(),
    looking out for certain tokens. After we've determnined where to split
    the string we send the tokens to name_selector, where we attempt to select
    the last name of the string.

    We then create a column that tells us whether it's person, or a legal entity,
    as per our identification method that we used in get_id().

    Then we use the trustee, successor, or as successor trustee parts of
    the string we constructed earlier to determine the role of the buyer
    or seller in the transaction(trustee, successor, successor trustee).

    We then remove the trustee, successor, as successor trustee parts of the string
    from buyer/seller id.

    Finally we create a transaction_type column that is just what kind of entity it is
    with a dash between them.

    TODO: Process more strings types:
        - If a name contains 'and', we split the string on it and take
          the token directly to the left. We could take a more sophisticated
          approach to determine if the last name in this case.
        - 'co-trustee' handling.
        -  Handle different name formats. Assume people use <FIRST M LAST>
           but sometimes its <LAST FIRST M> or other such formats.
        - Find trends in string cutoffs(some are cut off at 25, characters, others 25, etc)
          that could help use better process strings that are cutoff.
        - Cleanup/debug regex. This is a lot of dirty regex, and it is picking up
          some names that we don't want, or not correctly identifying every case that we do want.
          So it could use some work in some cases.
    """


entity_keywords = r"llc| ll$| l$|l l c|estate|training|construction|building|masonry|apartments|plumbing|service|professional|roofing|advanced|office|\blaw\b|loan|legal|production|woodwork|concepts|corp| company| united|\binc\b|county|entertainment|community|heating|cooling|partners|equity|indsutries|series|revitalization|collection|agency|renovation|consulting|flippers|estates|\bthe \b|dept|funding|opportunity|improvements|servicing|equities|sale|judicial| in$|bank|\btrust\b|holding|investment|housing|properties|limited|realty|development|capital|management|developers|construction|rentals|group|investments|invest|residences|enterprise|enterprises|ventures|remodeling|specialists|homes|business|venture|restoration|renovations|maintenance|ltd|real estate|builders|buyers|property|financial|associates|consultants|international|acquisitions|credit|design|homeownership|solutions|home|diversified|assets|family|land|revocable|services|rehabbing|living|county of cook|fannie mae|land|veteran|mortgage|savings|lp$"


def get_id(row, col: str):
    """ 
    Creates an ID from the buyer/seller name.

    Returns string as-is if identified as legal entity.
    Combined with other entities if its a common mispelling/cutoff.

    Attempts to identify last name if not a legal entity.

    Inputs:
        row: from apply()
        col (str): 'buyer' or 'seller'
    Outputs:
        id (str): string as-is if legal entity
                  identified last name if otherwise.
    """

    column = col + '_name'
    words = str(row[column]).lower()

    words = re.sub(r' amp ','', words)
    words = re.sub(' +', ' ', words)

    if words.isspace() or re.search(r'^[.]*$', words):
        id = 'Empty Name'
        return id

    if any(x in words for x in ['vt investment corpor', 'v t investment corp']):
        return 'vt investment corporation'

    if any(x in words for x in ['first integrity group inc', 'first integrity group in']):
        return 'first integrity group inc'

    if words in ['deutsche bank national tr']:
        return 'deutsche bank national trust company'

    if any(x in words for x in ['cirrus investment group l', 'cirrus investment group']):
        return 'cirrus investment group'

    if any(x in words for x in ['fannie mae aka federal na',  'fannie mae a k a federal', 'federal national mortgage']):
        return 'fannie mae'

    if any(x in words for x in ['the judicial sales corpor', 'judicial sales corp', 'judicial sales corporatio', 'judicial sale corp', 'the judicial sales corp']):
        return 'the judicial sales corporation'

    if any(x in words for x in ['jpmorgan chase bank n a', 'jpmorgan chase bank nati']):
        return 'jp morgan chase bank'

    if any(x in words for x in ['wells fargo bank na',  'wells fargo bank n a',  'wells fargo bank nationa',  'wells fargo bank n a a']):
        return 'wells fargo bank national'

    if any(x in words for x in ['bayview loan servicing l',  'bayview loan servicing ll']):
        return 'bayview loan servicing llc'

    if any(x in words for x in ['thr property illinois l', 'thr property illinois lp']):
        return 'thr property illinois lp'

    if any(x in words for x in ['ih3 property illinois lp', 'ih3 property illinois l']):
        return 'ih3 property illinois lp'

    if any(x in words for x in ['ih2 property illinois lp', 'ih2 property illinois l']):
        return 'ih2 property illinois lp'

    if any(x in words for x in ['secretary of housing and',  'the secretary of housing', 'secretary of housing ']):
        return 'secretary of housing and urban development'

    if any(x in words for x in ['secretary of veterans aff', 'the secretary of veterans']):
        return 'secretary of veterans affairs'

    if any(x in words for x in ['bank of america n a', 'bank of america na', 'bank of america national',]):
        return 'bank of america national'

    if any(x in words for x in ['us bank national association', 'u s bank national assoc', 'u s bank national associ', 'u s bank trust n a as', 'u s bank n a', 'us bank national associat', 'u s bank trust national']):
        return 'us bank national association'

    words = re.sub('suc t$|as succ t$|successor tr$|successor tru$|successor trus$|successor trust$|successor truste$|successor trustee$|successor t$|as successor t$',
                   'as successor trustee', words)
    words = re.sub('as t$|as s t$|as sole t$|as tr$|as tru$|as trus$|as trust$|as truste$|as trustee$|as trustee o$|as trustee of$|trustee of$|trustee of$|tr$|tru$|trus$|truste$|trustee$|, t|, tr|, tru|, trus|, trust|, truste',
                   'as trustee', words)
    words = re.sub('su$|suc$|succ$|succe$|succes$|success$|successo$|successor$|as s$|as su$|as suc$|as succ$|as succe$|as sucess$|as successo$|, s$|, su$|, suc$|, succ$|, succe$|, succes$|, success$|, successo$',
                   'as successor', words)

    if re.search(entity_keywords, words) or re.search(r'\d{4}|\d{3}', words) or re.search('as trustee$|as successor$|as successor trustee$', words):
        id = words
        return id

    words = re.sub(' in$|indi$|indiv$|indivi$|indivi$|individ$|individu$|individua$|individual$|not i$|not ind$| ind$| inde$|indep$|indepe$|indepen$|independ$|independe$|independen$|independent$',
                   '', words)

    tokens = split_logic(words)

    id = name_selector(tokens)

    return id


def split_logic(words: str):
    """
    Given a cleaned string, determines where to split the string.
    Splits on 'and', variations of FKA/NKA/KNA if present, on spaces if not.
    Helper to get_id().
    Inputs:
        words (str): cleaned str from get_id
    Outputs:
        'Empty Name' if string is empty
        tokens (list): lsit of tokens in string from split
    """

    words = re.sub(' +', ' ', words)

    if words.isspace() or re.search(r'^[.]*$', words) or words == 'Empty Name':
        return 'Empty Name'

    words = re.sub(' as$| as $|as $','', words)

    _and = re.search(r'\b and\b|\b an$\b|\b a$\b|f k a|\bfka\b| n k a|\bnka\b|\b aka\b|a k a|\b kna\b|k n a| f k$|n k$|a k$|\b not\b| married', words)

    if _and:
        tokens = words.split(_and.group())
        tokens = tokens[0].strip().split()
    else:
        tokens = words.split()

    return tokens


def name_selector(tokens):
    """
    Attempts to select the last name of a persons name based on number of tokens.
    Inputs:
        tokens: name to be identified
    Outputs:
        'Empty Name' if name is empty.
        id (str): identified last name.
    """
    if tokens == 'Empty Name':
        return tokens

    if tokens[-1] in ['jr', 'sr', 'ii', 'iii', 'iv', 'v']:
        tokens = tokens[:-1]
    #Ex: John Smith
    if len(tokens) == 2:
        id = tokens[1]
    # John George Smith
    if len(tokens) == 3:
        id = tokens[2]
    # John George Theodore Smith
    else:
        id = tokens[-1]

    return id


def get_category(row, col: str):

    column = col + '_id'
    words = row[column]

    if re.search(entity_keywords, words):
        return 'legal_entity'
    elif words == 'Empty Name':
        return 'none'
    else:
        return 'person'


def get_role(row, col: str):
    role = None
    column = col + '_id'
    words = row[column]

    suc_trust = re.search(' as successor trustee' , words)
    suc = re.search(' as successor' , words)
    trust = re.search(' as trustee' , words)

    if suc_trust:
        role = suc_trust.group()

    if suc:
        role = suc.group()

    if trust:
        role = trust.group()

    return role


def clean_id(row: object, col: str):
    column = col + '_id'
    words = row[column]

    words = re.sub(r' as successor trustee|\b as successor\b| as trustee', '', words)
    words = re.sub(' as$| as $|as $','', words)

    if not (re.search(entity_keywords, words) or re.search(r'\d{4}|\d{3}', words) or len(words.split()) == 1):
        words = name_selector(split_logic(words))

    return words


def transaction_type(row):
    buyer = row['buyer_category']
    seller = row['seller_category']

    return buyer + '-' + seller


def string_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Brings together all of the apply functions for string processing.
    Results in 7 additional columns.
    ID, category, and role for buyer and seller. As well as transaction category type
    for each record.
    Inputs:
        df (pd.dataFrame): dataframe with buyer/seller id columns.
    Ouputs:
        df(pd.DataFrame): dataframe with 7 new columns from apply functions
    """
    df.buyer_name = df.buyer_name.str.encode('ascii', 'ignore').str.decode('ascii')
    df.seller_name = df.seller_name.str.encode('ascii', 'ignore').str.decode('ascii')
    df.buyer_name = df.buyer_name.str.replace(r'[^a-zA-Z0-9\-]', ' ', regex=True).str.strip()
    df.seller_name = df.seller_name.str.replace(r'[^a-zA-Z0-9\-]', ' ', regex=True).str.strip()

    df['buyer_id'] = df.apply(get_id, args=('buyer',), axis=1)
    df['seller_id'] = df.apply(get_id, args=('seller',), axis=1)
    df['buyer_category'] = df.apply(get_category, args=('buyer',), axis=1)
    df['seller_category'] = df.apply(get_category, args=('seller',), axis=1)
    df['buyer_role'] = df.apply(get_role, args=('buyer',), axis=1)
    df['seller_role'] = df.apply(get_role, args=('seller',), axis=1)
    df['buyer_id'] = df.apply(clean_id, args=('buyer',), axis=1)
    df['seller_id'] = df.apply(clean_id, args=('seller',), axis=1)
    df['transaction_type'] = df.apply(transaction_type, axis=1)

    return df


columns = ['sale_price_log10', 'price_per_sqft_log10', 'pct', 'counts', 'days_since_last_transaction']
labels = {
    'pct_zscore': 'Price Change Outlier',
    'sale_price_log10_zscore' : 'Raw Price Outlier',
    'price_per_sqft_log10_zscore': 'Price/SQFT Outlier', 
    'counts_zscore': 'Transaction Volatility Outlier',
    'days_since_last_transaction_zscore': 'Days Since Last Transaction Outlier'}
    #'seller_id': 'Buyer ID and Seller ID match'}

go(columns, (2,2), ('township_code', 'class'), labels, 'flagged.csv')
