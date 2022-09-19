"""
This file contains all necessary functions to create a DataFrame ready to use for
non-arms length transaction detection using statistical and heurstic methods.
"""

import argparse
import re
import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

SHORT_TERM_OWNER_THRESHOLD = 365 # 365 = 365 days or 1 year


def go(permut: tuple, groups: tuple, output_file: str):

    df = read_data('sale_sample_18-21.parquet', 'cards.csv', 'char_sample.csv')

    df = create_stats(df, groups)

    df = string_processing(df)

    # need all important info about transaction
    df = iso_forest(df, groups, ['sale_price', 'price_per_sqft', 'days_since_last_transaction', 'pct', 'counts'])

    df = outlier_taxonomy(df, permut, groups)

    df = analyst_readable(df, groups)

    df.to_csv(output_file, index=False)


def outlier_taxonomy(df: pd.DataFrame, permut: tuple, groups: tuple):
    """
    Creates columns having to do with our chosen outlier taxonomy.
    Ex: Family sale, Home flip sale, Non-person sale, High price (raw and or sqft), etc.
    Inputs:
        df (pd.DataFrame): dataframe to create taxonomy on.
        permut (tuple): permutation of std deviations
        groups (tuple): columns to do grouping on.
                        Probably 'township_code' and 'class'.
    Ouputs:
        df (pd.DataFrame): dataframe with outlier taxonomy
    """

    df = check_days(df, SHORT_TERM_OWNER_THRESHOLD)

    df = pricing_info(df, permut, groups)

    df = outlier_type(df)
    df = outlier_flag(df)
    df = special_flag(df)

    return df


def analyst_readable(df, groups):
    """
    A function that helps make the resulting spreadsheet more readable for analysts.
    Relies on our existing outlier taxonomy.
    """

    df.set_index('sale_key', inplace=True)
    outs = df[df['is_outlier'] == 'Outlier']

    df['price_per_sqft_deviation_class_township_percentile'] = outs.groupby(list(groups))['price_per_sqft_deviation_class_township'].rank(pct=True)
    df['price_deviation_class_township_percentile'] = outs.groupby(list(groups))['price_deviation_class_township'].rank(pct=True)
    df['price_per_sqft_deviation_class_township_rank'] = outs.groupby(list(groups))['price_per_sqft_deviation_class_township'].rank()
    df['price_deviation_class_township_rank'] = outs.groupby(list(groups))['price_deviation_class_township'].rank()
    df['outlier_description'] = df.apply(outlier_description, axis=1)

    df.reset_index(inplace=True)

    df['pin'] = df['pin'].astype(str).str.pad(14,fillchar='0')

    df.sort_values(by=['outlier_type'], ascending=[True], inplace=True)

    df = df[['doc_no', 'deed_type', 'township_code','pin', 'class',
       'sale_date', 'seller_name', 'buyer_name', 'outlier_type', 'outlier_description',
       'pricing', 'special_flags', 'sale_price', 'price_per_sqft', 'sqft',
       'price_deviation_class_township', 'price_per_sqft_deviation_class_township',
       'sale_price_deviation_county', 'price_per_sqft_deviation_county',
       'pct', 'pct_deviation_class_township',
       'price_deviation_class_township_percentile', 'price_per_sqft_deviation_class_township_percentile',
       'price_per_sqft_deviation_class_township_rank', 'price_deviation_class_township_rank',
       'name_match', 'anomaly', 'short_owner', 'previous_price',
       'is_sale_between_related_individuals_or_corporate_affiliates',
       'is_transfer_of_less_than_100_percent_interest',
       'is_court_ordered_sale', 'is_sale_in_lieu_of_foreclosure',
       'is_short_sale', 'is_bank_reo_real_estate_owned', 'is_auction_sale',
       #'is_seller_buyer_a_relocation_company',
       'is_seller_buyer_a_financial_institution_or_government_agency',
       'is_buyer_a_real_estate_investment_trust', 'is_buyer_a_pension_fund',
       'is_buyer_an_adjacent_property_owner',
       'is_buyer_exercising_an_option_to_purchase',
       'is_simultaneous_trade_of_property', 'is_sale_leaseback','is_judicial_sale',
       'sale_type', 'price_movement', 'days_since_last_transaction',
       'transaction_type']]
    df['deed_type'] = np.select([(df['deed_type'] == '01'), (df['deed_type'] == '02'),
                                (df['deed_type'] == '03'), (df['deed_type'] == '04'),
                                 (df['deed_type'] == '05'), (df['deed_type'] == '06'),
                                 (df['deed_type'] == '99')],
                                 ['Warranty Deed', 'Trustee Deed', 
                                 'Quit Claim Deed','Executor Deed',
                                 'Other', 'Beneficial Intrst',
                                 'Unknown'])
    townships = [(df['township_code'] == 10), (df['township_code'] == 11),
                 (df['township_code'] == 12), (df['township_code'] == 13),
                 (df['township_code'] == 14), (df['township_code'] == 15),
                 (df['township_code'] == 16), (df['township_code'] == 17),
                 (df['township_code'] == 18), (df['township_code'] == 19),
                 (df['township_code'] == 20), (df['township_code'] == 21),
                 (df['township_code'] == 22), (df['township_code'] == 23),
                 (df['township_code'] == 24), (df['township_code'] == 25),
                 (df['township_code'] == 26), (df['township_code'] == 27),
                 (df['township_code'] == 28), (df['township_code'] == 29),
                 (df['township_code'] == 30), (df['township_code'] == 31),
                 (df['township_code'] == 32), (df['township_code'] == 33),
                 (df['township_code'] == 34), (df['township_code'] == 35),
                 (df['township_code'] == 36), (df['township_code'] == 37),
                 (df['township_code'] == 38), (df['township_code'] == 39),
                 (df['township_code'] == 70), (df['township_code'] == 71),
                 (df['township_code'] == 72), (df['township_code'] == 73),
                 (df['township_code'] == 74), (df['township_code'] == 75),
                 (df['township_code'] == 76), (df['township_code'] == 77)]
    town_names = ['Barrington', 'Berwyn', 'Bloom','Bremen', 'Calumet', 'Cicero',
                  'Elk Grove', 'Evanston', 'Hanover', 'Lemont', 'Leyden', 'Lyons',
                  'Maine', 'New Trier', 'Niles', 'Northfield', 'Norwood Park',
                  'Oak Park', 'Orland', 'Palatine', 'Palos', 'Proviso', 'Rich',
                  'River Forest', 'Riverside', 'Schaumburg', 'Stickney',
                  'Thornton', 'Wheeling', 'Worth', 'Hyde Park', 'Jefferson',
                  'Lake', 'Lake View', 'North Chicago', 'Rogers Park', 'South Chicago',
                  'West Chicago']

    df['township_code'] = np.select(townships, town_names)

    df.rename(columns={'counts': 'number_of_transactions',
                       'township_code': 'township'}, inplace=True)

    df['sale_date'] = df['sale_date'].dt.strftime('%Y-%m-%d')
    df['pct'] = df['pct'].fillna(0)
    df['pct_deviation_class_township'] = df['pct_deviation_class_township'].fillna(0)
    df['previous_price'] = df['previous_price'].fillna(0)
    df['days_since_last_transaction'] = df['days_since_last_transaction'].fillna(0)
    df['price_movement'] = df['price_movement'].fillna('First sale')

    return df


def iso_forest(df: pd.DataFrame,
               groups: tuple,
               columns: list,
               n_estimators: int = 1000,
               max_samples: int or float = .2) -> pd.DataFrame:
    """
    Runs an isolation forest model on our data for outlier detection.
    First does PCA, then, attaches township/class info, and then runs the
    IsoForest model with given parameters.
    Inputs:
        df (pd.DataFrame): dataframe with data for IsoForest
        groups (tuple): grouping for the data to input into the IsoForest
        columns (list): list with columns to run PCA/IsoForest on
        n_estimators (int): number of estimators in IsoForest
        max_samples(int or float): share of data to use as sample if float,
                                   number to use if int
    Outputs:
        df (pd.DataFrame): with 'anomaly' column from IsoForest.
    """
    group1 = groups[0]
    group2 = groups[1]

    df.set_index('sale_key', inplace=True)

    feed = pca(df, columns)

    feed.index = df.index
    feed[group1] = df[group1]
    feed[group2] = df[group2]

    isof = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, bootstrap=True, random_state=42)
    df['anomaly'] = isof.fit_predict(feed)

    df['anomaly'] = np.select([(df['anomaly'] == -1), (df['anomaly'] == 1)],
                                          ['Outlier', 'Not Outlier'], default= 'Not Outlier')

    df.reset_index(inplace=True)

    return df


def pca(df:pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Runs PCA on data, selects compoents where explained variance > 1.
    Inputs:
        df (pd.DataFrame): dataframe to run PCA on.
        columns (list): columns of dataframe to run PCA on.
    Outputs:
        df (pd.DataFrame): dataframe of principal components
    """

    feed_data = df[columns]
    feed_data.fillna(0, inplace=True)

    pca = PCA(n_components = len(feed_data.columns))
    pc = pca.fit_transform(feed_data)

    cols = ['PC' + str(num) for num in range(len(feed_data.columns))]

    pc_df = pd.DataFrame(data = pc, 
                         columns = cols)
    take = len(pca.explained_variance_[pca.explained_variance_ > 1])

    df = pc_df[pc_df.columns[:take]]

    return df


def read_data(sales_name: str, card_name: str, char_name: str) -> pd.DataFrame:
    """
    Read in data from multiple sources and merge them.
    Inputs:
        sales_name (str): name of sales data file
        card_name (str): name of card data file
        char_name (str): name of characteristics file
    Outputs:
        df (pd.DataFrame): dataframe of merged data
    """

    sales = pd.read_parquet(sales_name)
    cards = pd.read_csv(card_name)
    char_sample = pd.read_csv(char_name, dtype={'class': str}) # also has 'EX' in column

    cards.drop_duplicates(inplace=True)
    # some are duplicates other than sqft and year built, same year and other info but diff sqft
    # so we drop it cuz this look anamolous - this also probably has to do with multi_cards?
    char_sample.drop_duplicates(subset=['year', 'pin', 'class'], inplace=True)

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


def pricing_info(df: pd.DataFrame, permut: tuple, groups: tuple) -> pd.DataFrame:
    """
    Creates information about whether the price is an outlier, and its movement.
    Also fetches the sandard deviation for the record.
    pricing is whether it is a high/low outlier and whether it is a price swing.
    which_price is whether it is the raw price, price/sqft or both that are outliers.
    std_deviation is the std deviation for the raw price for that records class/township.
    Inputs:
        df (pd.DataFrame): dataframe of sales
        permut (tuple): tuple of standard deviation boundaries.
                        Ex: (2,2) is 2 std away on both sides.
    Outputs:
        df (pd.DataFrame): dataframe with 3 extra columns of price info.
    """

    df = z_normalize(df, ['sale_price', 'price_per_sqft'])

    prices = ['price_per_sqft_deviation_class_township',
              'price_deviation_class_township', 'pct_deviation_class_township']

    df['price_deviation_class_township'] = df.groupby(list(groups))['sale_price'].apply(z_normalize_groupby)
    df['price_per_sqft_deviation_class_township'] = df.groupby(list(groups))['price_per_sqft'].apply(z_normalize_groupby)
    df['pct_deviation_class_township'] = df.groupby(list(groups))['pct'].apply(z_normalize_groupby)

    holds = get_thresh(df, prices, permut)

    df['pricing'] = df.apply(price_column, args=(holds,), axis=1)
    df['which_price'] = df.apply(which_price, args=(holds,), axis=1)

    return df


def special_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates column that checks whether there is a special flag for this record.
    Inputs:
        df (pd.DataFrame): dataframe to add flags onto
    Outputs:
        df (pd.DataFrame): dataframe with 'special_flags' column
    """
    cond = [(df['name_match'] != 'No match'), (df['short_owner'] == 'Short-term owner'),
            (df['transaction_type'] == 'legal_entity-legal_entity')]
    labels = ['Family sale', 'Home flip sale', 'Non-person sale']

    df['special_flags'] = np.select(cond, labels, default = 'Not special')

    return df


def which_price(row: pd.Series, thresholds: dict) -> str:
    """
    Determines whether sale_price, price_per_sqft, or both are outliers,
    and returns a string resembling it.
    Inputs:
        thresholds (dict): dict of thresholds from get_thresh
    Outputs:
        value (str): string saying which of these are outliers.
    """

    value = 'Non-outlier'

    if thresholds.get('price_deviation_class_township').get((row['township_code'], row['class'])) and \
        thresholds.get('price_per_sqft_deviation_class_township').get((row['township_code'], row['class'])):
        s_std, *s_std_range = thresholds.get('price_deviation_class_township').get((row['township_code'], row['class']))
        s_lower, s_upper = s_std_range
        sq_std, *sq_std_range = thresholds.get('price_per_sqft_deviation_class_township').get((row['township_code'], row['class']))
        sq_lower, sq_upper = sq_std_range
        if not between_two_numbers(row['price_deviation_class_township'], s_lower, s_upper) and \
            between_two_numbers(row['price_per_sqft_deviation_class_township'], sq_lower, sq_upper):
            value = '(raw)'
        elif between_two_numbers(row['price_deviation_class_township'], s_lower, s_upper) and \
            not between_two_numbers(row['price_per_sqft_deviation_class_township'], sq_lower, sq_upper):
            value = '(sqft)'
        elif not between_two_numbers(row['price_deviation_class_township'], s_lower, s_upper) and \
            not between_two_numbers(row['price_per_sqft_deviation_class_township'], sq_lower, sq_upper):
            value = '(raw & sqft)'

    return value


def between_two_numbers(num: int or float, a: int or float, b: int or float) -> bool:
        return  a < num < b


def price_column(row: pd.Series, thresholds: dict) -> str:
    """
    Determines whether the record is a high price outlier or a low price outlier.
    If the record is also a price change outlier, than add 'swing' to the string.
    Inputs:
        thresholds (dict): dict of standard deviation thresholds from get_thresh()
    Outputs:
        value (str): string showing what kind of price outlier the record is.
    """
    value = 'Not price outlier'
    price = False

    if thresholds.get('price_deviation_class_township').get(
        (row['township_code'], row['class'])) and\
        thresholds.get('price_per_sqft_deviation_class_township').get(
            (row['township_code'], row['class'])):
        s_std, *s_std_range = thresholds.get('price_deviation_class_township').get(
            (row['township_code'], row['class']))
        s_lower, s_upper = s_std_range
        sq_std, *sq_std_range = thresholds.get('price_per_sqft_deviation_class_township').get(
            (row['township_code'], row['class']))
        sq_lower, sq_upper = sq_std_range

        if row['price_deviation_class_township'] > s_upper or\
            row['price_per_sqft_deviation_class_township'] > sq_upper:
            value = 'High price'
            price = True
        elif row['price_deviation_class_township'] < s_lower or\
            row['price_per_sqft_deviation_class_township'] < sq_lower:
            value = 'Low price'
            price = True

        if price and pd.notnull(row['pct_deviation_class_township']) and\
            thresholds.get('pct_deviation_class_township').get(
                (row['township_code'], row['class'])):
            # not every class/township combo has pct change info so we need this check
            p_std, *p_std_range = thresholds.get('pct_deviation_class_township').get(
                (row['township_code'], row['class']))
            p_lower, p_upper = p_std_range
            if row['price_movement'] == 'Away from mean' and \
                not between_two_numbers(row['pct_deviation_class_township'], p_lower, p_upper):
                value += ' swing'

    return value


def create_stats(df: pd.DataFrame, groups: tuple) -> pd.DataFrame:
    """
    Create all statistical outlier measures.
    Inputs:
        df (pd.DataFrame): Dataframe to create statistics from
        groups (tuple): grouping for groupby. Usually 'township_code' and 'class'
    Outputs:
        df(pd.DataFrame): dataframe with statistical measures calculated.
    """

    df = price_sqft(df)
    df = grouping_mean(df, groups)
    df = dup_stats(df, groups)
    df = transaction_days(df)
    df = percent_change(df)

    return df


def percent_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates CGR for all records. Requires that transaction_days() has already been run.
    Creates 'previous_price' column as intermediary to help calculate CGR.
    Calculate the compound growth rate where the previous transaction is the
    beginning value, the current price is the end value, and the number of periods
    is the number of days since the last transaction.
    This enables us to better compare percent change accross different time periods
    as opposed to pandas pct_change() function which does not account for time period.
    Helper for create_stats().
    Inputs:
        df (pd.DataFrame): datarame to create CGR on.
    Outputs:
        df (pd.DataFrame): dataframe with CGR statistic and previous_price column
    """

    df['previous_price'] = df.sort_values('sale_date').groupby(['pin'])['sale_price'].shift(axis=0)
    df['pct'] = ((df['sale_price'] / df['previous_price']) ** (1 / df['days_since_last_transaction'])) - 1

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


def price_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates price/sqft columns in DataFrame. Must contain 'sale_price',
    'sale_price_log10' and 'sqft' in the columns, where the first two names are
    self explanatory and 'sqft' is the properties square footage.
    Helper for create_stats().
    Inputs:
        df (pd.DataFrame): pandas dataframe with required columns.
    Outputs:
        df (pd.DataFrame): pandas dataframe with _per_sqft columns.
    """
    df['price_per_sqft'] = df['sale_price'] / df['sqft']
    df['price_per_sqft'].replace([np.inf, -np.inf], np.nan, inplace=True)

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

    group_mean = df.groupby([group1, group2])['sale_price'].mean()
    df.set_index([group1, group2], inplace=True)
    df[f'{group1}_{group2}_mean_sale'] = group_mean
    df[f'diff_from_{group1}_{group2}_mean_sale'] = abs(df[f'{group1}_{group2}_mean_sale'] - df['sale_price'])
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

    temp = dups.sort_values('sale_date').groupby(['pin'])[f'diff_from_{group1}_{group2}_mean_sale'].shift()
    dups['price_movement'] = dups[f'diff_from_{group1}_{group2}_mean_sale'].lt(temp).astype(float)
    dups['price_movement'] = np.select([(dups['price_movement'] == 0), (dups['price_movement'] == 1)],
                                        ['Away from mean', 'Towards mean'], default='First sale')

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


def check_days(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Creates a label of whether or not the transaction
    was only owned for a short term.
    If owned for less than the threshold, is a short term owner.
    Inputs:
        df (pd.DataFrame): dataframe to have short term owners checked
        threshold (int): the threshold fo being a short term owner
    Oututs:
        df (pd.DataFrame): datafrme with 'short_owner' column
    """
    df['short_owner'] = np.select([(df['days_since_last_transaction'] < threshold)],
                                  ['Short-term owner'], default = f'Over {threshold} days')

    return df


def get_thresh(df: pd.DataFrame, cols: list, permut: tuple) -> dict:
    """
    Creates a nested dictionary where the top level key is a column
    and the 2nd-level key is a (township_code, class) combo.
    Ex: stds['sale_price'][76, 203]
    Needed in order to keep track of specific thresholds for each township/class combo.
    Theoretically each std should be 1(because of z_normalization), but in practical terms
    it is in a very very small range around 1, so using a uniform cutoff of 2 and -2 loses us some precision.

    We also want to allow for some flexibility in how the thresholds are calculated; and this function allows for
    more flexbility in the event of future changes.
    Inputs:
        df (pd.DataFrame): Dataframe to create dictionary from.
        cols (list): list of columns to get standard deviations for.
        permut (tuple): standard deviation range for lower_limit and upper_limit
                        First term is how many stndard deviations away on the left
                        Second term is how many standard deviations away on the right.
    Outputs:
        stds (dict): nested dictionary of std deviations for all columns
                     from DataFrame.
    """
    stds = {}

    for col in cols:
        df[col] = df[col].astype(float)
        grouped = df.dropna(subset=['township_code', col]).groupby(['township_code', 'class'])[col]
        lower_limit = grouped.mean() - (grouped.std(ddof=0) * permut[0])
        upper_limit = grouped.mean() + (grouped.std(ddof=0) * permut[1])
        std = grouped.std(ddof=0)
        lower_limit = lower_limit.to_dict()
        upper_limit = upper_limit.to_dict()
        std = std.to_dict()
        print(std)

        limits =  {x: (std.get(x, 0), lower_limit.get(x, 0), upper_limit.get(x, 0))
                for x in set(std).union(upper_limit, lower_limit)}
        stds[col] = limits

    return stds


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
        df[col + '_deviation_county'] = zscore(df[col], nan_policy='omit')

    return df


def z_normalize_groupby(s: pd.Series):
    """
    Function used to z_normailize groups of records.
    Pandas stiches it back together into a complete column.
    Meant for groupby.apply().
    Inputs:
        s(pd.Series): grouped series from groupby.apply
    Ouputs:
        z_normalized series grouped by class and township
        that is then stiched into complete column by pandas
    """
    return zscore(s, nan_policy='omit')


def outlier_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs np.select that creates an outlier taxonomy.
    Inputs:
        df (pd.DataFrame): dataframe with necessary columns created from previous functions.
    Outputs:
        df (pd.DataFrame): dataframe with 'outlier_type' column.
    """
    conditions = [
    (df['short_owner'] == 'Short-term owner') & (df['pricing'].str.contains('High')),
    (df['name_match'] != 'No match') & (df['pricing'].str.contains('High')),
    (df['transaction_type'] == 'legal_entity-legal_entity') & (df['pricing'].str.contains('High')),
    (df['anomaly'] == 'Outlier') & (df['pricing'].str.contains('High')),
    (df['pricing'].str.contains('High price swing')),
    (df['pricing'].str.contains('High')) & (df['which_price'] == '(raw & sqft)'),
    (df['pricing'].str.contains('High')) & (df['which_price'] == '(raw)'),
    (df['pricing'].str.contains('High')) & (df['which_price'] == '(sqft)'),
    (df['short_owner'] == 'Short-term owner') & (df['pricing'].str.contains('Low')),
    (df['name_match'] != 'No match') & (df['pricing'].str.contains('Low')),
    (df['transaction_type'] == 'legal_entity-legal_entity') & (df['pricing'].str.contains('Low')),
    (df['anomaly'] == 'Outlier') & (df['pricing'].str.contains('Low')),
    (df['pricing'].str.contains('Low price swing')),
    (df['pricing'].str.contains('Low')) & (df['which_price'] == '(raw & sqft)'),
    (df['pricing'].str.contains('Low')) & (df['which_price'] == '(raw)'),
    (df['pricing'].str.contains('Low')) & (df['which_price'] == '(sqft)')]

    labels = ['Home flip sale (high)', 'Family sale (high)',
              'Non-person sale (high)', 'Anomaly (high)',
              'High price swing',
              'High price (raw & sqft)', 'High price (raw)',
              'High price (sqft)',
              'Home flip sale (low)', 'Family sale (low)',
              'Non-person sale (low)', 'Anomaly (low)',
              'Low price swing',
              'Low price (raw & sqft)', 'Low price (raw)',
              'Low price (sqft)']

    df["outlier_type"] = np.select(conditions, labels, default='Not outlier')

    return df


def outlier_description(row: pd.Series) -> str:
    """
    Creates an column is an easily interpretable summary of
    highly relevant information for the particular record.
    Depends on the outlier taxonomy.
    Meant for apply().
    Inputs:
        row(pd.Series): from apply
    Outputs:
        value (str): description for the outlier
    """
    likely_message_expression = "However, this name match occurs much more often than expected by random chance, and the parties may not be related."

    if '(raw & sqft)' in row['which_price']:
        price_expression = f"raw price of ${row['sale_price'] / 1000:,}K that is {round(row['price_deviation_class_township'], 1)}"\
                           f" deviations away from the township/class mean and a price per sqft of ${round(row['price_per_sqft'], 1):,} that is {round(row['price_per_sqft_deviation_class_township'], 1)}"\
                           f" deviations from township/class mean"
    if '(raw)' in row['which_price']:
        price_expression = f"raw price of ${row['sale_price'] / 1000:,}K that is {round(row['price_deviation_class_township'], 1)} deviations from township/class mean"
    if '(sqft)' in row['which_price']:
        price_expression = f"""price per sqft of ${round(row['price_per_sqft'], 1):,} that is {round(row['price_per_sqft_deviation_class_township'], 1)} deviations from township/class mean"""

    if 'Home flip sale' in row['outlier_type']:
        value = f"Likely home flip sale with {price_expression}"\
                f" The price changed from ${row['previous_price']/ 1000:,}K to ${row['sale_price'] / 1000:,}K"\
                f" and the previous owner owned the property for only {row['days_since_last_transaction']} days."
    elif 'Family sale' in row['outlier_type']:
        value = f"Likely family sale due to matching last name of: '{row['name_match']}'."\
                f" It has a {'high' if 'high' in row['outlier_type'] else 'low'} {price_expression}"
                #f" {likely_message_expression if row['match_likely'] == 1 else ''}"\
    elif 'Non-person' in row['outlier_type']:
        value = f"Both buyer and seller were identified as legal entities."\
                f" It is a {'high' if 'high' in row['outlier_type'] else 'low'} {price_expression}"
    elif 'High price swing' in row['outlier_type']:
        value = f"Transaction is both a compound growth rate outlier {round(row['pct_deviation_class_township'], 1)}"\
                f" deviations away from the mean as well as a high {price_expression}"
    elif 'Low price swing' in row['outlier_type']:
        value = f"Transaction is both a compound growth rate outlier"\
                f" {round(row['pct_deviation_class_township'], 1)} deviations away from the mean as well as a low {price_expression}"
    elif 'Anomaly' in row['outlier_type']:
        value = f"Detected by anomaly algorithm with"\
                f" {'high' if 'high' in row['outlier_type'] else 'Low'} {price_expression}"
    elif '(raw & sqft)' in row['outlier_type']:
        value = f"""{'High' if 'High' in row['outlier_type'] else 'Low'} {price_expression}
        """
    elif '(raw)' in row['outlier_type']:
        value = f"""{'High' if 'High' in row['outlier_type'] else 'Low'} {price_expression}
        """
    elif '(sqft)' in row['outlier_type']:
        value = f"""{'High' if 'High' in row['outlier_type'] else 'Low'} {price_expression}
        """
    else:
        value = 'Not outlier'

    return value


def outlier_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a flag that shows whether the record is an
    outlier (a special flag) according to our outlier taxonomy.
    Inputs:
        df (pd.DataFrame): dataframe to create outlier flag
    Outputs:
        df (pd.DataFrame): dataframe with 'is_outlier' column
    """

    df['is_outlier'] = np.select([(df['outlier_type'] == 'Not outlier')],
                                 ['Not outlier'], default= 'Outlier')

    return df


# STRING CLEANUP

"""
    An outline of our overall approach:

    Tries to create an identifier from the buyer/seller name.
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

    TODO: Process more string types:
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


entity_keywords = r"llc| ll$| l$|l l c|estate|training|construction|building|masonry|"\
                  r"apartments|plumbing|service|professional|roofing|advanced|office|"\
                  r"\blaw\b|loan|legal|production|woodwork|concepts|corp| company|"\
                  r" united|\binc\b|county|entertainment|community|heating|cooling"\
                  r"|partners|equity|indsutries|series|revitalization|collection|"\
                  r"agency|renovation|consulting|flippers|estates|\bthe \b|dept|"\
                  r"funding|opportunity|improvements|servicing|equities|sale|"\
                  r"judicial| in$|bank|\btrust\b|holding|investment|housing"\
                  r"|properties|limited|realty|development|capital|management"\
                  r"|developers|construction|rentals|group|investments|invest|"\
                  r"residences|enterprise|enterprises|ventures|remodeling|"\
                  r"specialists|homes|business|venture|restoration|renovations"\
                  r"|maintenance|ltd|real estate|builders|buyers|property|financial"\
                  r"|associates|consultants|international|acquisitions|credit|design"\
                  r"|homeownership|solutions|home|diversified|assets|family|land|"\
                  r"revocable|services|rehabbing|living|county of cook|fannie mae|"\
                  r"land|veteran|mortgage|savings|lp$"


def get_id(row: pd.Series, col: str) -> str:
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

    if any(x in words for x in ['fannie mae aka federal na',
                                'fannie mae a k a federal', 'federal national mortgage']):
        return 'fannie mae'

    if any(x in words for x in ['the judicial sales corpor', 'judicial sales corp',
                                'judicial sales corporatio', 'judicial sale corp', 'the judicial sales corp']):
        return 'the judicial sales corporation'

    if any(x in words for x in ['jpmorgan chase bank n a', 'jpmorgan chase bank nati']):
        return 'jp morgan chase bank'

    if any(x in words for x in ['wells fargo bank na',  'wells fargo bank n a',
                                'wells fargo bank nationa',  'wells fargo bank n a a']):
        return 'wells fargo bank national'

    if any(x in words for x in ['bayview loan servicing l',  'bayview loan servicing ll']):
        return 'bayview loan servicing llc'

    if any(x in words for x in ['thr property illinois l', 'thr property illinois lp']):
        return 'thr property illinois lp'

    if any(x in words for x in ['ih3 property illinois lp', 'ih3 property illinois l']):
        return 'ih3 property illinois lp'

    if any(x in words for x in ['ih2 property illinois lp', 'ih2 property illinois l']):
        return 'ih2 property illinois lp'

    if any(x in words for x in ['secretary of housing and',
                                'the secretary of housing', 'secretary of housing ']):
        return 'secretary of housing and urban development'

    if any(x in words for x in ['secretary of veterans aff', 'the secretary of veterans']):
        return 'secretary of veterans affairs'

    if any(x in words for x in ['bank of america n a',
                                'bank of america na', 'bank of america national',]):
        return 'bank of america national'

    if any(x in words for x in ['us bank national association', 'u s bank national assoc',
                                'u s bank national associ', 'u s bank trust n a as', 'u s bank n a',
                                'us bank national associat', 'u s bank trust national']):
        return 'us bank national association'

    words = re.sub('suc t$|as succ t$|successor tr$|successor tru$|'\
                   'successor trus$|successor trust$|successor truste$|'\
                   'successor trustee$|successor t$|as successor t$',
                   'as successor trustee', words)
    words = re.sub('as t$|as s t$|as sole t$|as tr$|as tru$|as trus$|as trust$|'\
                   'as truste$|as trustee$|as trustee o$|as trustee of$|trustee of$|'\
                   'trustee of$|tr$|tru$|trus$|truste$|trustee$|, t|, tr|, tru|, trus|'\
                   ', trust|, truste',
                   'as trustee', words)
    words = re.sub('su$|suc$|succ$|succe$|succes$|success$|successo$|successor$|as s$|as su$|'\
                   'as suc$|as succ$|as succe$|as sucess$|as successo$|, s$|, su$|, suc$|, succ$|'\
                   ', succe$|, succes$|, success$|, successo$',
                   'as successor', words)

    if re.search(entity_keywords, words) or re.search(r'\d{4}|\d{3}', words) or \
        re.search('as trustee$|as successor$|as successor trustee$', words):
        id = words
        return id

    words = re.sub(' in$|indi$|indiv$|indivi$|indivi$|individ$|individu$|individua$|individual$'\
                   '|not i$|not ind$| ind$| inde$|indep$|indepe$|indepen$|independ$|independe$'\
                   '|independen$|independent$',
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

    _and = re.search(r'\b and\b|\b an$\b|\b a$\b|f k a|\bfka\b| n k a|\bnka\b|'\
                     r'\b aka\b|a k a|\b kna\b|k n a| f k$|n k$|a k$|\b not\b| married', words)

    if _and:
        tokens = words.split(_and.group())
        tokens = tokens[0].strip().split()
    else:
        tokens = words.split()

    return tokens


def name_selector(tokens) -> str:
    """
    Attempts to select the last name of a persons name based on number of tokens.
    Inputs:
        tokens: name to be identified
    Outputs:
        'Empty Name' if name is empty.
        id (str): identified last name
    """
    if tokens == 'Empty Name':
        return tokens
    # Ex: John Smith Jr
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


def get_category(row: pd.Series, col: str) -> str:
    """
    Gets category buyer/seller id. legal_entity if in entity keywords,
    person if otherwise.
    Inputs:
        row: from pandas dataframe
        col (str): column to process. 'buyer' or 'seller'
    Outputs:
        category (str): category of buyer/seller id    
    """

    column = col + '_id'
    words = row[column]

    if re.search(entity_keywords, words):
        category = 'legal_entity'
    elif words == 'Empty Name':
        category = 'none'
    else:
        category = 'person'

    return category


def get_role(row: pd.Series, col: str) -> str:
    """
    Picks the role th person is playing in the transaction off of the
    buyer/seller_id. Meant for apply()
    Ex: 'as trustee', or 'as successor'
    Inputs:
        row: from pandas dataframe
        col (str): column to process. 'buyer' or 'seller'
    Outputs:
        roles(str): the role of the person n the transaction

    """
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


def clean_id(row: pd.Series, col: str) -> str:
    """
    Cleans id field after get_role() by removing role.
    Inputs:
        row: from padnas dataframe
        col (str): column to process. 'seller' or 'buyer'
    Outputs:
        words (str): seller/buyer id without role.
    """

    column = col + '_id'
    words = row[column]

    words = re.sub(r' as successor trustee|\b as successor\b| as trustee', '', words)
    words = re.sub(' as$| as $|as $','', words)

    if not (re.search(entity_keywords, words) or \
            re.search(r'\d{4}|\d{3}', words) or \
            len(words.split()) == 1):
        words = name_selector(split_logic(words))

    return words


def create_judicial_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a column that contains 1 if sold from a judicial corp
    and 0 otherwise. Mean for use with apply().
    Inputs:
        df (pd.DataFrame): dataframe to create flag on
    Outputs:
        df (pd.DataFrame): dataframe with 'is_judicial_sale' column
    """

    df['is_judicial_sale'] = np.select([(df['seller_id'] == 'the judicial sale corporation') | (df['seller_id'] == 'intercounty judicial sale')],
                                        ['1'], default = '0' )

    return df


def create_match_flag(row: pd.Series) -> str:
    """
    Creates a column that says whether the buyer/seller id match.
    Meant for apply().
    Inputs:
        row: from dataframe
    Outputs:
        value (str): whether the buyer and seller ID match
    """
    if row['buyer_id'] == row['seller_id'] and row['buyer_id'] != 'Empty Name':
        value = 'Buyer ID and Seller ID match'
    else:
        value = 'No match'

    return value


def create_name_match(row: pd.Series) -> str:
    """
    Creates a column that contains the actual string that was matched.
    Meant for apply().
    Inputs:
        row: from pandas dataframe
    Outputs:
        value (str or None): string match if applicable, None otherwise
    """
    if row['buyer_id'] == row['seller_id'] and row['buyer_id'] != 'Empty Name':
        value = row['seller_id']
    else:
        value = 'No match'

    return value


def calc_name_occurences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the expected number of occurences of a name matching with the same name
    and the actual number of occcurences of each of these name matches.
    Ex: probablity of 'garcia' matching with 'garcia' and
    the actual number of times garcia was matched
    Gives the is_match_likely apply() function required informtion.
    Inputs:
        df (pd.DataFrame): dataframe to calculate name probabilties for
    Outputs:
        df (pd.DataFrame): dataframe with 'actual_occurences' and
                           'expected_occurences' column
    """
    buyers = df.buyer_id.value_counts().reset_index().set_index('index')
    sellers = df.seller_id.value_counts().reset_index().set_index('index')

    both = buyers.join(sellers)
    both.fillna(0, inplace=True)
    both['total'] = both['buyer_id'] + both['seller_id']
    both.drop("Empty Name", axis=0, inplace=True)
    both['prob'] = both['total'] / both['total'].sum()
    both['prob_of_match'] = both['prob']**2
    both['expected_occurences'] = both['prob_of_match'] * both['total'].sum()
    both = both[['expected_occurences']]
    both = both.reset_index()
    both.rename(columns={'index':'name_match'}, inplace=True)

    occur = df.name_match.value_counts().reset_index().set_index('index')
    # drops the most frequently occuring: 'No match'
    occur.drop('No match', axis=0, inplace=True)
    occur = occur.reset_index()
    occur.rename(columns={'name_match':'actual_occurences', 'index':'name_match'}, inplace=True)

    df = df.merge(both, on='name_match', how='left')
    df = df.merge(occur, on='name_match', how='left')

    return df


def is_match_likely(row: pd.Series) -> int:
    """
    Creates a column that contains 1 if name match is expected 5 times as often as
    it occurs and 0 otherwise.

    Matches of uncommon last names have very low expected occurences, they may be the only
    record with that name. So doing 'actual_occurences > expected_occurences' goes off on just about
    every uncommon last name that is likely a real match.

    Conversely, matches of common last names have very high expected occurences.
    In addition, what last name someone has is heavily influenced by where they are in the city,
    and similar last name tend to culster together, therefore many name matches happen much
    more than you would expect by pure chance.

    So 'actual_occurences < expected_occurences' lets us see which names occur less
    frequently than predicted, signalling that there is some spatial dependence.
    The multiplier is essentially controlling for how much spatial dependence is acceptable.
    Meant for use with apply().

    Inputs:
        row: from pandas dataframe
    Outputs:
        value (int): 1 if match occurs more than expected by pure chance,
                     0 otherwise.
    """
    value = 0

    if row['name_match'] != 'No match':
        if row['actual_occurences'] < row['expected_occurences'] * 5:
            value = 1

    return value


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
    # df['buyer_role'] = df.apply(get_role, args=('buyer',), axis=1)
    # df['seller_role'] = df.apply(get_role, args=('seller',), axis=1)
    df['buyer_id'] = df.apply(clean_id, args=('buyer',), axis=1)
    df['seller_id'] = df.apply(clean_id, args=('seller',), axis=1)
    df['transaction_type'] = df['buyer_category'] + '-' + df['seller_category']

    df = create_judicial_flag(df)
    df['name_match'] = df.apply(create_name_match, axis=1)

    #df = calc_name_occurences(df)
    #df['match_likely'] = df.apply(is_match_likely, axis=1)

    return df


if __name__ == "__main__":

    def tuple_int(strings: str) -> tuple:
        """
        This is a function that takes the string input from --permut
        and turns it into an actual tuple of numbers.
        """
        strings = strings.replace("(", "").replace(")", "")
        mapped_int = map(int, strings.split(","))
        return tuple(mapped_int)

    def tuple_str(strings: str) -> tuple:
        """
        This is a function that takes the string input from --permut
        and turns it into an actual tuple of strings.
        """
        strings = strings.replace("(", "").replace(")", "")
        mapped_str = map(str, strings.split(","))
        return tuple(mapped_str)

    parser = argparse.ArgumentParser()
    parser.add_argument('--permut', dest='permut', required=True, type=tuple_int,
                        help= """The standard deviation permuatation to be run as a tuple sorrounded by quotes.
                                 Ex: '(2,2)' to run with 2 standard deviations on both sides.
                                 """)
    parser.add_argument('--groups', dest='groups', required=True, type=tuple_str,
                        help= """The columns to groupby when determining outliers.
                                 Ex: '('township_code, 'class')'
                                 """)
    parser.add_argument('--file', dest='file', required=True,
                        help= """The name of the output file as a string
                                 Ex: 'flagged.csv'
                                 """)

    args = parser.parse_args()

    go(tuple(args.permut), tuple(args.groups), args.file)
