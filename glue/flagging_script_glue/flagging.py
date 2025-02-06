"""
This file contains all necessary functions to create a DataFrame ready to use for
non-arms length transaction detection using statistical and heuristic methods.
"""

import re
import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Constants
SHORT_TERM_OWNER_THRESHOLD = 365  # days

# Compile entity keywords regex for performance
ENTITY_KEYWORDS = re.compile(
    r"llc| ll$| l$|l l c|estate|training|construction|building|masonry|"
    r"apartments|plumbing|service|professional|roofing|advanced|office|"
    r"\blaw\b|\bloan\b|legal|production|woodwork|concepts|corp|company|"
    r" united|\binc\b|county|entertainment|community|heating|cooling"
    r"|partners|equity|indsutries|series|revitalization|collection|"
    r"agency|renovation|consulting|flippers|estates|\bthe \b|dept|"
    r"funding|opportunity|improvements|servicing|equities|\bsale\b|"
    r"judicial| in$|bank|\btrust\b|holding|investment|housing"
    r"|properties|limited|realty|development|capital|management"
    r"|developers|construction|rentals|group|investments|invest|"
    r"residences|enterprise|enterprises|ventures|remodeling|"
    r"specialists|homes|business|venture|restoration|renovations"
    r"|maintenance|ltd|real estate|builders|buyers|property|financial"
    r"|associates|consultants|international|acquisitions|credit|design"
    r"|homeownership|solutions|\bhome\b|diversified|assets|family|\bland\b"
    r"|revocable|services|rehabbing|\bliving\b|county of cook|fannie mae"
    r"|veteran|mortgage|savings|lp$|federal natl|hospital|southport|mtg"
    r"|propert|rehab|neighborhood|advantage|chicago|cook c|\bbk\b|\bhud\b"
    r"|department|united states|\busa\b|hsbc|midwest|residential|american"
    r"|tcf|advantage|real e|advantage|fifth third|baptist church"
    r"|apostolic church|lutheran church|catholic church|\bfed\b|nationstar"
    r"|advantage|commercial|health|condominium|nationa|association|homeowner"
    r"|christ church|christian church|baptist church|community church"
    r"|church of c|\bdelaw\b|lawyer|delawar",
    re.IGNORECASE,
)


# =============================================================================
# Utility Functions
# =============================================================================
def create_group_string(groups: tuple, sep: str = "_") -> str:
    """Joins group names with a separator to create a string for column naming."""
    return sep.join(groups)


def log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Applies base-10 log transformation to the specified columns."""
    for col in columns:
        df[col] = np.log10(df[col])
    return df


def z_normalize_groupby(s: pd.Series) -> pd.Series:
    """Returns the z-score normalization for a series (used with groupby.apply)."""
    return zscore(s, nan_policy="omit")


def between_two_numbers(num: float, a: float, b: float) -> bool:
    """Checks if num is strictly between a and b."""
    return a < num < b


# =============================================================================
# Statistical & Pricing Functions
# =============================================================================
def grouping_mean(df: pd.DataFrame, groups: tuple, condos: bool) -> pd.DataFrame:
    """Computes the mean sale price (and price per sqft for non-condos) using transform."""
    group_str = create_group_string(groups)
    df[f"sv_mean_price_{group_str}"] = df.groupby(list(groups))[
        "meta_sale_price"
    ].transform("mean")
    if not condos:
        df[f"sv_mean_price_per_sqft_{group_str}"] = df.groupby(list(groups))[
            "sv_price_per_sqft"
        ].transform("mean")
    return df


def deviation_dollars(df: pd.DataFrame, groups: tuple) -> pd.DataFrame:
    """Calculates deviations (in dollars) from group means."""
    group_str = create_group_string(groups)
    df[f"sv_deviation_{group_str}_mean_price"] = (
        df["meta_sale_price"] - df[f"sv_mean_price_{group_str}"]
    )
    if f"sv_mean_price_per_sqft_{group_str}" in df.columns:
        df[f"sv_deviation_{group_str}_mean_price_per_sqft"] = (
            df["sv_price_per_sqft"] - df[f"sv_mean_price_per_sqft_{group_str}"]
        )
    return df


def price_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the price per square foot."""
    df["sv_price_per_sqft"] = df["meta_sale_price"] / df["char_bldg_sf"]
    df["sv_price_per_sqft"].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def transaction_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the days elapsed since the last transaction.
    Assumes that 'meta_sale_date' is datetime.
    """
    mask = df["original_observation"] == True
    df.loc[mask, "sv_days_since_last_transaction"] = (
        df.sort_values("meta_sale_date")
        .groupby("pin")["meta_sale_date"]
        .transform(lambda x: x.diff().dt.days)
    )
    return df


def percent_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the compound growth rate (CGR) using the previous sale price and
    the days between transactions. Only applied to original observations.
    """
    mask = df["original_observation"] == True
    sorted_df = df.sort_values("meta_sale_date")
    df.loc[mask, "sv_previous_price"] = sorted_df.groupby("pin")[
        "meta_sale_price"
    ].transform(lambda x: x.shift())
    with np.errstate(divide="ignore", invalid="ignore"):
        df.loc[mask, "sv_cgdr"] = (
            df.loc[mask, "meta_sale_price"] / df.loc[mask, "sv_previous_price"]
        ) ** (1 / df.loc[mask, "sv_days_since_last_transaction"]) - 1
    return df


def dup_stats(df: pd.DataFrame, groups: tuple) -> pd.DataFrame:
    """
    For properties with multiple transactions, calculates duplicate sale counts and
    the direction of price movement relative to the group mean.
    """
    group_str = create_group_string(groups)
    dup_mask = df.duplicated("pin", keep=False)
    df.loc[dup_mask, "sv_sale_dup_counts"] = df.groupby("pin")["pin"].transform("count")
    dev_col = f"sv_deviation_{group_str}_mean_price_abs"
    df.loc[dup_mask, dev_col] = abs(
        df.loc[dup_mask, f"sv_mean_price_{group_str}"]
        - df.loc[dup_mask, "meta_sale_price"]
    )
    df.loc[dup_mask, "sv_price_movement"] = (
        df.loc[dup_mask]
        .sort_values("meta_sale_date")
        .groupby("pin")[dev_col]
        .transform(
            lambda s: s.lt(s.shift())
            .map({True: "Towards mean", False: "Away from mean"})
            .fillna("First sale")
        )
    )
    return df


def create_stats(df: pd.DataFrame, groups: tuple, condos: bool) -> pd.DataFrame:
    """Runs all the statistical calculations on the DataFrame."""
    if not condos:
        df = price_sqft(df)
    df = grouping_mean(df, groups, condos)
    if not condos:
        df = deviation_dollars(df, groups)
    df = dup_stats(df, groups)
    df = transaction_days(df)
    df = percent_change(df)
    return df


def pricing_info(
    df: pd.DataFrame, permut: tuple, groups: tuple, condos: bool
) -> pd.DataFrame:
    """
    Adds pricing deviation information (z-scores) and computes per-row lower/upper thresholds
    for each deviation measure using vectorized operations.
    Then applies functions to determine the pricing outlier type.
    """
    group_str = create_group_string(groups)
    cols_to_log = ["meta_sale_price"] + ([] if condos else ["sv_price_per_sqft"])
    df = log_transform(df, cols_to_log)

    # (Optional) Persist group-level statistics
    df["group_mean"] = df.groupby(list(groups))["meta_sale_price"].transform("mean")
    df["group_std"] = df.groupby(list(groups))["meta_sale_price"].transform("std")
    if not condos:
        df["group_sqft_mean"] = df.groupby(list(groups))["sv_price_per_sqft"].transform(
            "mean"
        )
        df["group_sqft_std"] = df.groupby(list(groups))["sv_price_per_sqft"].transform(
            "std"
        )

    # Compute deviation columns using z-normalization within groups
    df[f"sv_price_deviation_{group_str}"] = df.groupby(list(groups))[
        "meta_sale_price"
    ].apply(z_normalize_groupby)
    if not condos:
        df[f"sv_price_per_sqft_deviation_{group_str}"] = df.groupby(list(groups))[
            "sv_price_per_sqft"
        ].apply(z_normalize_groupby)
    df[f"sv_cgdr_deviation_{group_str}"] = df.groupby(list(groups))["sv_cgdr"].apply(
        z_normalize_groupby
    )

    # Compute lower and upper thresholds (per row) for each deviation column
    for col in [f"sv_price_deviation_{group_str}", f"sv_cgdr_deviation_{group_str}"]:
        df[f"{col}_lower"] = df.groupby(list(groups))[col].transform("mean") - permut[
            0
        ] * df.groupby(list(groups))[col].transform("std")
        df[f"{col}_upper"] = df.groupby(list(groups))[col].transform("mean") + permut[
            1
        ] * df.groupby(list(groups))[col].transform("std")
    if not condos:
        col = f"sv_price_per_sqft_deviation_{group_str}"
        df[f"{col}_lower"] = df.groupby(list(groups))[col].transform("mean") - permut[
            0
        ] * df.groupby(list(groups))[col].transform("std")
        df[f"{col}_upper"] = df.groupby(list(groups))[col].transform("mean") + permut[
            1
        ] * df.groupby(list(groups))[col].transform("std")

    # Apply outlier type functions that use the computed threshold columns
    df["sv_pricing"] = df.apply(lambda row: price_column(row, groups, condos), axis=1)
    if not condos:
        df["sv_which_price"] = df.apply(lambda row: which_price(row, groups), axis=1)
    return df


def which_price(row: pd.Series, groups: tuple) -> str:
    """
    Determines which price measure (raw, per sqft, or both) is flagged as an outlier
    by comparing the deviation values with their per-row thresholds.
    """
    group_str = create_group_string(groups)
    raw_val = row[f"sv_price_deviation_{group_str}"]
    raw_lower = row[f"sv_price_deviation_{group_str}_lower"]
    raw_upper = row[f"sv_price_deviation_{group_str}_upper"]
    raw_out = not between_two_numbers(raw_val, raw_lower, raw_upper)

    sqft_val = row.get(f"sv_price_per_sqft_deviation_{group_str}")
    if sqft_val is not None:
        sqft_lower = row[f"sv_price_per_sqft_deviation_{group_str}_lower"]
        sqft_upper = row[f"sv_price_per_sqft_deviation_{group_str}_upper"]
        sqft_out = not between_two_numbers(sqft_val, sqft_lower, sqft_upper)
    else:
        sqft_out = False

    if raw_out and not sqft_out:
        return "(raw)"
    elif not raw_out and sqft_out:
        return "(sqft)"
    elif raw_out and sqft_out:
        return "(raw & sqft)"
    else:
        return "Non-outlier"


def price_column(row: pd.Series, groups: tuple, condos: bool) -> str:
    """
    Determines whether the record is a high or low price outlier and, if applicable,
    whether it exhibits a price swing. Comparisons are made by checking the record's
    deviation against its per-row lower/upper threshold.
    """
    group_str = create_group_string(groups)
    value = "Not price outlier"
    price_flag = False
    raw_val = row[f"sv_price_deviation_{group_str}"]
    raw_lower = row[f"sv_price_deviation_{group_str}_lower"]
    raw_upper = row[f"sv_price_deviation_{group_str}_upper"]

    if condos:
        if raw_val > raw_upper:
            value = "High price"
            price_flag = True
        elif raw_val < raw_lower:
            value = "Low price"
            price_flag = True
        if price_flag and pd.notnull(row.get(f"sv_cgdr_deviation_{group_str}")):
            cgdr_val = row[f"sv_cgdr_deviation_{group_str}"]
            cgdr_lower = row[f"sv_cgdr_deviation_{group_str}_lower"]
            cgdr_upper = row[f"sv_cgdr_deviation_{group_str}_upper"]
            if row["sv_price_movement"] == "Away from mean" and not between_two_numbers(
                cgdr_val, cgdr_lower, cgdr_upper
            ):
                value += " swing"
    else:
        raw_out = raw_val > raw_upper or raw_val < raw_lower
        sqft_val = row[f"sv_price_per_sqft_deviation_{group_str}"]
        sqft_lower = row[f"sv_price_per_sqft_deviation_{group_str}_lower"]
        sqft_upper = row[f"sv_price_per_sqft_deviation_{group_str}_upper"]
        sqft_out = sqft_val > sqft_upper or sqft_val < sqft_lower

        if raw_out or sqft_out:
            if raw_out:
                value = "High price" if raw_val > raw_upper else "Low price"
            elif sqft_out:
                value = "High price" if sqft_val > sqft_upper else "Low price"
            price_flag = True

            if price_flag and pd.notnull(row.get(f"sv_cgdr_deviation_{group_str}")):
                cgdr_val = row[f"sv_cgdr_deviation_{group_str}"]
                cgdr_lower = row[f"sv_cgdr_deviation_{group_str}_lower"]
                cgdr_upper = row[f"sv_cgdr_deviation_{group_str}_upper"]
                if row[
                    "sv_price_movement"
                ] == "Away from mean" and not between_two_numbers(
                    cgdr_val, cgdr_lower, cgdr_upper
                ):
                    value += " swing"
    return value


def check_days(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Flags a transaction as a short-term ownership if the days since last transaction
    are below the given threshold.
    """
    df["sv_short_owner"] = np.where(
        df["sv_days_since_last_transaction"] < threshold,
        "Short-term owner",
        f"Over {threshold} days",
    )
    return df


def outlier_type(
    df: pd.DataFrame, condos: bool, raw_price_threshold: int
) -> pd.DataFrame:
    """
    Creates indicator columns for various outlier types based on both characteristic-
    and pricing-based conditions.
    """
    char_conditions = [
        df["sv_short_owner"] == "Short-term owner",
        df["sv_name_match"] != "No match",
        df[["sv_buyer_category", "sv_seller_category"]].eq("legal_entity").any(axis=1),
        df["sv_anomaly"] == "Outlier",
        df["sv_pricing"].str.contains("High price swing")
        | df["sv_pricing"].str.contains("Low price swing"),
    ]
    char_labels = [
        "sv_ind_char_short_term_owner",
        "sv_ind_char_family_sale",
        "sv_ind_char_non_person_sale",
        "sv_ind_char_statistical_anomaly",
        "sv_ind_char_price_swing_homeflip",
    ]

    if condos:
        price_conditions = [
            df["sv_pricing"].str.contains("High"),
            df["sv_pricing"].str.contains("Low"),
        ]
        price_labels = [
            "sv_ind_price_high_price",
            "sv_ind_price_low_price",
        ]
    else:
        price_conditions = [
            df["sv_pricing"].str.contains("High")
            & df["sv_which_price"].str.contains("raw"),
            df["sv_pricing"].str.contains("Low")
            & df["sv_which_price"].str.contains("raw"),
            df["sv_pricing"].str.contains("High")
            & df["sv_which_price"].str.contains("sqft"),
            df["sv_pricing"].str.contains("Low")
            & df["sv_which_price"].str.contains("sqft"),
        ]
        price_labels = [
            "sv_ind_price_high_price",
            "sv_ind_price_low_price",
            "sv_ind_price_high_price_sqft",
            "sv_ind_price_low_price_sqft",
        ]
    # Raw price threshold (comparing the unlogged value)
    price_conditions.append((10 ** df["meta_sale_price"]) > raw_price_threshold)
    price_labels.append("sv_ind_raw_price_threshold")

    for label, condition in zip(
        price_labels + char_labels, price_conditions + char_conditions
    ):
        df[label] = condition.astype(int)
    return df


# =============================================================================
# Isolation Forest & PCA Functions
# =============================================================================
def pca_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Runs PCA on the specified columns (after filling NAs and infinities)
    and returns the principal components with explained variance > 1.
    """
    feed = df[columns].fillna(0).replace([np.inf, -np.inf], 0)
    pca_model = PCA(n_components=len(feed.columns))
    pcs = pca_model.fit_transform(feed)
    pc_df = pd.DataFrame(
        pcs, columns=[f"PC{i}" for i in range(len(feed.columns))], index=df.index
    )
    n_components = sum(pca_model.explained_variance_ > 1)
    return pc_df.iloc[:, :n_components]


def iso_forest(
    df: pd.DataFrame,
    groups: tuple,
    columns: list,
    n_estimators: int = 1000,
    max_samples=0.2,
) -> pd.DataFrame:
    """
    Runs Isolation Forest on PCA-transformed features (with additional group labels)
    to flag statistical anomalies.
    """
    df.set_index("meta_sale_document_num", inplace=True)
    pca_features = pca_transform(df, columns)

    label_encoders = {}
    for group in groups:
        if not pd.api.types.is_numeric_dtype(df[group]):
            le = LabelEncoder()
            df[group] = le.fit_transform(df[group])
            label_encoders[group] = le
        pca_features[group] = df[group]

    iso = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=True,
        random_state=42,
    )
    df["sv_anomaly"] = iso.fit_predict(pca_features)
    df["sv_anomaly"] = np.where(df["sv_anomaly"] == -1, "Outlier", "Not Outlier")

    for group, le in label_encoders.items():
        df[group] = le.inverse_transform(df[group])
    df.reset_index(inplace=True)
    return df


# =============================================================================
# String Processing Functions
# =============================================================================
def get_id(row: pd.Series, col_prefix: str) -> str:
    """
    Generates an identifier from the buyer/seller name. If the name appears to be
    a legal entity (based on keywords, presence of digits, or certain suffixes),
    returns the cleaned string; otherwise attempts to extract a last name.
    """
    col = col_prefix + "_name"
    name_str = str(row[col]).lower().strip()
    if pd.isnull(name_str) or name_str in {
        "none",
        "nan",
        "unknown",
        "missing seller name",
        "missing buyer name",
    }:
        return "Empty Name"

    name_str = re.sub(r" amp ", " ", name_str)
    name_str = re.sub(r"\s+", " ", name_str).strip()
    if not name_str or re.fullmatch(r"[.]*", name_str):
        return "Empty Name"

    # Handle specific known cases
    special_cases = {
        "vt investment corpor": "vt investment corporation",
        "v t investment corp": "vt investment corporation",
        "national residential nomi": "national residential nominee services",
        "first integrity group inc": "first integrity group inc",
        "first integrity group in": "first integrity group inc",
        "deutsche bank national tr": "deutsche bank national trust company",
        "cirrus investment group l": "cirrus investment group",
        "cirrus investment group": "cirrus investment group",
        "fannie mae aka federal na": "fannie mae",
        "fannie mae a k a federal": "fannie mae",
        "federal national mortgage": "fannie mae",
        "judicial sales corpor": "the judicial sales corporation",
        "judicial sales corp": "the judicial sales corporation",
        "judicial sales corporatio": "the judicial sales corporation",
        "judicial sale corp": "the judicial sales corporation",
        "the judicial sales corp": "the judicial sales corporation",
        "jpmorgan chase bank n a": "jp morgan chase bank",
        "jpmorgan chase bank nati": "jp morgan chase bank",
        "wells fargo bank na": "wells fargo bank national",
        "wells fargo bank n a": "wells fargo bank national",
        "wells fargo bank nationa": "wells fargo bank national",
        "wells fargo bank n a a": "wells fargo bank national",
        "wells fargo bk": "wells fargo bank national",
        "bayview loan servicing l": "bayview loan servicing llc",
        "bayview loan servicing ll": "bayview loan servicing llc",
        "thr property illinois l": "thr property illinois lp",
        "thr property illinois lp": "thr property illinois lp",
        "ih3 property illinois lp": "ih3 property illinois lp",
        "ih3 property illinois l": "ih3 property illinois lp",
        "ih2 property illinois lp": "ih2 property illinois lp",
        "ih2 property illinois l": "ih2 property illinois lp",
        "secretary of housing and": "secretary of housing and urban development",
        "the secretary of housing": "secretary of housing and urban development",
        "secretary of housing ": "secretary of housing and urban development",
        "secretary of veterans aff": "secretary of veterans affairs",
        "the secretary of veterans": "secretary of veterans affairs",
        "bank of america n a": "bank of america national",
        "bank of america na": "bank of america national",
        "bank of america national": "bank of america national",
        "us bank national association": "us bank national association",
        "u s bank national assoc": "us bank national association",
        "u s bank national associ": "us bank national association",
        "u s bank trust n a as": "us bank national association",
        "u s bank n a": "us bank national association",
        "us bank national associat": "us bank national association",
        "u s bank trust national": "us bank national association",
        "us bk": "us bank national association",
        "u s bk": "us bank national association",
    }
    for key, val in special_cases.items():
        if key in name_str:
            return val

    # Normalize trustee/successor tokens
    name_str = re.sub(
        r"(suc t$|as succ t$|successor tr$|successor tru$|successor trus$|"
        r"successor trust$|successor truste$|successor trustee$|successor t$|as successor t$)",
        "as successor trustee",
        name_str,
    )
    name_str = re.sub(
        r"(as t$|as s t$|as sole t$|as tr$|as tru$|as trus$|as trust$|as truste$|"
        r"as trustee$|as trustee o$|as trustee of$|, t|, tr|, tru|, trus|, trust|, truste)",
        "as trustee",
        name_str,
    )
    name_str = re.sub(
        r"(su$|suc$|succ$|succe$|succes$|success$|successo$|successor$|as s$|as su$|"
        r"as suc$|as succ$|as succe$|as sucess$|as successo$|, s$|, su$|, suc$|, succ$|, succe$|, succes$|, success$|, successo$)",
        "as successor",
        name_str,
    )

    if (
        ENTITY_KEYWORDS.search(name_str)
        or re.search(r"\d{3,4}", name_str)
        or re.search(r"as trustee$|as successor$|as successor trustee$", name_str)
    ):
        return name_str

    name_str = re.sub(
        r"( in$|indi$|indiv$|indivi$|individ$|individu$|individua$|individual$|"
        r"not i$|not ind$| ind$| inde$|indep$|indepe$|indepen$|independ$|independe$|independen$|independent$)",
        "",
        name_str,
    )
    tokens = split_logic(name_str)
    return name_selector(tokens)


def split_logic(name_str: str):
    """
    Splits a cleaned string into tokens using keywords such as 'and' or common abbreviations.
    Returns a list of tokens (or "Empty Name" if input is not valid).
    """
    name_str = re.sub(r"\s+", " ", name_str).strip()
    if not name_str or re.fullmatch(r"[.]*", name_str) or name_str == "Empty Name":
        return "Empty Name"
    name_str = re.sub(r"\s+as$|\s+as\s+$|as\s+$", "", name_str)
    m = re.search(
        r"\b and\b|\b an\b|\b a\b|f k a|\bfka\b| n k a|\bnka\b|\b aka\b|a k a(?=\s|$)|\b kna\b|k n a| f k$|n k$|a k$|\b not\b| married",
        name_str,
    )
    if m:
        tokens = name_str.split(m.group())
        return tokens[0].strip().split()
    return name_str.split()


def name_selector(tokens) -> str:
    """
    Given a list of name tokens, returns the last token as an identifier,
    ignoring common suffixes.
    """
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    if tokens == "Empty Name" or not tokens:
        return "Empty Name"
    while tokens and tokens[-1] in suffixes:
        tokens = tokens[:-1]
    return tokens[-1] if tokens else "Empty Name"


def get_category(row: pd.Series, col_prefix: str) -> str:
    """
    Determines whether the identifier belongs to a legal entity or a person.
    """
    col = col_prefix + "_id"
    name_str = row[col]
    if ENTITY_KEYWORDS.search(name_str):
        return "legal_entity"
    elif name_str == "Empty Name":
        return "none"
    else:
        return "person"


def clean_id(row: pd.Series, col_prefix: str) -> str:
    """
    Cleans the identifier by removing role-related tokens and, if appropriate,
    reselecting the name token.
    """
    col = col_prefix + "_id"
    name_str = row[col]
    name_str = re.sub(
        r" as successor trustee|\bas successor\b| as trustee", "", name_str
    )
    name_str = re.sub(r"\s+as$|\s+as\s+$|as\s+$", "", name_str)
    if not (
        ENTITY_KEYWORDS.search(name_str)
        or re.search(r"\d{3,4}", name_str)
        or len(name_str.split()) == 1
    ):
        name_str = name_selector(split_logic(name_str))
    return name_str


def get_role(row: pd.Series, col_prefix: str) -> str:
    """
    Extracts the role (e.g., 'as trustee', 'as successor') from the identifier.
    """
    col = col_prefix + "_id"
    name_str = row[col]
    for role_token in [" as successor trustee", " as successor", " as trustee"]:
        m = re.search(role_token, name_str)
        if m:
            return m.group()
    return None


def create_judicial_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a binary flag (as string '1' or '0') indicating whether the seller's
    identifier corresponds to a judicial sales entity.
    """
    df["sv_is_judicial_sale"] = np.where(
        df["sv_seller_id"].isin(
            ["the judicial sale corporation", "intercounty judicial sale"]
        ),
        "1",
        "0",
    )
    return df


def create_name_match(row: pd.Series) -> str:
    """
    If the buyer and seller identifiers match (and are nontrivial and not both legal entities),
    returns the match; otherwise returns "No match".
    """
    if (
        row["sv_buyer_id"] == row["sv_seller_id"]
        and row["sv_buyer_id"] != "Empty Name"
        and row["sv_transaction_type"] != "legal_entity-legal_entity"
        and len(row["sv_buyer_id"]) > 1
    ):
        return row["sv_seller_id"]
    return "No match"


def string_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes buyer and seller name strings to generate identifiers, categories,
    roles, and transaction type.
    """
    for col in ["meta_sale_buyer_name", "meta_sale_seller_name"]:
        df[col] = (
            df[col]
            .str.encode("ascii", "ignore")
            .str.decode("ascii")
            .str.replace(r"[^a-zA-Z0-9\-]", " ", regex=True)
            .str.strip()
        )
    df["sv_buyer_id"] = df.apply(get_id, args=("meta_sale_buyer",), axis=1)
    df["sv_seller_id"] = df.apply(get_id, args=("meta_sale_seller",), axis=1)
    df["sv_buyer_category"] = df.apply(get_category, args=("sv_buyer",), axis=1)
    df["sv_seller_category"] = df.apply(get_category, args=("sv_seller",), axis=1)
    df["sv_buyer_id"] = df.apply(clean_id, args=("sv_buyer",), axis=1)
    df["sv_seller_id"] = df.apply(clean_id, args=("sv_seller",), axis=1)
    df["sv_transaction_type"] = df["sv_buyer_category"] + "-" + df["sv_seller_category"]
    df = create_judicial_flag(df)
    df["sv_name_match"] = df.apply(create_name_match, axis=1)
    return df


# =============================================================================
# Main Pipeline
# =============================================================================
def go(
    df: pd.DataFrame,
    groups: tuple,
    iso_forest_cols: list,
    dev_bounds: tuple,
    condos: bool,
    raw_price_threshold: int,
) -> pd.DataFrame:
    """
    Runs the entire processing pipeline:
      1. Statistical measures & outlier preparations.
      2. String processing.
      3. Isolation Forest anomaly detection.
      4. Outlier taxonomy assignment.
    """
    model_type = "condos" if condos else "residential"
    print(f"Flagging for {model_type}")
    print("Initializing statistics...")
    df = create_stats(df, groups, condos=condos)
    print("Statistics complete. Processing strings...")
    df = string_processing(df)
    print("String processing complete. Running isolation forest...")
    df = iso_forest(df, groups, iso_forest_cols)
    print("Isolation forest complete. Assigning outlier taxonomy...")
    df = check_days(df, SHORT_TERM_OWNER_THRESHOLD)
    df = pricing_info(df, dev_bounds, groups, condos=condos)
    df = outlier_type(df, condos=condos, raw_price_threshold=raw_price_threshold)
    print("Processing finished.")
    return df
