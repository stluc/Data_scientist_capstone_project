import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def replace_values_in_df(df, lookup_list, value):
    """
    Replace the values columns corresponding to the list of columns into the dataframe.
    Trailing * is a key character for multiple columns. value is the value to be replaced with NaN.
    """
    list_of_cols = []
    for element in lookup_list:
        # Find all columns starting with word if keyword ends with *
        if element[-1] == '*':
            lookup_in_df = [
                col for col in df.columns if col.startswith(element[:-1])]
            list_of_cols.extend(lookup_in_df)
        else:
            list_of_cols.append(element)
    list_of_cols.sort()

    for col in list_of_cols:
        if col in df.columns:
            df[col] = df[col].replace(value, np.nan)


def get_values_distribution(df, values=None):
    """
    Returns an array of the percentage of input values contained in each column of the dataframe.
    Input:
        df - Dataframe
        values - (list) of values to be counted inside each column. Default NaN.
    Output:
        values_perc - (Array) percentages of input values contained in each column
    """

    if values is None:
        values = [np.nan]
    values_perc = np.zeros(len(df.columns))
    for i in range(len(df.columns)):
        for value in values:
            if np.isnan(value):
                values_perc[i] += df.iloc[:, i].isna().sum() / len(df) * 100
            else:
                values_perc[i] += (df.iloc[:, i] == value).sum() / len(df) * 100

    return values_perc


def columns_over_threshold(df, threshold=80, values=None, verbose=True):
    """
    Returns the list of columns containing NaN values [%] >= threshold.
    Input:
        df - Dataframe
        threshold - (numerical) Lower threshold percentage. Default 80%.
        values - (list) of values to be counted inside each column. Default NaN.
    Output:
        cols_over_threshold - (List) of columns >= threshold.
    """
    if values is None:
        values = [np.nan]
    df_percentages = get_values_distribution(df, values)
    cols_over_threshold = []
    for i, col in enumerate(df.columns):
        if df_percentages[i] >= threshold:
            cols_over_threshold.append(col)
            if verbose:
                print('{}: {:0.2f}%'.format(col, df_percentages[i]))
    print(f'{len(cols_over_threshold)} columns contain more than {threshold}% of {values}.\n')

    return cols_over_threshold


def impute_dataframe(df, imputer, numerical_cols=None):
    """
    Return a copy of the DataFrame imputed by means of the input imputer.
    Column order is not preserved, object columns are converted into dummy columns and appended.
    """
    if numerical_cols is None:
        numerical_cols = []
    if numerical_cols:
        df_imputed = df[numerical_cols]
    else:
        df_imputed = df.select_dtypes('number')
    df_imputed[:] = imputer.transform(df_imputed)
    try:
        df_categoricals = pd.get_dummies(df.select_dtypes('object'), drop_first=True)
        df_imputed = pd.concat([df_imputed, df_categoricals], axis=1)
    except:
        pass

    return df_imputed


def ordinal_encode_df(df):
    """ Transform the object categories by means of ordinal encoding"""
    ordinal_enc = OrdinalEncoder().fit(df.select_dtypes('object').replace(np.nan, 'nan'))
    object_ordinals = pd.DataFrame(ordinal_enc.transform(df.select_dtypes('object').replace(np.nan, 'nan'))).astype(
        'int')
    # we put back in the nan values
    nan_list = ordinal_enc.transform(np.array([['nan', 'nan', 'nan']]))
    for i in range(3):
        object_ordinals.iloc[:, i] = object_ordinals.iloc[:, i].replace(nan_list[0, i], np.nan)
    for i, col in enumerate(df.select_dtypes('object').columns):
        df[col] = object_ordinals.iloc[:, i]


def get_scaled_df(df):
    # generate list of columns not to be scaled
    # get columns which were turned into dummies i.e 0-1
    one_hot_list = [col for col in df.columns if df[col].max() <= 1]

    scaler = StandardScaler(copy=False)
    df_scaled = pd.DataFrame(scaler.fit_transform(df[df.columns.difference(one_hot_list)]))
    df_scaled.columns = df[df.columns.difference(one_hot_list)].columns
    df_scaled = pd.concat([df_scaled, df[one_hot_list]], axis=1)

    return df_scaled


def clean_df(df, imputer, cols_to_drop=None):
    # drop unnecessary
    if cols_to_drop is None:
        cols_to_drop = []
    df_clean = df.drop(["LNR", "EINGEFUEGT_AM"], axis=1, errors='ignore')
    
    # replace and remove unknown values
    try:
        df_clean['CAMEO_DEUG_2015'] = df_clean['CAMEO_DEUG_2015'].replace('X', -1).astype('float16')
    except:
        pass
    try:
        df_clean['CAMEO_INTL_2015'] = df_clean['CAMEO_INTL_2015'].replace('XX', -1).astype('float16')
    except:
        pass
        
    # reduce the memory usage
    df_clean = df_clean.astype('float16', errors='ignore')

    # replace with NaNs the unknown values
    set_0 = 'AGER_TYP, ALTERSKATEGORIE_GROB, ALTER_HH, ANREDE_KZ, CJT_GESAMTTYP, GEBAEUDETYP, GEOSCORE_KLS7, ' \
            'HAUSHALTSSTRUKTUR, HH_EINKOMMEN_SCORE, KBA05_*, KKK, NATIONALITAET_KZ, PRAEGENDE_JUGENDJAHRE, REGIOTYP, ' \
            'RETOURTYP_BK_S, TITEL_KZ, WOHNDAUER_2008, WACHSTUMSGEBIET_NB, W_KEIT_KIND_HH'.split(', ')
    set_neg1 = 'AGER_TYP, ALTERSKATEGORIE_GROB, ANREDE_KZ, BALLRAUM, BIP_FLAG, CAMEO_DEUG_2015, CAMEO_INTL_2015, ' \
               'D19_KK_KUNDENTYP, EWDICHTE, FINANZTYP, FINANZ_*, GEBAEUDETYP, GEOSCORE_KLS7, HAUSHALTSSTRUKTUR, ' \
               'HEALTH_TYP, HH_EINKOMMEN_SCORE, INNENSTADT, KBA05_*, KBA13_*, KKK, NATIONALITAET_KZ, ORTSGR_KLS9, ' \
               'OST_WEST_KZ, PLZ8_*, PRAEGENDE_JUGENDJAHRE, REGIOTYP, SEMIO_*, SHOPPER_TYP, SOHO_KZ, TITEL_KZ, ' \
               'VERS_TYP, WOHNDAUER_2008, WOHNLAGE, WACHSTUMSGEBIET_NB, W_KEIT_KIND_HH, ZABEOTYP'.split(', ')
    set_9 = 'KBA05_*, SEMIO_*, ZABEOTYP'.split(', ')
    replace_values_in_df(df_clean, set_0, 0)
    replace_values_in_df(df_clean, set_neg1, -1)
    replace_values_in_df(df_clean, set_9, 9)

    # Ordinal encode categorical features
    ordinal_encode_df(df_clean)

    # Remove the passed in columns
    df_clean.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

    # Replace missing values
    df_clean = impute_dataframe(df_clean, imputer)
    df_clean[df_clean < 0] = 0
    df_clean = df_clean.round().apply(pd.to_numeric, downcast='unsigned', errors='ignore')

    # Select the categorical features for dummy creation
    D19_categories = [col for col in df_clean.columns if (
            col.startswith('D19')
            and not (col.endswith('ANZ_12'))
            and not (col.endswith('ANZ_24'))
            and not (col.endswith('DATUM'))
            and not (col.endswith('QUOTE_12'))
    )]
    cat_features = 'AGER_TYP, ALTERSKATEGORIE_GROB, ANREDE_KZ, CAMEO_DEUG_2015, CAMEO_DEU_2015, CAMEO_INTL_2015, ' \
                   'CJT_GESAMTTYP, KK_KUNDENTYP, FINANZTYP, GEBAEUDETYP, GREEN_AVANTGARDE, HEALTH_TYP, ' \
                   'KBA05_HERSTTEMP, KBA05_MAXHERST, KBA05_MAXVORB, KBA05_MODTEMP, LP_FAMILIE_FEIN, LP_FAMILIE_GROB, ' \
                   'LP_LEBENSPHASE_FEIN, LP_LEBENSPHASE_GROB, LP_STATUS_FEIN, LP_STATUS_GROB, NATIONALITAET_KZ, ' \
                   'OST_WEST_KZ, PRAEGENDE_JUGENDJAHRE, REGIOTYP, RETOURTYP_BK_S, SHOPPER_TYP, SOHO_KZ, TITEL_KZ, ' \
                   'VERS_TYP, WOHNLAGE, ZABEOTYP, UNGLEICHENN_FLAG, DSL_FLAG, HH_DELTA_FLAG, KBA05_SEG6, ' \
                   'KONSUMZELLE'.split(', ')
    for cat in D19_categories:
        cat_features.append(cat)
    cat_features = [cat for cat in cat_features if cat in df_clean.columns]
    
    # Get dummies for selected categorical features
    df_clean = pd.get_dummies(df_clean, columns=cat_features, drop_first=True)
    
    # Reduce memory usage
    df_clean = df_clean.apply(pd.to_numeric, downcast='unsigned', errors='ignore')

    return df_clean
