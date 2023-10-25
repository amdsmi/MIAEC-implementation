import pandas as pd
import time
from tqdm import tqdm
import os


def MIAEC(dataset, treshhold=0.1):
    '''

    this is the function that gives the dataset including missing values
    and fill the them with most probable data according to the complete row of data

    input :

    dataset   ====> dataset including missing values
    threshold ====> if the parentage of complete rows lees then threshold
                    we use the filling dataset and use it as possible value

    output :

    data=====> same dataset with fill null values


    '''

    na_dataset = dataset.copy()

    # poss_val is the complete dataset that will use for finding most confident data row for filing the missing value in the row
    poss_val = dataset.dropna()

    # if the percentage of possible value less then the threshold:
    # first we fill the categorical with mode and the numerical by median
    # and than use the complete dataset as possible value
    if len(poss_val) / len(dataset) < treshhold:

        poss_val = pd.DataFrame()

        for k in dataset.keys():
            if dataset[k].dtypes == 'object':
                poss_val[k] = dataset[k].fillna(dataset[k].mode()[0])
            else:
                poss_val[k] = dataset[k].fillna(dataset[k].median())

    # all rows including value that fill the mising in for loop
    evid_chain = dataset[dataset.isnull().any(axis=1)]

    # replace the nan data with missng name
    evid = evid_chain.fillna('missing')

    # in order to locate the index of missing data row extract the null row index
    evide_loc = evid.index
    # loop through all evidence chain
    for i in tqdm(evide_loc):

        # define a mask for finding the data that non null in row
        not_nan = evid.loc[i] != 'missing'
        # defining a mask for finding the location of missing dara
        is_nan = evid.loc[i] == 'missing'

        # calculate the number of non missing value in incomplete row for that will use for the probability
        lenght = sum(not_nan)

        # all data columns except the ones that associate with a missing value
        poss = poss_val.loc[:, not_nan]

        # find the difference between a incomplete data row and complete row
        conf = poss == evid.loc[:, not_nan].loc[i]

        # create a column that its element is confidence for each row
        conf['conf_level'] = conf.sum(axis=1) / lenght

        # find all row with maximum confidence (it could be more then one row )
        max_conf_index = conf[conf['conf_level'] == conf['conf_level'].max()].index

        # separate the row or rows with maximum confidence
        poss_row = poss_val.loc[max_conf_index]

        # find the column in the possible value that should fill in the evidence chain number i
        imputation = poss_row.loc[:, is_nan]

        # for loop on each column
        # if the column is categorical or object fill with mode
        # and if the column is numerical fill with median
        for col in imputation.keys():
            if imputation[col].dtypes == 'object':
                dataset.loc[i, col] = imputation[col].mode()[0]
            else:
                dataset.loc[i, col] = imputation[col].median()

    return (na_dataset, dataset)


def error(complet_path, incomplet_path, treshhold=0.1):
    '''

    this is a function that compute the NORMS and AE
    for numerical and categorical data according to dataset
    complete dataset associated with incomplete datasets
    and report the promoters such as run time, number of categorical variable
    number of numerical variable

    input:

        complete_path   ======> path of original dataset
        incomplete_path ======> path of incomplete datasets

    output:

        info =====> a dataframe includes reports


    '''
    # read original dataset
    orginal = pd.read_excel(complet_path, header=None)

    # find the name of incomplete datasets
    _, _, files = next(os.walk(incomplet_path))

    info = {}

    # for loop on all incomplete datasets
    for i in range(len(files)):

        # read the incomplete dataset
        data = pd.read_excel(incomplet_path + '/' + files[i], header=None)

        start = time.time()

        # compute the imputed dataset
        null, impute = MIAEC(data, treshhold)

        end = time.time()

        # run time for one dataset
        duration = end - start

        # separate categorical and numerical columns for original dataset
        cat_org = orginal.loc[:, orginal.dtypes == 'object']
        num_org = orginal.loc[:, orginal.dtypes != 'object']

        # separate categorical and numerical columns for incomplete dataset
        cat_null = null.loc[:, null.dtypes == 'object']
        num_null = null.loc[:, null.dtypes != 'object']

        # separate categorical and numerical columns for imputed dataset
        cat_imp = impute.loc[:, impute.dtypes == 'object']
        num_imp = impute.loc[:, impute.dtypes != 'object']

        # define the number of category number column and number of sample
        num_samp = cat_org.shape[0]
        num_category = cat_org.shape[1]
        num_nummeric = num_org.shape[1]

        if num_nummeric != 0:

            # define the mask for null value and compute NRMS for numeric data
            NRMS_mask = num_null.isna()
            NRMS = ((num_imp[NRMS_mask] - num_org[NRMS_mask]) ** 2).sum().sum() / (
                        (num_org[NRMS_mask]) ** 2).sum().sum()
        else:
            NRMS = 'nan'

        if num_category != 0:

            # define the mask for null value and compute AE for categorical data
            AE_mask = cat_null.isna()
            AE = (cat_imp[AE_mask] == cat_org[AE_mask]).sum().sum() / null.isna().sum().sum()
        else:
            AE = 'nan'

        single_info = {
            'run_time': duration,
            'row': num_samp,
            'category_number': num_category,
            'AE': AE,
            'numeric_number': num_nummeric,
            'NRMS': NRMS
        }

        name = files[i][:-5]
        info[f'{name}'] = single_info

    return pd.DataFrame(info).T


com = '/content/drive/MyDrive/orginal/4-gauss.xlsx'
incom = '/content/drive/MyDrive/faghihi_data/4-gauss'

information = error(com, incom)

information.to_csv('4-gauss.csv')