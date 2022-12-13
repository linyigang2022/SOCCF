from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from sklearn import preprocessing
import time


raw_file = "raw_data/Foursquare.dat"
raw_sep = '|'
filter_min = 5
processed_file_prefix = "processed_data/Foursquare_"
sample_num = 100
sample_pop = True

# ================================================================================
# obtain implicit feedbacks
df = pd.read_csv(raw_file, sep=raw_sep, header=None, names=['id','user_id','item_id','latitude','longitude','time'], 
                 skiprows=2, skipfooter=2)
df = df[['user_id','item_id','time']]

print("==== statistic of raw data ====")
print("#users: %d" % len(df.user_id.unique()))
print("#items: %d" % len(df.item_id.unique()))
print("#actions: %d" % len(df))

# ========================================================
# sort by time
df['timestamp'] = df['time'].apply(lambda x: time.mktime(time.strptime(x.strip(), "%Y-%m-%d %H:%M:%S")))
df.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
df = df[['user_id','item_id','timestamp']]

# ========================================================
# drop duplicated user-item pairs
df.drop_duplicates(subset=['user_id','item_id'], keep='first', inplace=True)

# ========================================================
# discard cold-start items
count_i = df.groupby('item_id').user_id.count()
item_keep = count_i[count_i >= filter_min].index
df = df[df['item_id'].isin(item_keep)]

# discard cold-start users
count_u = df.groupby('user_id').item_id.count()
user_keep = count_u[count_u >= filter_min].index
df = df[df['user_id'].isin(user_keep)]

# renumber user ids and item ids
le = preprocessing.LabelEncoder()
df['user_id'] = le.fit_transform(df['user_id']) + 1
list_u = df['user_id'].unique() - 1
list_u_ori = list(le.inverse_transform(list_u))
df_u = pd.DataFrame({'after': list_u + 1, 'before': list_u_ori})
df_u.to_csv(processed_file_prefix + "ucoding.csv", header=True, index=False)

df['item_id'] = le.fit_transform(df['item_id']) + 1
list_i = df['item_id'].unique() - 1
list_i_ori = list(le.inverse_transform(list_i))
df_i = pd.DataFrame({'after': list_i + 1, 'before': list_i_ori})
df_i.to_csv(processed_file_prefix + "icoding.csv", header=True, index=False)

# output statistical information
print("==== statistic of processed data (whole) ====")
n = len(df.user_id.unique())
m = len(df.item_id.unique())
p = len(df)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = df.groupby(['user_id']).item_id.count()
print("min #actions per user: %.2f" % count_u.min())
print("max #actions per user: %.2f" % count_u.max())
print("ave #actions per user: %.2f" % count_u.mean())

# ========================================================
# split data into test set, valid set and train set,
# adopting the leave-one-out evaluation for next-item recommendation task

# ========================================
# obtain possible records in test set
df_test = df.groupby(['user_id']).tail(1)

df.drop(df_test.index, axis='index', inplace=True)

# ========================================
# obtain possible records in valid set
df_valid = df.groupby(['user_id']).tail(1)

df.drop(df_valid.index, axis='index', inplace=True)

# ========================================
# drop cold-start items in valid set and test set
df_valid = df_valid[df_valid.item_id.isin(df.item_id)]
df_test = df_test[df_test.user_id.isin(df_valid.user_id) & (
    df_test.item_id.isin(df.item_id) | df_test.item_id.isin(df_valid.item_id))]

# output data file
df_valid.to_csv(processed_file_prefix + "valid.csv", header=False, index=False)
df_test.to_csv(processed_file_prefix + "test.csv", header=False, index=False)
df.to_csv(processed_file_prefix + "train.csv", header=False, index=False)

# output statistical information
print("==== statistic of processed data (split) ====")
print("#train_users: %d" % len(df.user_id.unique()))
print("#train_items: %d" % len(df.item_id.unique()))
print("#valid_users: %d" % len(df_valid.user_id.unique()))
print("#test_users: %d" % len(df_test.user_id.unique()))

# ========================================================
# For each user, randomly sample some negative items,
# and rank these items with the ground-truth item when testing or validation
df_concat = pd.concat([df, df_valid, df_test], axis='index')
sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})

# ========================================
# sample according to popularity
if sample_pop == True:
    sr_item2pop = df_concat.item_id.value_counts(sort=True, ascending=False)
    arr_item = sr_item2pop.index.values
    arr_pop = sr_item2pop.values

    def get_negative_sample(pos):
        neg_idx = ~np.in1d(arr_item, pos)
        neg_item = arr_item[neg_idx]
        neg_pop = arr_pop[neg_idx]
        neg_pop = neg_pop / neg_pop.sum()

        return np.random.choice(neg_item, size=sample_num, replace=False, p=neg_pop)

    arr_sample = df_negative.user_id.apply(
        lambda x: get_negative_sample(sr_user2items[x])).values

# ========================================
# sample uniformly
else:
    arr_item = df_concat.item_id.unique()
    arr_sample = df_negative.user_id.apply(
        lambda x: np.random.choice(
            arr_item[~np.in1d(arr_item, sr_user2items[x])], size=sample_num, replace=False)).values

# output negative data
df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis='columns')
df_negative.to_csv(processed_file_prefix + "negative.csv", header=False, index=False)
