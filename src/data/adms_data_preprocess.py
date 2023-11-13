import pandas as pd
import numpy as np
import sys

sys.path.append('.')
sys.path.append('..')
import missingno as msno
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split


def plot_tot_tob(TOT, TOB):
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))
    sns.histplot(TOT, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('TOT')
    # sns.histplot(TOT[TOT < np.quantile(TOT, 0.9)], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('TOT < 90th percentile')
    sns.histplot(TOB, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('TOB')
    # sns.histplot(TOB[TOB < np.quantile(TOB, 0.9)], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('TOB < 90th percentile')
    plt.tight_layout()
    plt.show()


def main():
    original_dataset = pd.read_pickle('../data/labeled_dataset.pkl')
    takeove_labeled = pd.read_csv('../data/takeover_labeled.csv')

    CLIP_THR = 0.995
    # 区分AD，TD
    TD = takeove_labeled[takeove_labeled['wti_content'] < 37]
    # AD = takeove_labeled[takeove_labeled['wti_content'] >= 37]
    TD_dataset = original_dataset[original_dataset['uuid'].isin(TD['uuid'])]
    TOT = TD_dataset['tot'] * 10**-9
    # TOB = TD_dataset['tob'] * 10**-9
    TD_dataset.loc[:, 'tot'] = TD_dataset['tot'] * 10**-9
    TD_dataset.loc[:, 'tob'] = TD_dataset['tob'] * 10**-9
    # plot_tot_tob(TD_dataset.loc[:, 'tot'], TD_dataset.loc[:, 'tob'])

    # 删除掉TOT and TOB < 0 的数据
    TD_dataset_positive = TD_dataset[TOT > 0]
    # plot_tot_tob(TD_dataset_positive.loc[:, 'tot'], TD_dataset_positive.loc[:, 'tob'])

    # index 取并集，取tot和tob99分位以内的数据
    cond = (TD_dataset_positive['tot'] < np.quantile(TD_dataset_positive['tot'], CLIP_THR)) & \
            (TD_dataset_positive['tob'] < np.quantile(TD_dataset_positive['tob'], CLIP_THR))
    TD_dataset_positive_99 = TD_dataset_positive[cond]
    # plot_tot_tob(TD_dataset_positive_99.loc[:, 'tot'], TD_dataset_positive_99.loc[:, 'tob'])

    # 去掉null值
    TD_dataset_positive_99_no_nan = TD_dataset_positive_99.dropna()
    # plot_tot_tob(TD_dataset_positive_99_no_nan.loc[:, 'tot'], TD_dataset_positive_99_no_nan.loc[:, 'tob'])

    # 去掉DANADSts < 6的数据
    TD_dataset_positive_99_no_nan = TD_dataset_positive_99_no_nan[
        TD_dataset_positive_99_no_nan['c_DANADSts'] >= 6]

    # 取对数, 使Label趋向于正态分布
    # plot_tot_tob(boxcox(TD_dataset_positive_99_no_nan.loc[:, 'tot'] + 1), boxcox(TD_dataset_positive_99_no_nan.loc[:, 'tob'] + 1))
    # plot_tot_tob(np.log(TD_dataset_positive_99['tot'] + 1), np.log(TD_dataset_positive_99['tob'] + 1))

    # 特征处理：离散值 [1,14)，连续值 [14, -2)
    # 离散值
    sparse_fea = TD_dataset_positive_99_no_nan.iloc[:, 3:16]
    unique_counts = sparse_fea.nunique()
    # 连续值 归一化
    dense_fea = TD_dataset_positive_99_no_nan.iloc[:, 16:-2]
    dense_fea_min = dense_fea.min()
    dense_fea_max = dense_fea.max()
    TD_dataset_positive_99_no_nan.iloc[:, 16:-2] = (
        dense_fea - dense_fea_min) / (dense_fea_max - dense_fea_min)

    # 数据集划分
    train_df, temp_df = train_test_split(TD_dataset_positive_99_no_nan,
                                         test_size=0.4,
                                         random_state=42)
    valid_df, test_df = train_test_split(temp_df,
                                         test_size=0.5,
                                         random_state=42)
    print("训练集大小：", train_df.shape)
    print("验证集大小：", valid_df.shape)
    print("测试集大小：", test_df.shape)

    # 保存中间结果
    min_max_df = pd.concat([dense_fea_min, dense_fea_max], axis=1)
    min_max_df.columns = ['min', 'max']
    min_max_df.to_pickle("../data/min_max_df.pkl")
    # train valid test 数据集保存
    train_df.to_pickle("../data/train.pkl")
    valid_df.to_pickle("../data/valid.pkl")
    test_df.to_pickle("../data/test.pkl")


if __name__ == "__main__":
    main()
