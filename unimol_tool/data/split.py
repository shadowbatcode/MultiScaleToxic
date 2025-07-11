# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.model_selection import (
    GroupKFold, 
    KFold, 
    StratifiedKFold,
)
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import linkage, fcluster
from ..utils import logger


class Splitter(object):
    """
    The Splitter class is responsible for splitting a dataset into train and test sets 
    based on the specified method.
    """
    def __init__(self, method='random', kfold=5, seed=42,cluster_threshold=0.5,min_domain_length=5, **params):
        """
        Initializes the Splitter with a specified split method and random seed.

        :param split_method: (str) The method for splitting the dataset, in the format 'Nfold_method'. 
                             Defaults to '5fold_random'.
        :param seed: (int) Random seed for reproducibility in random splitting. Defaults to 42.
        """
        self.method = method
        self.n_splits = kfold
        self.seed = seed
        self.cluster_threshold = cluster_threshold
        self.min_domain_length = min_domain_length
        self.splitter = self._init_split()
    def _init_split(self):
        """
        Initializes the actual splitter object based on the specified method.

        :return: The initialized splitter object.
        :raises ValueError: If an unknown splitting method is specified.
        """
        if self.n_splits == 1:
            return None
        if self.method == 'random':
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.method == 'scaffold' or self.method == 'group':
            splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == 'stratified':
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.method == 'select':
            splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == 'hybrid':
            splitter = GroupKFold(n_splits=self.n_splits)
        else:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))

        return splitter

    def split(self, smiles, target=None, group=None, scaffolds=None, **params):
        """
        Splits the dataset into train and test sets based on the initialized method.

        :param data: The dataset to be split.
        :param target: (optional) Target labels for stratified splitting. Defaults to None.
        :param group: (optional) Group labels for group-based splitting. Defaults to None.

        :return: An iterator yielding train and test set indices for each fold.
        :raises ValueError: If the splitter method does not support the provided parameters.
        """
        if self.n_splits == 1:
            logger.warning('Only one fold is used for training, no splitting is performed.')
            return [(np.arange(len(smiles)), ())]
        if smiles is None and 'atoms' in params:
            smiles = params['atoms']
            logger.warning('Atoms are used as SMILES for splitting.')
        if self.method in ['random']:
            self.skf = self.splitter.split(smiles)
        elif self.method in ['scaffold']:
            self.skf = self.splitter.split(smiles, target, scaffolds)
        elif self.method in ['group']:
            self.skf = self.splitter.split(smiles, target, group)
        elif self.method in ['stratified']:
            self.skf = self.splitter.split(smiles, group)
        elif self.method == 'hybrid':
            return self._hybrid_split(**params)
        elif self.method in ['select']:
            unique_groups = np.unique(group)
            if len(unique_groups) == self.n_splits:
                split_folds = []
                for unique_group in unique_groups:
                    train_idx = np.where(group != unique_group)[0]
                    test_idx = np.where(group == unique_group)[0]
                    split_folds.append((train_idx, test_idx))
                self.split_folds = split_folds
                return self.split_folds
            else:
                logger.error('The number of unique groups is not equal to the number of splits.')
                exit(1)
        else:
            logger.error('Unknown splitter method: {}'.format(self.method))
            exit(1)
        self.split_folds = list(self.skf)
        return self.split_folds
    def _hybrid_split(self, **params):
        """原子维度混合分割的改进实现"""
        atom_coords = params['coordinates']
        if atom_coords is None:
            raise ValueError("需要提供原子坐标数据进行混合分割")

        # 结构聚类生成分组标签
        groups = self._cluster_structures(atom_coords)
        unique_groups = np.unique(groups)

        # 确保分组数与fold数一致
        if len(unique_groups) < self.n_splits:
            raise ValueError(f"聚类分组数({len(unique_groups)})小于分割数({self.n_splits})")

        # 构建与select方法相同的输出结构
        split_folds = []
        for fold_id in range(self.n_splits):
            # 使用模运算确保均匀分配
            test_mask = (groups % self.n_splits) == fold_id
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            # 应用最小长度过滤
            if len(test_idx) < self.min_domain_length:
                continue
                
            split_folds.append((train_idx, test_idx))

        # 平衡fold数量
        if len(split_folds) != self.n_splits:
            self.n_splits = len(split_folds)
            logger.warning(f"实际分割数调整为{self.n_splits}以适应有效分组")

        self.split_folds = split_folds
        return self.split_folds
    def _cluster_structures(self, atom_coords):
        """改进的原子结构聚类"""
        from sklearn.cluster import KMeans
        
        # 提取增强的几何特征
        features = self._extract_geometric_features(atom_coords)
        
        # 使用K-means确保分组数可控
        cluster = KMeans(n_clusters=self.n_splits, 
                        n_init=10,
                        random_state=self.seed)
        return cluster.fit_predict(features)

    def _extract_geometric_features(self, coords):
        """增强的特征提取方法"""
        features = []
        for c in coords:
            # 计算中心距
            centroid = np.mean(c, axis=0)
            dists = np.linalg.norm(c - centroid, axis=1)
            
            # 添加统计特征
            features.append([
                np.mean(dists),    # 平均径向距离
                np.std(dists),     # 距离离散度
                np.max(dists),     # 最大延伸距离
                self._calc_gyration_radius(c),  # 回转半径
                self._calc_surface_roughness(c) # 表面粗糙度
            ])
        return np.array(features)

    def _calc_gyration_radius(self, coord):
        """计算回转半径"""
        centroid = np.mean(coord, axis=0)
        return np.sqrt(np.mean(np.sum((coord - centroid)**2, axis=1)))

    def _calc_surface_roughness(self, coord):
        """表面粗糙度指标"""
        hull = ConvexHull(coord)
        return hull.area / (hull.volume + 1e-6)