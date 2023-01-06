# LAB1 Learning-based Cardinality Estimation

1.	简介
查询优化器依赖于查询谓词的基数估计来产生一个好的执行计划。 当查询包含多个谓词时，当今的优化器会使用各种假设（例如谓词之间的独立性）来估计基数。 这些方法通常会产生较大的基数估计误差。 最近，提出了一些新的基于学习的模型来估计基数，例如深度神经网络、梯度提升树和和积网络。 基于方法，可以将它们分为两组： 查询驱动方法：它们将基数估计建模为回归问题，旨在通过特征向量在查询和估计结果之间建立映射。 数据驱动方法：他们将基数估计建模为联合概率分布估计问题，旨在从表数据构建联合分布，如图1.1。
本篇主要实现基于学习的基数估计的实验，一种是使用基于神经网络等回归模型的查询驱动方法，另一种是基于 SPN 的数据驱动方法。
 
图 1.1 两种基于学习的基数估计方法
2.	选用方法介绍
2.1	查询驱动
可以将基数估计视为回归问题。考虑t带有列的表c1, c2, ..., cn。对于一个查询select * from t where c1 > l1 and c1 < r1 and c1 > l2 and c2 < r2 and ... and cn > ln and cn < rn，想要预测在被谓词过滤后剩下多少行。从谓词中提取特征如下：[l1, r1, l2, r2, ..., ln, rn]此外，还有几种启发式方法使用来自单列直方图的信息来生成谓词连接的选择性估计：
1.	属性值独立性 (AVI)：它假定不同属性的值是相互独立选择的。在此假设下，n 列上的谓词的组合选择性分数计算为s1 x s2 x ... x sn其中si第 i 列的选择性。
2.	指数回退 (EBO)：当列具有相关值时，AVI 可能导致显着低估。EBO 通过仅使用 4 个影响递减的最具选择性的谓词来计算组合选择性。组合的选择性分数由下式给出，s_{1} x s_{2}^{1/2} x s_{3}^{1/4} x s_{4}^{1/8}其中s_{k}表示所有谓词中第 k 个最具选择性的分数。
3.	最小选择性 (MinSel)：它将组合选择性计算为跨单个谓词的最小选择性。
对于给定的谓词连接，组合的实际选择性取决于谓词之间的相关程度。如果谓词没有相关性，AVI 将产生良好的估计，而 MinSel 代表与 AVI 相比的另一个极端。EBO有望捕获完全独立和完全相关之间的一些中间场景。将三种启发式方法产生的估计值添加到回归模型的特征中：[l1, r1, l2, r2, ..., ln, rn, est_avi, est_ebo, est_min_sel]可以提高回归模型的准确性和鲁棒性。可以使用多层感知器（MLP）、梯度提升决策树（GBDT）等回归模型。至于损失函数，由于希望最小化估计行与实际行之间的相对误差，因此可以在生成标签时对实际行进行对数变换，使用如下损失函数：loss = MSE(log(act_rows) - log(est_rows)) = MSE(log(act_rows / est_rows)) = MSE(q-error)其中q-error = max(act_rows / est_rows, est_rows / act_rows)，这是基数估计的常用指标。
2.2	数据驱动
在这里使用和积网络 (SPN) 来学习数据的联合概率分布。它可以捕获多个列之间的相关性。直观上，求和节点拆分行，乘积节点拆分列。叶节点代表一列。例如，在下图2.1中，SPN 是通过列region和age客户表学习的。
1.	顶部的总和节点将数据分成两组，左侧组包含 30% 的行，以年长的欧洲客户为主，右侧组包含 70% 的行，以年轻的亚洲客户为主。
2.	然后next在两组中，region并且每组age都被一个产品节点分割。
3.	叶节点确定列region和age每个组的概率分布。
 
图 2.1和积网络（SPN）
学习 SPN 的工作原理是递归地将数据拆分为不同的行组（引入求和节点）或独立列组（引入乘积节点）。
1.	对于行的拆分，可以使用 KMeans 等标准算法。
2.	对于列的拆分，可以使用一些相关性度量，例如 Pearson 相关系数。
有了SPN的定义，可以计算任意列上谓词的概率。直观地说，条件首先在每个相关的叶子上进行评估。之后，自下而上评估 SPN。例如在上图中，为了估计where region='Europe' and age<30，计算相应蓝色区域叶节点中欧洲客户的概率（80% 和 10%）和客户小于 30 岁的概率（15% 和 20%）绿龄叶节点。然后在上面的产品节点级别将这些概率相乘，分别得到 12% 和 2% 的概率。最后，在根级别（sum 节点），必须考虑集群的权重，这导致 12% · 0.3 + 2% · 0.7 = 5%。乘以表中的行数，可以得到大约 50 名 30 岁以下的欧洲客户
3.	数据集与实验环境
数据集IMDB.title，IMDB号称全球最大的电影数据库。实验涉及到的数据来自title表，由于在查询中出现的列都是INT类型，因此只考虑表中的kind_id，production_year ，imdb_id，episode_of_id，season_nr,episode_nr属性值，现在有若干基于这些属性值的合法查询语句，这些语句只有select，from，where，and关键字，没有分组，排序，聚集函数等操作，并且规定from之中最多选择两张表，where之中最多使用一次连接操作。 其中有20000条sql语句已经知道了最终的查询结果的规模（符合要求的元组数量，用于模型的训练），在 query_train.json之中给出，作为训练集，剩余5000条sql语句未知查询结果，在query_test.json之中 给出了这些sql语句。要求通过SVM，CNN，线性回归等机器学习或者深度学习算法[3]，通过训练获得模型，并利用这个模型预测得出剩余5000条sql语句的查询结果规模。

Python3.8、pytorch1.13.0、xgboost1.7.1


## Reference

1. Anshuman Dutt, Chi Wang, Azade Nazi, Srikanth Kandula, Vivek R. Narasayya, Surajit Chaudhuri:
Selectivity Estimation for Range Predicates using Lightweight Models. Proc. VLDB Endow. 12(9): 1044-1057 (2019)
2. Benjamin Hilprecht, Andreas Schmidt, Moritz Kulessa, Alejandro Molina, Kristian Kersting, Carsten Binnig:
DeepDB: Learn from Data, not from Queries! Proc. VLDB Endow. 13(7): 992-1005 (2020)
