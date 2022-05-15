# 待改造
## 模型训练
1. 是否ST
2. 涨幅区分
3. 日线周线一起
4. PRE预先保存
5. FIT 调用CALLBACK打印日志和改变学习率
6. 加入位置编码有用吗？
7. 分钟数据跳掉 有需要复权的。
8. 日线数据复权有意义吗？ 是日期接续还是重叠好。

# 优化日记:
1. 20220504 先训练5年，按年迭代训练验证，即前一年训练，后一年用来验证，滚动进行 
   ROUND 0, 5YEAR: sell_auc: 0.6488 - bsc_auc: 0.7464 - val_sell_auc: 0.5839 - val_bsc_auc: 0.7085
   ROUND 1, 1YEAR: sell_auc: 0.6782 - bsc_auc: 0.7858 - val_sell_auc: 0.5670 - val_bsc_auc: 0.6898
   ROUND 2, 1YEAR: sell_auc: 0.6702 - bsc_auc: 0.7740 - val_sell_auc: 0.5904 - val_bsc_auc: 0.7109 
   ROUND 3, 1YEAR: sell_auc: 0.6816 - bsc_auc: 0.7872 - val_sell_auc: 0.5617 - val_bsc_auc: 0.6932
   mean:           sell_auc: 0.6697 - bsc_auc: 0.7733 - val_sell_auc: 0.5757 - val_bsc_auc: 0.7006
   可以看出后续滚动训练开始过拟合增加，所以需要在后续滚动训练中减少学习率
   
2. 20220504 先预训练5年，再训练5年，按年迭代训练验证，即前一年训练，后一年用来验证，滚动进行 
   ROUND 0, 5YEAR: sell_auc: 0.6449 - bsc_auc: 0.7446 - val_sell_auc: 0.5922 - val_bsc_auc: 0.7188
   ROUND 1, 1YEAR: sell_auc: 0.6696 - bsc_auc: 0.7796 - val_sell_auc: 0.5665 - val_bsc_auc: 0.6887
   ROUND 2, 1YEAR: sell_auc: 0.6571 - bsc_auc: 0.7643 - val_sell_auc: 0.6068 - val_bsc_auc: 0.7186
   ROUND 3, 1YEAR: sell_auc: 0.6787 - bsc_auc: 0.7833 - val_sell_auc: 0.5767 - val_bsc_auc: 0.7072
   mean:           sell_auc: 0.6625 - bsc_auc: 0.7679 - val_sell_auc: 0.5855 - val_bsc_auc: 0.7083
   预训练对于验证集提升还是有明显效果的↑。
   
3. 20220505 seqlen from 90 to 180 先预训练5年，再训练5年，按年迭代训练验证，即前一年训练，后一年用来验证，滚动进行 
   ROUND 0, 5YEAR: sell_auc: 0.6706 - bsc_auc: 0.7653 - val_sell_auc: 0.5819 - val_bsc_auc: 0.7102
   ROUND 1, 1YEAR: sell_auc: 0.6878 - bsc_auc: 0.7948 - val_sell_auc: 0.5626 - val_bsc_auc: 0.6889
   ROUND 2, 1YEAR: sell_auc: 0.6866 - bsc_auc: 0.7891 - val_sell_auc: 0.5821 - val_bsc_auc: 0.7101
   ROUND 3, 1YEAR: sell_auc: 0.6667 - bsc_auc: 0.7768 - val_sell_auc: 0.5671 - val_bsc_auc: 0.7034
   mean:           sell_auc: 0.6779 - bsc_auc: 0.7815 - val_sell_auc: 0.5734 - val_bsc_auc: 0.7031
   增加seq len后，增加了过拟合，效果↓
   
3. 20220506 seqlen:90 mask prob from 0.1 to 0.15 预训练5年，再训练5年，然后逐年滚动验证
   ROUND 0, 5YEAR: sell_auc: 0.6477 - bsc_auc: 0.7464 - val_sell_auc: 0.5907 - val_bsc_auc: 0.7147 
   ROUND 1, 1YEAR: sell_auc: 0.6696 - bsc_auc: 0.7794 - val_sell_auc: 0.5723 - val_bsc_auc: 0.6909
   ROUND 2, 1YEAR: sell_auc: 0.6548 - bsc_auc: 0.7610 - val_sell_auc: 0.6001 - val_bsc_auc: 0.7171
   ROUND 3, 1YEAR: sell_auc: 0.6784 - bsc_auc: 0.7846 - val_sell_auc: 0.5669 - val_bsc_auc: 0.7011
   mean:           sell_auc: 0.6626 - bsc_auc: 0.7678 - val_sell_auc: 0.5825 - val_bsc_auc: 0.7059
   增加mask prob后，效果↓
   
4. 20220509 先训练5年，按年迭代训练验证，即前一年训练，后一年用来验证，滚动进行 
   ROUND 0, 5YEAR: sell_auc: 0.7437 - bsc_auc: 0.7544 - val_sell_auc: 0.6868 - val_bsc_auc: 0.7151
   ROUND 1, 1YEAR: sell_auc: 0.7559 - bsc_auc: 0.7889 - val_sell_auc: 0.6863 - val_bsc_auc: 0.7038
   ROUND 2, 1YEAR: sell_auc: 0.7681 - bsc_auc: 0.7955 - val_sell_auc: 0.6853 - val_bsc_auc: 0.7175 
   ROUND 3, 1YEAR: sell_auc: 0.7133 - bsc_auc: 0.7522 - val_sell_auc: 0.6780 - val_bsc_auc: 0.7093
   ROUND 4, 1YEAR: sell_auc: 0.7712 - bsc_auc: 0.8036 - val_sell_auc: 0.6811 - val_bsc_auc: 0.6919
   mean:           sell_auc: 0.7504 - bsc_auc: 0.7789 - val_sell_auc: 0.6835 - val_bsc_auc: 0.7075
   无预训练，相对1有明显提升
   
5. 20220510 先训练5年，按年迭代训练验证，即前一年训练，后一年用来验证，滚动进行 
   ROUND 0, 5YEAR: val_sell_auc: 0.6742 - val_bsc_auc: 0.5562 - val_bso_auc: 0.5338 - val_bs_auc: 0.5428 - val_bsl_auc: 0.6039 - val_bgc2_auc: 0.5896 - val_bgc5_auc: 0.6482
   ROUND 1, 1YEAR: val_sell_auc: 0.6767 - val_bsc_auc: 0.5575 - val_bso_auc: 0.5492 - val_bs_auc: 0.5534 - val_bsl_auc: 0.6360 - val_bgc2_auc: 0.6325 - val_bgc5_auc: 0.6995
   ROUND 2, 1YEAR: val_sell_auc: 0.6884 - val_bsc_auc: 0.5847 - val_bso_auc: 0.5780 - val_bs_auc: 0.5896 - val_bsl_auc: 0.6583 - val_bgc2_auc: 0.6708 - val_bgc5_auc: 0.7303
   ROUND 3, 1YEAR: val_sell_auc: 0.6738 - val_bsc_auc: 0.5684 - val_bso_auc: 0.5611 - val_bs_auc: 0.5677 - val_bsl_auc: 0.6458 - val_bgc2_auc: 0.6913 - val_bgc5_auc: 0.7545
   ROUND 4, 1YEAR: val_sell_auc: 0.6778 - val_bsc_auc: 0.5667 - val_bso_auc: 0.5477 - val_bs_auc: 0.5674 - val_bsl_auc: 0.6438 - val_bgc2_auc: 0.6843 - val_bgc5_auc: 0.7546
   MEAN   ,      : val_sell_auc: 0.6782 - val_bsc_auc: 0.5667 - val_bso_auc: 0.5540 - val_bs_auc: 0.5642 - val_bsl_auc: 0.6376 - val_bgc2_auc: 0.6537 - val_bgc5_auc: 0.7174
   无预训练，增加PRE_CLOSE,修正LABEL错误（之前LABEL错误大部分都是0），AUC下降↓
   
6. 20220510 去除 PRE_CLOSE 先训练5年，按年迭代训练验证，即前一年训练，后一年用来验证，滚动进行5年
   ROUND 0, 5YEAR: val_sell_auc: 0.6791 - val_bsc_auc: 0.5714 - val_bso_auc: 0.5451 - val_bs_auc: 0.5631 - val_bsl_auc: 0.6194 - val_bgc2_auc: 0.6080 - val_bgc5_auc: 0.6578
   ROUND 1, 1YEAR: val_sell_auc: 0.6702 - val_bsc_auc: 0.5436 - val_bso_auc: 0.5383 - val_bs_auc: 0.5352 - val_bsl_auc: 0.6183 - val_bgc2_auc: 0.6165 - val_bgc5_auc: 0.6751
   ROUND 2, 1YEAR: val_sell_auc: 0.6831 - val_bsc_auc: 0.5744 - val_bso_auc: 0.5681 - val_bs_auc: 0.5766 - val_bsl_auc: 0.6510 - val_bgc2_auc: 0.6691 - val_bgc5_auc: 0.7388
   ROUND 3, 1YEAR: val_sell_auc: 0.6752 - val_bsc_auc: 0.5723 - val_bso_auc: 0.5632 - val_bs_auc: 0.5731 - val_bsl_auc: 0.6492 - val_bgc2_auc: 0.6870 - val_bgc5_auc: 0.7505
   ROUND 4, 1YEAR: val_sell_auc: 0.6794 - val_bsc_auc: 0.5699 - val_bso_auc: 0.5581 - val_bs_auc: 0.5750 - val_bsl_auc: 0.6466 - val_bgc2_auc: 0.6850 - val_bgc5_auc: 0.7606
   MEAN   ,      : val_sell_auc: 0.6774 - val_bsc_auc: 0.5663 - val_bso_auc: 0.5546 - val_bs_auc: 0.5646 - val_bsl_auc: 0.6369 - val_bgc2_auc: 0.6531 - val_bgc5_auc: 0.7166
   无预训练，去除 PRE_CLOSE，AUC略微下降↓
   
7. 20220511 增加 PRE_CLOSE 先训练5年，按年迭代训练验证，即前一年训练，后一年用来验证，滚动进行5年
   ROUND 0, 5YEAR: val_sell_auc: 0.6829 - val_bsc_auc: 0.5764 - val_bso_auc: 0.5490 - val_bs_auc: 0.5627 - val_bsl_auc: 0.6181 - val_bgc2_auc: 0.6067 - val_bgc5_auc: 0.6633
   ROUND 1, 1YEAR: val_sell_auc: 0.6739 - val_bsc_auc: 0.5452 - val_bso_auc: 0.5467 - val_bs_auc: 0.5453 - val_bsl_auc: 0.6295 - val_bgc2_auc: 0.6310 - val_bgc5_auc: 0.6891
   ROUND 2, 1YEAR: val_sell_auc: 0.6845 - val_bsc_auc: 0.5808 - val_bso_auc: 0.5738 - val_bs_auc: 0.5843 - val_bsl_auc: 0.6580 - val_bgc2_auc: 0.6631 - val_bgc5_auc: 0.7271
   ROUND 3, 1YEAR: val_sell_auc: 0.6749 - val_bsc_auc: 0.5631 - val_bso_auc: 0.5541 - val_bs_auc: 0.5608 - val_bsl_auc: 0.6458 - val_bgc2_auc: 0.6819 - val_bgc5_auc: 0.7433
   ROUND 4, 1YEAR: val_sell_auc: 0.6835 - val_bsc_auc: 0.5642 - val_bso_auc: 0.5524 - val_bs_auc: 0.5646 - val_bsl_auc: 0.6456 - val_bgc2_auc: 0.6834 - val_bgc5_auc: 0.7622
   MEAN   ,      : val_sell_auc: 0.6799 - val_bsc_auc: 0.5659 - val_bso_auc: 0.5552 - val_bs_auc: 0.5635 - val_bsl_auc: 0.6394 - val_bgc2_auc: 0.6532 - val_bgc5_auc: 0.7170
   无预训练，增加 PRE_CLOSE，日线至前一天，AUC略微提升↑
   
8. 20220511 日线与分钟线收盘对齐
   ROUND 0, 5YEAR: val_sell_auc: 0.6818 - val_bsc_auc: 0.5629 - val_bso_auc: 0.5386 - val_bs_auc: 0.5416 - val_bsl_auc: 0.6146 - val_bgc2_auc: 0.5882 - val_bgc5_auc: 0.6515
   ROUND 1, 1YEAR: val_sell_auc: 0.6811 - val_bsc_auc: 0.5594 - val_bso_auc: 0.5696 - val_bs_auc: 0.5769 - val_bsl_auc: 0.6365 - val_bgc2_auc: 0.6503 - val_bgc5_auc: 0.7106
   ROUND 2, 1YEAR: val_sell_auc: 0.6849 - val_bsc_auc: 0.5915 - val_bso_auc: 0.5727 - val_bs_auc: 0.5966 - val_bsl_auc: 0.6584 - val_bgc2_auc: 0.6704 - val_bgc5_auc: 0.7262
   ROUND 3, 1YEAR: val_sell_auc: 0.6721 - val_bsc_auc: 0.5580 - val_bso_auc: 0.5568 - val_bs_auc: 0.5580 - val_bsl_auc: 0.6423 - val_bgc2_auc: 0.6780 - val_bgc5_auc: 0.7455
   ROUND 4, 1YEAR: val_sell_auc: 0.6770 - val_bsc_auc: 0.5653 - val_bso_auc: 0.5581 - val_bs_auc: 0.5714 - val_bsl_auc: 0.6452 - val_bgc2_auc: 0.6877 - val_bgc5_auc: 0.7662
   MEAN   ,      : val_sell_auc: 0.6794 - val_bsc_auc: 0.5674 - val_bso_auc: 0.5592 - val_bs_auc: 0.5689 - val_bsl_auc: 0.6394 - val_bgc2_auc: 0.6549 - val_bgc5_auc: 0.7200
   日线与分钟线收盘对齐，AUC略微提升↑
   
9. 20220512 日线改用未复权
   ROUND 0, 5YEAR: val_sell_auc: 0.6823 - val_bsc_auc: 0.5726 - val_bso_auc: 0.5473 - val_bs_auc: 0.5569 - val_bsl_auc: 0.6227 - val_bgc2_auc: 0.6029 - val_bgc5_auc: 0.6607
   ROUND 1, 1YEAR: val_sell_auc: 0.6708 - val_bsc_auc: 0.5282 - val_bso_auc: 0.5323 - val_bs_auc: 0.5216 - val_bsl_auc: 0.6221 - val_bgc2_auc: 0.6198 - val_bgc5_auc: 0.6790
   ROUND 2, 1YEAR: val_sell_auc: 0.6748 - val_bsc_auc: 0.5762 - val_bso_auc: 0.5569 - val_bs_auc: 0.5763 - val_bsl_auc: 0.6467 - val_bgc2_auc: 0.6599 - val_bgc5_auc: 0.7295
   ROUND 3, 1YEAR: val_sell_auc: 0.6675 - val_bsc_auc: 0.5422 - val_bso_auc: 0.5415 - val_bs_auc: 0.5324 - val_bsl_auc: 0.6346 - val_bgc2_auc: 0.6753 - val_bgc5_auc: 0.7387
   ROUND 4, 1YEAR: val_sell_auc: 0.6806 - val_bsc_auc: 0.5657 - val_bso_auc: 0.5500 - val_bs_auc: 0.5703 - val_bsl_auc: 0.6460 - val_bgc2_auc: 0.6819 - val_bgc5_auc: 0.7590
   MEAN   ,      : val_sell_auc: 0.6752 - val_bsc_auc: 0.5570 - val_bso_auc: 0.5456 - val_bs_auc: 0.5515 - val_bsl_auc: 0.6344 - val_bgc2_auc: 0.6480 - val_bgc5_auc: 0.7134
   日线改用未复权，AUC 明显降低↓
   
9. 20220515 日线改用未复权
   ROUND 0, 5YEAR: loss: 0.6081 - auc: 0.7170 - val_loss: 0.6089 - val_auc: 0.6907
   ROUND 1, 1YEAR: loss: 0.5864 - auc_1: 0.7148 - val_loss: 0.5993 - val_auc_1: 0.6965
   ROUND 2, 1YEAR: loss: 0.5952 - auc_1: 0.7224 - val_loss: 0.6304 - val_auc_1: 0.6804
   ROUND 3, 1YEAR: loss: 0.6073 - auc_1: 0.7254 - val_loss: 0.6027 - val_auc_1: 0.6903
   ROUND 4, 1YEAR: loss: 0.5978 - auc_1: 0.7177 - val_loss: 0.6299 - val_auc_1: 0.6887
   MEAN   ,      : loss: 0.5990 - auc: 0.7195 - val_loss: 0.6142 - val_auc: 0.6893
   增加START TOKEN， 同时对最后一层增加DROPOUT 和 L1，L2约束。 初始学习率分别设置为5E-5和1e-5，只有花SELL. 效果显著提升

# api 实时K线接口：
1. 新浪：
https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol=sh000300&scale=30&ma=no&datalen=250
2. 新浪财经：
http://ifzq.gtimg.cn/appstock/app/kline/mkline?param=sh000300,m30,,320&_var=m30_today&r=0.260880015116