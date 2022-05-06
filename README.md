# 待改造
## 模型训练
1. 是否ST
2. 涨幅区分
3. 日线周线一起
4. PRE预先保存
5. FIT 调用CALLBACK打印日志和改变学习率
6. 加入位置编码有用吗？

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
   预训练对于验证集提升还是有明显效果的。
   
3. 20220505 seqlen from 90 to 180 先预训练5年，再训练5年，按年迭代训练验证，即前一年训练，后一年用来验证，滚动进行 
   ROUND 0, 5YEAR: sell_auc: 0.6706 - bsc_auc: 0.7653 - val_sell_auc: 0.5819 - val_bsc_auc: 0.7102
   ROUND 1, 1YEAR: sell_auc: 0.6878 - bsc_auc: 0.7948 - val_sell_auc: 0.5626 - val_bsc_auc: 0.6889
   ROUND 2, 1YEAR: sell_auc: 0.6866 - bsc_auc: 0.7891 - val_sell_auc: 0.5821 - val_bsc_auc: 0.7101
   ROUND 3, 1YEAR: sell_auc: 0.6667 - bsc_auc: 0.7768 - val_sell_auc: 0.5671 - val_bsc_auc: 0.7034
   mean:           sell_auc: 0.6779 - bsc_auc: 0.7815 - val_sell_auc: 0.5734 - val_bsc_auc: 0.7031
   增加seq len后，增加了过拟合，效果更差
# api 实时K线接口：
1. 新浪：
https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol=sh000300&scale=30&ma=no&datalen=250
2. 新浪财经：
http://ifzq.gtimg.cn/appstock/app/kline/mkline?param=sh000300,m30,,320&_var=m30_today&r=0.260880015116