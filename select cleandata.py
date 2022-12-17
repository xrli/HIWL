import shutil
import os
import pandas as pd
import numpy as np

gzimg_path = r'F:\dataSet\galaxy-zoo-the-galaxy-challenge\images_training_rev1'  #原始星系图片存放路径
gzlabel_path = r'F:\dataSet\galaxy-zoo-the-galaxy-challenge\training_solutions_rev1.csv'  #原始星系图片对应csv标签存放路径
gzclean_path = r'F:\dataSet\clean gzdata5' #干净样本存放路径

labels=pd.read_csv(gzlabel_path)
norepeat_df = labels
#阈值筛选
rou = norepeat_df[((norepeat_df['Class1.1']>0.469)&(norepeat_df['Class7.1']>0.50))]
bet = norepeat_df[((norepeat_df['Class1.1']>0.469)&(norepeat_df['Class7.2']>0.50))]
cig = norepeat_df[((norepeat_df['Class1.1']>0.469)&(norepeat_df['Class7.3']>0.50))]
edg = norepeat_df[((norepeat_df['Class1.2']>0.430)&(norepeat_df['Class2.1']>0.602))]
spi = norepeat_df[((norepeat_df['Class1.2']>0.430)&(norepeat_df['Class2.2']>0.715)&(norepeat_df['Class4.1']>0.619))]

#结合类别放入文件夹
origin_path = gzimg_path
root = gzclean_path
class5=['rou', 'bet', 'cig', 'edg', 'spi']
classdict={'rou':rou, 'bet':bet, 'cig':cig, 'edg':edg, 'spi':spi}
for classi in class5:
    #按不同类创建文件夹
    if not os.path.exists(os.path.join(root, classi)):
        os.makedirs(os.path.join(root, classi))
    #对源文件夹类别中的每个图片进行复制导入
    for i in (classdict[classi])['GalaxyID']:
        shutil.copy(os.path.join(origin_path,'%s.jpg'%i),os.path.join(root,classi,'%s.jpg'%i))