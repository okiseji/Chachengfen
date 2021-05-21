# -*- coding: UTF-8 -*-
import requests
import json
import numpy as np
from sklearn.cluster import KMeans
import math
import time as timefun
import matplotlib.pyplot as plt
import csv
import unicodecsv as ucsv
from mpl_toolkits.mplot3d import Axes3D
import threading
import random
import os
from multiprocessing import Pool,Queue,Lock

def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    # @wraps(function)
    def function_timer(*args, **kwargs):
        print ('[Function: {name} start...]'.format(name = function.__name__))
        t0 = timefun.time()
        result = function(*args, **kwargs)
        t1 = timefun.time()
        print ('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer

cookies = {
    "SESSDATA": "50048352%2C1637033094%2C63522%2A51",
}
headers = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.62',
}
s=requests.Session()
#TODO 强烈建议下列属性前面加上@staticmethod装饰器，而且很明显的是loadData()这些东西应该不属于这里
class algo(object):
    def __init__(self):
        self.weight1=[0.5,0.8,0.5]#up feature的权重矩阵
        self.weight2=[0.8,0.3,0.4]#tag feature的权重矩阵
        self.sigma=100#这个别用
    #已弃用
    def loadData(self,dict):
        retName = []
        retData = []
        for i in dict.keys():
            retName.append(i)
            retData.append(dict[i])
        for m in range(1, len(retData)):
            return retName, retData
    def normalize(self,X,axis=-1,p=2):
        lp_norm=np.atleast_1d(np.linalg.norm(X,p,axis))
        lp_norm[lp_norm==0]=1
        return X/np.expand_dims(lp_norm,axis)
    def euclidean_distance2(self,one_sample, X):
        weight_matrix=np.mat(self.weight2)
        one_sample = one_sample.reshape(1, -1)
        X = X.reshape(X.shape[0], -1)
        distances = (np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2)*(weight_matrix.reshape(3,1))).sum(axis=1)
        return distances
    def euclidean_distance1(self, one_sample, X):
        weight_matrix = np.mat(self.weight1)
        one_sample = one_sample.reshape(1, -1)
        X = X.reshape(X.shape[0], -1)
        distances = (np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2) * (weight_matrix.reshape(3, 1))).sum(axis=1)
        return distances
    def normal_distribution(self,x, mean, sigma):
        return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)
    def timeconverge(self,tm,sigma):
        time_now=int(timefun.time())
        # ln=self.normal_distribution(tm-time_now,0,self.sigma)*1000000000
        during=tm-time_now
        return self.normal_distribution(during, 0, sigma) / self.normal_distribution(0, 0, sigma)
    def up_data_converge(self,follower,sigma):
        '''粉丝数参数是500,投稿数参数是5,平均观看参数是500'''
        return 1-(self.normal_distribution(follower, 0, sigma)/self.normal_distribution(0, 0, sigma))
    def up_cluster(self,dict):
        names, data = self.loadData(dict)
        km = KMeans(n_clusters=4)
        label = km.fit_predict(np.array(data).reshape(-1, 1))
        expenses = np.sum(km.cluster_centers_, axis=1)
        print(expenses)
        citycluster = [[], [], [], []]
        for i in range(len(names)):
            citycluster[label[i]].append(names[i])
        for i in range(len(citycluster)):
            print("expenses:%2f" % expenses[i])
            print(citycluster[i])
    # 已弃用
    def ratio(self,theta):
        count=theta[0]
        use=theta[1]
        time=theta[2]
class Kmeans():
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001,type=1):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        self.type=type
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids
    def _closest_centroid(self, sample, centroids):
        #别吵这里很关键
        if self.type==2:
            distances = algo.euclidean_distance2(algo(),one_sample=sample, X=centroids)
        if self.type==1:
            distances = algo.euclidean_distance1(algo(), one_sample=sample, X=centroids)
        closest_i = np.argmin(distances)
        return closest_i
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = int(self._closest_centroid(sample=sample, centroids=centroids))
            clusters[centroid_i].append(sample_i)
        return clusters
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred
    def predict(self, X):
        centroids = self.init_random_centroids(X)
        for _ in range(self.max_iterations):
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids
            centroids = self.update_centroids(clusters, X)
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break
            print(self.save_model(centroids))
            return self.get_cluster_labels(clusters, X)
    def save_model(self,centroids):#array
        list=centroids.tolist()
        return list
    def load_model(self,centroids,X):#list
        centroids=np.asarray(centroids)
        clusters = self.create_clusters(centroids, X)
        return self.get_cluster_labels(clusters, X)
'''寄了的函数'''

def name(mid):
    link="http://api.bilibili.com/x/space/acc/info?mid="+str(mid)
    # s = requests.Session()
    r = s.get(link, cookies=cookies, headers=headers)
    dict = json.loads(r.text)
    return dict["data"]["name"]

def save_to_file(list,name):
    f=open(name,'w',newline='')
    w=ucsv.writer(f,encoding='gbk')
    w.writerows(list)
    f.close()
def read_from_file(name):
    f=open(name,'r',encoding='gbk')
    r=csv.reader(f)
    list=[]
    for col in r:
        ncol = []
        for i in range(1,4):
            ncol.append(float(col[i]))#在这里改数据类型
        list.append(ncol)
    f.close()
    return list
'''用了捏'''
def get_follow_list(mid):
    follow_list=[]
    for i in range(1,5):
        link="https://api.bilibili.com/x/relation/followings?vmid="+str(mid)+"&pn="+str(i)+"&ps=0&order=desc&jsonp=jsonp"
        # s = requests.Session()
        r = s.get(link, cookies=cookies, headers=headers)
        dict = json.loads(r.text)
        #do not use 'or' since the hidden of the follow list
        if dict["code"] == -400:
            break
        if "data" not in dict.keys():
            break
        if dict["data"]["list"] == []:
            break
        for follow in dict["data"]["list"]:
            follow_list.append(follow["mid"])
    # print(follow_list[:50])#50个人足以吧
    return follow_list[:50]
def get_up_info(mid):
    # s=requests.Session()
    link1="https://api.bilibili.com/x/relation/stat?vmid="+str(mid)
    link2="https://api.bilibili.com/x/space/upstat?mid="+str(mid)
    r1=s.get(link1,headers=headers)
    r2=s.get(link2,cookies=cookies,headers=headers)
    dict1 = json.loads(r1.text)
    dict2 = json.loads(r2.text)
    follower=dict1["data"]["follower"]
    video_view=dict2["data"]["archive"]["view"]
    article_view=dict2["data"]["article"]["view"]
    view=video_view+article_view
    list=[]
    num=len(get_video_list(mid))
    # print(num)
    list.append(algo().up_data_converge(follower,500))
    list.append(algo().up_data_converge(num,5))
    if num!=0:
        list.append(algo().up_data_converge(int(view/num),50))
    else:
        list.append(0)
    # print(list)
    return list
def get_video_list(mid):
    i = 1
    list={}
    # while 1:
    link = "http://api.bilibili.com/x/space/arc/search?mid="+str(mid)+"&pn="+str(i)+"&ps=20"#20个视频足以
    # s=requests.Session()
    timefun.sleep(0.5)
    r=s.get(link, headers=headers)
    dict=json.loads(r.text)
    # if dict["code"]==-400 or dict["data"]["list"]["vlist"]==[]:
    #     break
    for video in dict["data"]["list"]["vlist"]:
        url=video["bvid"]
        time=video["created"]
        list[url]=time
    i+=1
    return list
# @func_timer
def get_tag_value(bvid,time):
    tagvec={}
    link="http://api.bilibili.com/x/tag/archive/tags?bvid="+str(bvid)
    # print(link)
    # s=requests.Session()
    timefun.sleep(0.5)
    r = s.get(link, headers=headers)
    dict = json.loads(r.text)
    try:
        for tagdict in dict["data"]:
            if tagdict["count"]["use"]>1000 and time>0.00001:
                name=tagdict["tag_name"]
                tagvec[name] = [1,tagdict["count"]["use"],time]
        return tagvec
    except:
        return {}
# TODO: modify into multiprocessing
def get_tags_vec(mid):
    all_dict={}
    list = get_video_list(mid)
    for video in list.keys():
        taginfo=get_tag_value(video,list[video])
        for videotags in taginfo.keys():
            #TODO 命名有问题，videotags -> videotag
            if videotags in  all_dict.keys():
                all_dict[videotags][0]+=1 #TODO 这个值似乎没有必要捏，可以这样写: for videotags,num in taginfo.items()
                all_dict[videotags][1] = taginfo[videotags][1]
                all_dict[videotags][2]+=taginfo[videotags][2]
            else:
                all_dict[videotags]=taginfo[videotags]
    for tag in all_dict.keys():
        all_dict[tag][2]=all_dict[tag][2]/all_dict[tag][0]
        g=algo().up_data_converge(all_dict[tag][1],5000)
        all_dict[tag][1] = float('%.5f' % g)
        f = algo().timeconverge(all_dict[tag][2], 50000000)
        all_dict[tag][2]=float('%.5f' % f)
        k = all_dict[tag][0] / len(list)
        all_dict[tag][0]=float('%.5f' % k)
    return all_dict
def is_up(mid):
    #读取filename 改完了
    model = [[0, 0, 0],[1, 1, 1]]
    list=get_up_info(mid)
    X=np.asarray(list)
    clf = Kmeans(k=2,type=1)
    y_pred = clf.load_model(model,X)
    if y_pred[0]==1.0:
        return 1
    else:
        return 0
def get_up_vec(dict):
    list=[]
    namelist=[]
    vector=[]
    for i in dict.keys():
        list.append(dict[i])
        namelist.append(i)
    model = [[0, 0, 0], [1, 1, 1]]
    X=np.asarray(list)
    clf = Kmeans(k=2,type=2)
    y_pred = clf.load_model(model,X)
    for i in range(0,len(X)):
        if y_pred[i]==1.0:
            vector.append(namelist[i])
    if len(vector)>10:
        return vector[:10]
    else:
        return vector
def get_user_info(mid):
    dict={}
    uplist=get_follow_list(mid)
    for up in uplist:
        if is_up(up):
            list=get_up_vec(get_tags_vec(up))
            for tag in list:
                if tag not in dict.keys():
                    dict[tag]=1 #此处是否还要加上权重？没必要
                dict[tag]+=1
    return dict
def get_user_vec(mid):
    dict=get_user_info(mid)
    dict = combine(dict)
    if len(dict)>20:
        list = sorted(dict.items(), key=lambda d: d[1])[-20:]
    else:
        list = sorted(dict.items(), key=lambda d: d[1])
    ndict = {}
    for i in list:
        ndict[i[0]] = i[1]
    print(ndict)
    return ndict
def combine(dict):
    cdict={'v':['虚拟','V','v'],
           'jp':['日本','日语','日文'],
           'lol':['LOL','lol','英雄联盟','otto','炫','若','电棍'],
           'erciyuan':['二次元','动漫','动画','国创','漫画','AMV',],
           'xianchong':['生活','食','vlog','VLOG','Vlog','餐','味道','动物','宠','猫','吃','旅行','购物','文艺','料理','狗','妆','励志','店','肤','口红','治愈'],
           'game':['steam','STEAM','游戏','RPG','单机','实况','攻略'],
           'learn':['学习','数学','研','高中','校园','初中'],
           'music&dance':['唱','歌','奏','音乐','声','曲'],
           'rsq&dance':['少女','妹','JK','舞蹈'],
           'history':['钢铁雄心','p社','P社','KR','Kaiserreich','架空历史','Paradox','欧陆风云','王国风云','德国','苏','俄','毛子','历史','战','罗马'],
           'jz':['魔怔','TNO','Jreg','JrEg','意识形态','五学','第五','抽象','碳水','兔兔','奋斗','赢麻了','小艾','深深','神神','抗抗','鼠鼠','神兔','典中典','左'],
           'zz':['中国','美国','印度','拜登','中美关系'],
           }
    poplist = ['A-SOUL','A-soul', '嘉然', '贝拉', '乃琳', '向晚','知识分享官','会员','电影','沙雕','搞笑','吐槽','视频','打卡挑战','野生技术协会','必剪创作','自制','教程','二创','新年']
    newnumlist={}
    sum=0
    for i in dict.keys():
        if dict[i]==1:
            poplist.append(i)
        sum += dict[i]
        for p in cdict.keys():
            for q in cdict[p]:
                if q in i:
                    num=dict[i]
                    if p not in newnumlist:
                        newnumlist[p]=0
                    newnumlist[p]+=num
                    poplist.append(i)
    poplist2=[]
    for i in poplist:
        for q in dict.keys():
            if i in q:
                poplist2.append(q)
    for i in poplist2:
        if i in dict.keys():
            dict.pop(i)
    for i in newnumlist.keys():
        if newnumlist[i]>0:
            dict[i]=newnumlist[i]
    for i in dict.keys():
        f=dict[i]/sum
        dict[i]=float('%.5f'%f)
    return dict
def get_comment_users(oid):
    list = []
    for pn in range(1,10):
        link="http://api.bilibili.com/x/v2/reply?jsonp=jsonp&pn="+str(pn)+"&type=1&oid="+str(oid)
        # s = requests.Session()
        r = s.get(link, cookies=cookies, headers=headers)
        dict = json.loads(r.text)
        for maincomment in dict["data"]["replies"]:
            list.append(maincomment["mid"])
            if maincomment["replies"] != None:
                for comment in maincomment["replies"]:
                    list.append(comment["mid"])
    print(list)
    return list
def task(i):
    list=usrlist
    print('Run task %s (%s)...' % (i, os.getpid()))
    time_start = timefun.time()
    dict=get_user_vec(list[i])
    f = open('final_data3.csv', 'ab')
    w = ucsv.writer(f, encoding='gbk', )
    list = make_write_list(dict)
    w.writerow(list)
    f.close()
    time_end = timefun.time()
    print('time cost', time_end - time_start, 's')
def make_write_list(dict):
    list=[]
    for i in dict.keys():
        list.append(i)
        list.append(dict[i])
    return list
usrlist=[11160129, 11160129, 5269035, 26346687, 11621965, 55810643, 55810643, 55810643, 277773844, 3465547,
         11160129, 12239076, 17637149, 409259291, 33084382, 342849130, 12239076, 22628428, 3465547, 320225571,
         11417821, 62098885, 1842470, 13172201, 21176819, 231714473, 244249751, 111970933, 256270509, 215345667,
         422791026, 33159698, 334391066, 23372733, 10034871, 231714473, 83022313, 3465547, 55810643, 55810643,
         55810643, 689666512, 28238197, 689666512, 92689393, 2317897, 18777051, 2101096513, 29580976, 389270791,
         34357325, 5896507, 38951939, 430985638, 2352159, 344486151, 319082273, 327043081, 488846518, 100479393,
         524447220, 177607438, 4729643, 2101096513, 4729643, 1536643353, 301383656, 38987835, 3465547, 402177858,
         2126522389, 10034871, 231714473, 276924136, 48745691, 3911582, 24432860, 3911582, 25257749, 2101096513,
         151223261, 12599130, 175529337, 695969669, 2271012, 4271451, 90082869, 272912036, 1354510, 55810643,
         55810643, 55810643, 790574, 186857552, 22765782, 186857552, 7790127, 32808012, 23802946, 7958301,
         231714473, 23274682, 85510270, 15888, 347282884, 11426718, 16576175, 453862614, 205295005, 33118861,
         33159698, 260588960, 20981358, 23620881, 5808919, 38987835, 66421879, 7536867, 2101096513, 1536643353,
         100989153, 2190714, 47537, 629147219, 90626882, 11160129, 76576170, 507391361, 47537, 2126522389,
         507391361, 387557433, 20253196, 13951885, 357450259, 55810643, 55810643, 55810643, 548473218, 20340537,
         22508104, 6076247, 2101096513, 18375574, 38523561, 341193324, 38401083, 376071980, 26441289, 471655008,
         38987835, 53301341, 98847394, 37487310, 34357325, 165434476, 4208930, 21521290, 6948593, 15888,
         21463142, 34403533, 657857, 2101096513, 646072980, 7536867, 16449995, 88164206, 5269035, 5269035,
         351749883, 7479954, 23802946, 453862614, 23802946, 629147219, 85527523, 551943, 11417821, 33159698,
         28938622, 4322916, 23802946, 206217613, 16187016, 3134312, 8475032, 179399138, 296879476, 32987763,
         393805590, 2126522389, 23802946, 2126522389, 233618077, 327593898, 2020933, 1695830497, 23802946,
         231714473, 1695830497, 13029637, 1754915, 231714473, 451373265, 13029637, 7795212, 55810643, 55810643,
         263500210, 55810643, 107286923, 8176998, 80693847, 2889486, 8176998, 157622402, 1349918, 1754915,
         288701944, 24458495, 23274682, 43073974, 23802946, 415324528, 231714473, 231045855, 277419062, 50966754,
         32369380, 103241072, 15409023, 38584103, 65809613, 123258651, 3085856, 323786640, 3987048, 482626835,
         46782757, 22682236, 548473218, 105030952, 1790104282, 12680219, 1349918, 586703897, 255478384, 318916963,
         74415209, 403256552, 268464071, 8572309, 34883317, 341342103, 35151400, 319190249, 38401083, 8572309]

if __name__=='__main__':
    # print(len(usrlist))
    for q in range(10, 20):
        print('Parent process %s.' % os.getpid())
        p = Pool(10)
        for i in range(10*q,10*q+10):
            p.apply_async(task, args=(i,))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')




    # X,y = datasets.make_blobs(n_samples=1000,n_features=3,centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],cluster_std=[0.2, 0.1, 0.2, 0.2],random_state=9)
    # model=[[0.10942633084997674, 0.35183741177574857, 0.5618462520781641], [0.9698537751610932, 0.9528995663760753, 0.9714264607377109]]
    # fig = plt.figure(figsize=(12, 8))
    # ax = Axes3D(fig, rect=(0, 0, 1, 1), elev=300, azim=200)
    # plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], X[y_pred == 0][:, 2])
    # plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], X[y_pred == 1][:, 2])
    # plt.scatter(X[y_pred == 2][: , 0], X[y_pred == 2][:, 1], X[y_pred == 2][:, 2])
    # plt.scatter(X[y_pred == 3][:, 0], X[y_pred == 3][:, 1], X[y_pred == 3][:, 2])
    # plt.show()
