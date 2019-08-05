# encoding=utf-8
import pymongo
from items import InformationItem, TweetsItem, FollowsItem, FansItem

class MongoDBPipleline(object):
    def __init__(self):
        client = pymongo.MongoClient("localhost", 27017)
        db = client["Sina"]
        self.Information = db["Information"]
        self.Tweets = db["Tweets"]
        self.num = 0
        # self.Follows = db["Follows"]
        # self.Fans = db["Fans"]

    def process_item(self, item, spider):
        """ 判断item的类型，并作相应的处理，再入数据库 """
        # inf = InformationItem()
        # print(isinstance(inf, InformationItem), type(InformationItem),type(inf))
        # print("process item....:",type(item), isinstance(item,InformationItem),isinstance(item, TweetsItem), type(item)==TweetsItem)
        if isinstance(item, InformationItem) or item.type == 1:
            try:
                self.Information.insert_one(dict(item))
                # print("information")
            except Exception:
                pass
        elif isinstance(item, TweetsItem) or item.type ==2 :
            try:
                self.Tweets.insert_one(dict(item))
            except Exception as e:
                print(e)
            try:
                with open("E:\\tasks\\PPGCN\\Data\\trainData\\event_"+str(self.num)+".txt","w") as f:
                    # 第0行：发布时间； 第1行：微博内容； 第2行：文本属性; 第3行：label
                    content = str(dict(item))
                    f.write(content)
                    f.close()
                self.num += 1
                # print("tweets")
            except Exception as e:
                print(e)
        # elif isinstance(item, FollowsItem):
        #     followsItems = dict(item)
        #     follows = followsItems.pop("follows")
        #     for i in range(len(follows)):
        #         followsItems[str(i + 1)] = follows[i]
        #     try:
        #         self.Follows.insert(followsItems)
        #     except Exception:
        #         pass
        # elif isinstance(item, FansItem):
        #     fansItems = dict(item)
        #     fans = fansItems.pop("fans")
        #     for i in range(len(fans)):
        #         fansItems[str(i + 1)] = fans[i]
        #     try:
        #         self.Fans.insert(fansItems)
        #     except Exception:
        #         pass
        return item
