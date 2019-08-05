# encoding=utf-8
import re
from datetime import datetime, timedelta
import pymongo
from scrapy.spiders import CrawlSpider
from scrapy.selector import Selector
from scrapy.http import Request
from Sina_spider1.items import InformationItem, TweetsItem, FollowsItem, FansItem


class Spider(CrawlSpider):
    name = "sinaSpider"
    host = "https://weibo.cn"
    # start_urls = []

    def start_requests(self):
        scrawl_ID = set(self.get_start_urls())  # 记录待爬的微博ID
        finish_ID = set()  # 记录已爬的微博ID
        while scrawl_ID.__len__():
            ID = scrawl_ID.pop()
            finish_ID.add(ID)  # 加入已爬队列
            ID = str(ID)
            # follows = []
            # followsItems = FollowsItem()
            # followsItems["_id"] = ID
            # followsItems["follows"] = follows
            # fans = []
            # fansItems = FansItem()
            # fansItems["_id"] = ID
            # fansItems["fans"] = fans

            # url_follows = "http://weibo.cn/%s/follow" % ID
            # url_fans = "http://weibo.cn/%s/fans" % ID
            url_tweets = "https://weibo.cn/%s/profile?filter=1&page=1" % ID
            url_information0 = "https://weibo.cn/attgroup/opening?uid=%s" % ID
            # yield Request(url=url_follows, meta={"item": followsItems, "result": follows},
            #               callback=self.parse3)  # 去爬关注人
            # yield Request(url=url_fans, meta={"item": fansItems, "result": fans}, callback=self.parse3)  # 去爬粉丝
            yield Request(url=url_information0, meta={"ID": ID}, callback=self.parse0)  # 去爬个人信息
            yield Request(url=url_tweets, meta={"ID": ID}, callback=self.parse2)  # 去爬微博

    def parse0(self, response):
        """ 抓取个人信息1 """
        print("parse0: ",response)
        informationItems = InformationItem()
        selector = Selector(response)
        text0 = selector.xpath('body/div[@class="u"]/div[@class="tip2"]').extract_first()
        if text0:
            num_tweets = re.findall(u'\u5fae\u535a\[(\d+)\]', text0)  # 微博数
            num_follows = re.findall(u'\u5173\u6ce8\[(\d+)\]', text0)  # 关注数
            num_fans = re.findall(u'\u7c89\u4e1d\[(\d+)\]', text0)  # 粉丝数
            if num_tweets:
                informationItems["Num_Tweets"] = int(num_tweets[0])
            print(".....",informationItems)
            if num_follows:
                informationItems["Num_Follows"] = int(num_follows[0])
            if num_fans:
                informationItems["Num_Fans"] = int(num_fans[0])
            informationItems["_id"] = response.meta["ID"]
            # informationItems["LastCrawlTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            url_information1 = "https://weibo.cn/%s/info" % response.meta["ID"]
            yield Request(url=url_information1, meta={"item": informationItems}, callback=self.parse1)

    def parse1(self, response):
        """ 抓取个人信息2 """
        informationItems = response.meta["item"]
        selector = Selector(response)
        text1 = ";".join(selector.xpath('body/div[@class="c"]/text()').extract())  # 获取标签里的所有text()
        nickname = re.findall(u'\u6635\u79f0[:|\uff1a](.*?);', text1)  # 昵称
        gender = re.findall(u'\u6027\u522b[:|\uff1a](.*?);', text1)  # 性别
        place = re.findall(u'\u5730\u533a[:|\uff1a](.*?);', text1)  # 地区（包括省份和城市）
        signature = re.findall(u'\u7b80\u4ecb[:|\uff1a](.*?);', text1)  # 个性签名
        birthday = re.findall(u'\u751f\u65e5[:|\uff1a](.*?);', text1)  # 生日
        sexorientation = re.findall(u'\u6027\u53d6\u5411[:|\uff1a](.*?);', text1)  # 性取向
        marriage = re.findall(u'\u611f\u60c5\u72b6\u51b5[:|\uff1a](.*?);', text1)  # 婚姻状况
        url = re.findall(u'\u4e92\u8054\u7f51[:|\uff1a](.*?);', text1)  # 首页链接

        if nickname:
            informationItems["NickName"] = nickname[0]
        if gender:
            informationItems["Gender"] = gender[0]
        if place:
            place = place[0].split(" ")
            informationItems["Province"] = place[0]
            if len(place) > 1:
                informationItems["City"] = place[1]
        if signature:
            informationItems["Signature"] = signature[0]
        if birthday:
            try:
                birthday = datetime.datetime.strptime(birthday[0], "%Y-%m-%d")
                informationItems["Birthday"] = birthday - datetime.timedelta(hours=8)
            except Exception:
                pass
        if sexorientation:
            if sexorientation[0] == gender[0]:
                informationItems["Sex_Orientation"] = "gay"
            else:
                informationItems["Sex_Orientation"] = "Heterosexual"
        if marriage:
            informationItems["Marriage"] = marriage[0]
        if url:
            informationItems["URL"] = url[0]
        yield informationItems

    def parse2(self, response):
        """ 抓取微博数据 """
        selector = Selector(response)
        tweets = selector.xpath('body/div[@class="c" and @id]')
        ctime = datetime.now()
        crawl = True
        # lastCrawlTime = self.get_lastWeiboDate(response.meta["ID"])
        for tweet in tweets:
            tweetsItems = TweetsItem()
            id = tweet.xpath('@id').extract_first()  # 微博ID

            content=""  # 微博内容
            for c in tweet.xpath('div/span[@class="ctt"]/text() | div/span[@class="ctt"]/a/text()'
                                 '| div/a/text() | div/text()'):
                content += c.extract()  # 微博内容
            content = content.split("\xa0")[0].replace("#","").replace("全文","")
            cooridinates = tweet.xpath('div/a/@href').extract_first()  # 定位坐标
            like = re.findall(u'\u8d5e\[(\d+)\]', tweet.extract())  # 点赞数
            transfer = re.findall(u'\u8f6c\u53d1\[(\d+)\]', tweet.extract())  # 转载数
            comment = re.findall(u'\u8bc4\u8bba\[(\d+)\]', tweet.extract())  # 评论数
            others = tweet.xpath('div/span[@class="ct"]/text()').extract_first()  # 求时间和使用工具（手机或平台）

            tweetsItems["ID"] = response.meta["ID"]
            tweetsItems["_id"] = response.meta["ID"] + "-" + id
            if content:
                tweetsItems["Content"] = content.strip(u"[\u4f4d\u7f6e]") # 去掉最后的"[位置]"
            if cooridinates:
                cooridinates = re.findall('center=([\d|.|,]+)', cooridinates)
                if cooridinates:
                    tweetsItems["Co_oridinates"] = cooridinates[0]
            if like:
                tweetsItems["Like"] = int(like[0])
            if transfer:
                tweetsItems["Transfer"] = int(transfer[0])
            if comment:
                tweetsItems["Comment"] = int(comment[0])
            if others:
                others = others.split(u"\u6765\u81ea")
                others[0]=others[0].replace(u"\xa0",'')
                if "分钟前" in others[0]:
                    tweetsItems["PubTime"] = (datetime.now()-timedelta(minutes=int(others[0].split("分钟前")[0]))).strftime("%Y-%m-%d %H:%M:%S")
                elif "今天" in others[0]:
                    tweetsItems["PubTime"] = others[0].replace("今天",datetime.now().strftime("%Y-%m-%d"))+":00"
                elif "月" in others[0]:
                    tweetsItems["PubTime"] = (str(datetime.now().year)+"-"+others[0]+":00").replace("月","-").replace("日",'')
                else:
                    tweetsItems["PubTime"] = others[0]
                # if len(others) == 2:
                #     tweetsItems["Tools"] = others[1]
            # 提取hashtag
            tweetsItems["Label"] = tweet.xpath('div/span[@class="ctt"]/a/text()').extract_first()  # hashtag
            ctime = datetime.strptime(tweetsItems["PubTime"], "%Y-%m-%d %H:%M:%S")
            if self.yetNotCrawled(tweetsItems["_id"])==False:
                crawl = False
                break
            yield tweetsItems
        url_next = selector.xpath(
            u'body/div[@class="pa" and @id="pagelist"]/form/div/a[text()="\u4e0b\u9875"]/@href').extract()
        if url_next and crawl and self.tweetTimeDelta(ctime)<=30:
            yield Request(url=self.host + url_next[0], meta={"ID": response.meta["ID"]}, callback=self.parse2)

    def parse3(self, response):
        """ 抓取关注或粉丝 """
        items = response.meta["item"]
        selector = Selector(response)
        text2 = selector.xpath(
            u'body//table/tr/td/a[text()="\u5173\u6ce8\u4ed6" or text()="\u5173\u6ce8\u5979"]/@href').extract()
        for elem in text2:
            elem = re.findall('uid=(\d+)', elem)
            if elem:
                response.meta["result"].append(elem[0])
                ID = int(elem[0])
                if ID not in self.finish_ID:  # 新的ID，如果未爬则加入待爬队列
                    self.scrawl_ID.add(ID)
        url_next = selector.xpath(
            u'body//div[@class="pa" and @id="pagelist"]/form/div/a[text()="\u4e0b\u9875"]/@href').extract()
        if url_next:
            yield Request(url=self.host + url_next[0], meta={"item": items, "result": response.meta["result"]},
                          callback=self.parse3)
        else:  # 如果没有下一页即获取完毕
            yield items

    def get_start_urls(self):
        ''' 读取要抓取的微博账号 '''
        ids = []
        with open("uids/traffic_related_uids.txt", "r") as f:
            lines = f.readlines()
            for id in lines:
                ids.append(id.split(":")[0])
            f.close()
        return ids

    def tweetTimeDelta(self, time):
        return (datetime.now()-time).days

    def yetNotCrawled(self,wbid):
        try:
            sina = pymongo.MongoClient()['Sina']
            cur = sina['Information'].find({"_id":wbid})
            return cur.count()==0
        except:
            return True  # 继续抓
