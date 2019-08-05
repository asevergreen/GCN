# encoding=utf-8

import base64
import requests
import time
import sys
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import logging
import pymongo
from lxml import etree
from yumdama import identify
from items import CookieItem


IDENTIFY = 1  # 验证码输入方式:        1:看截图aa.png，手动输入     2:云打码
COOKIE_GETWAY = 2 # 0 代表从https://login.sina.com.cn/sso/login.php?client=ssologin.js(v1.4.18) 获取cookie   # 1 代表从https://weibo.cn/login/获取Cookie
dcap = dict(DesiredCapabilities.PHANTOMJS)  # PhantomJS需要使用老版手机的user-agent，不然验证码会无法通过
dcap["phantomjs.page.settings.userAgent"] = (
    "Mozilla/5.0 (Linux; U; Android 2.3.6; en-us; Nexus S Build/GRK39F) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1"
)
logger = logging.getLogger(__name__)
logging.getLogger("selenium").setLevel(logging.WARNING)  # 将selenium的日志级别设成WARNING，太烦人


"""
输入你的微博账号和密码，可去淘宝买。
建议买几十个，微博限制的严，太频繁了会出现302转移。
或者你也可以把时间间隔调大点。
"""
myWeiBo = []
#获取爬虫账号
def get_myWeiBo():
    # dir = sys.path[0]
    # print(dir)
    with open("accounts/accounts.txt", "r") as f:
        lines = f.readlines()
        print(lines)
        for line in lines:
            account = line.split(":")
            myWeiBo.append({'no':account[0], "psw":account[1]})
        print(myWeiBo)
    f.close()

# 根据COOKIE_GETWAY变量调用如下type123...
def getCookie(account, password):
    if COOKIE_GETWAY == 0:
        return get_cookie_from_login_sina_com_cn(account, password)
    elif COOKIE_GETWAY ==1:
        return get_cookie_from_weibo_cn(account, password)
    elif COOKIE_GETWAY ==2:
        return get_cookie_from_weibo_cn_driver(account, password)   # 此选项请在调试模式下进行，手动登入后点击下一步获取cookie
    elif COOKIE_GETWAY ==3:
        return {'SUHB': '0nl-zNLTIuJPDo', 'SSOLoginState': '1563625674', '_T_WM': '60048819861', 'SUBP': '0033WrSXqPxfM725Ws9jqgMF55529P9D9Wh_47W2aJynG4J02MDA4Xkz5JpX5KzhUgL.FoqNeKB7ehqceoB2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoM4S0.4Sh2E1KzX', 'MLOGIN': '1', 'M_WEIBOCN_PARAMS': 'uicode%3D20000174', 'WEIBOCN_FROM': '1110006030', 'SUB': '_2A25wN3iaDeRhGeBJ6lYR8CjKyTiIHXVT2BjSrDV6PUJbktAKLUankW1NRk95Xh8kgiRfgxwMgxUXozX4zqJ_B8vC', 'XSRF-TOKEN': 'aeff39'}
    else:
        logger.error("COOKIE_GETWAY Error!")

# type 0: 从新浪通行证获得cookie
def get_cookie_from_login_sina_com_cn(account, password):
    """ 获取一个账号的Cookie """
    loginURL = "https://login.sina.com.cn/sso/login.php?client=ssologin.js(v1.4.18)"
    username = base64.b64encode(account.encode("utf-8")).decode("utf-8")
    postData = {
        "entry": "sso",
        "gateway": "1",
        "from": "null",
        "savestate": "30",
        "useticket": "0",
        "pagerefer": "",
        "vsnf": "1",
        "su": username,
        "service": "sso",
        "sp": password,
        "sr": "1440*900",
        "encoding": "UTF-8",
        "cdult": "3",
        "domain": "sina.com.cn",
        "prelt": "0",
        "returntype": "TEXT",
    }
    session = requests.Session()
    r = session.post(loginURL, data=postData)
    jsonStr = r.content.decode("gbk")
    info = json.loads(jsonStr)
    if info["retcode"] == "0":
        logger.warning("Get Cookie Success!( Account:%s )" % account)
        cookie = session.cookies.get_dict()
        # return json.dumps(cookie)
        return cookie
    else:
        logger.warning("Failed!( Reason:%s )" % info["reason"])
        return ""
# type1: 从selenium-PhantomJS获取cookie
def get_cookie_from_weibo_cn(account, password):
    """ 获取一个账号的Cookie """
    try:
        # browser = webdriver.PhantomJS(desired_capabilities=dcap)
        browser = webdriver.PhantomJS(desired_capabilities=dcap)
        browser.get("https://weibo.cn/login/")
        time.sleep(1)

        failure = 0
        while "微博" in browser.title and failure < 5:
            failure += 1
            browser.save_screenshot("aa.png")
            username = browser.find_element_by_name("mobile")
            username.clear()
            username.send_keys(account)

            psd = browser.find_element_by_xpath('//input[@type="password"]')
            psd.clear()
            psd.send_keys(password)
            try:
                code = browser.find_element_by_name("code")
                code.clear()
                if IDENTIFY == 1:
                    code_txt = input("请查看路径下新生成的aa.png，然后输入验证码:")  # 手动输入验证码
                else:
                    from PIL import Image
                    img = browser.find_element_by_xpath('//form[@method="post"]/div/img[@alt="请打开图片显示"]')
                    x = img.location["x"]
                    y = img.location["y"]
                    im = Image.open("aa.png")
                    im.crop((x, y, 100 + x, y + 22)).save("ab.png")  # 剪切出验证码
                    code_txt = identify()  # 验证码打码平台识别
                code.send_keys(code_txt)
            except Exception as e:
                pass

            commit = browser.find_element_by_name("submit")
            commit.click()
            time.sleep(3)
            if "我的首页" not in browser.title:
                time.sleep(4)
            if '未激活微博' in browser.page_source:
                print('账号未开通微博')
                return {}

        cookie = {}
        if "我的首页" in browser.title:
            for elem in browser.get_cookies():
                cookie[elem["name"]] = elem["value"]
            logger.warning("Get Cookie Success!( Account:%s )" % account)
        # return json.dumps(cookie)
        return cookie
    except Exception as e:
        logger.warning("Failed %s!" % account)
        return ""
    finally:
        try:
            browser.quit()
        except Exception as e:
            pass
# type2: 调试模式下手动登录获取cookie【一般不采用】
def waitElePresent(driver, eid):
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, eid)))
    except Exception as e:
        print(e)
def get_cookie_from_weibo_cn_driver(account, password):
    """ 获取一个账号的Cookie """
    try:
        # browser = webdriver.PhantomJS(desired_capabilities=dcap)
        browser = webdriver.Chrome()
        browser.maximize_window()
        browser.implicitly_wait(20)
        browser.get("https://weibo.cn/login/")
        waitElePresent(browser, "loginName")
        browser.find_element_by_id("loginName").send_keys(account)
        waitElePresent(browser, "loginPassword")
        browser.find_element_by_id("loginPassword").send_keys(password)
        waitElePresent(browser, "loginAction")
        browser.find_element_by_id("loginAction").click()
        time.sleep(3)
        # 需要登录验证
        if("https://security" in browser.current_url):
            waitElePresent(browser, "app")
        cookie = {}
        for elem in browser.get_cookies():
            cookie[elem["name"]] = elem["value"]
        logger.warning("Get Cookie Success!( Account:%s )" % account)
        # return json.dumps(cookie)
        return cookie
    except Exception as e:
        print(e)
        logger.warning("Failed %s!" % account)
        return ""
    finally:
        try:
            browser.quit()
        except Exception as e:
            pass


# 为所有爬虫账号获得cookie
def getCookies(weibo):
    """ 获取Cookies """
    cookies = []
    for elem in weibo:
        account = elem['no']
        password = elem['psw']
        cookie = mycookies.find({"_id":account})
        logger.warning("get from db: %d" % cookie.count())
        if cookie.count() > 0:
            cookie = cookie[0]['cookie_value']
        else:
            cookie = None
        # 更新失效的cookie
        if cookie is None or cookie_invalid(cookie, account):
            cookie  =  getCookie(account, password)
            if cookie != None:  # 获取cookie成功
                cookies.append(cookie)
                store_my_cookie(account, cookie)
        # 使用db中的cookie
        else:
            cookies.append(cookie)
    return cookies

# 将cookie存在数据库中重复利用，据百度新浪cookie有效期为7天少8h
def store_my_cookie(id, cookie):
    citem = CookieItem()
    citem["_id"] = id
    citem["cookie_value"] = cookie
    mycookies.insert_one(citem)
    # print('store this cookie!')
# getCookies()方法使用前，先从db中取cookie，使用该方法验证cookie有效性
def cookie_invalid(cookie, accout):
    url = 'https://weibo.cn/%s/info' % accout
    html = requests.get(url, cookies=cookie).content
    selector = etree.HTML(html)
    # print(html, selector.xpath('//title/text()')[0])
    if "登录" in selector.xpath('//title/text()')[0]:
        mycookies.delete_one({"_id":accout})
        return True
    return False




"""
 Cookie db
"""
client = pymongo.MongoClient("localhost", 27017)
db = client["Sina"]
mycookies = db["MyCookies"]
'''
get or update cookies
'''
get_myWeiBo()   # 爬虫账号
cookies = getCookies(myWeiBo)   # cookies
logger.warning("Get Cookies Finish!( Num:%d)" % len(cookies))
print(cookies)



# USE　for DEBUG
if __name__ == '__main__':
    print(cookie_invalid("", str(5865526492)))
    print(cookie_invalid(cookies[0], str(5865526492)))
