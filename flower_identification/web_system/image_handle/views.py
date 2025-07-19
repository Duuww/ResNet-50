import os
from django.conf import settings
from django.shortcuts import render
from .models import ImageCheck
from utils import restful
from utils.image_check import check_handle
from web_system.settings import admin_title, index_info
import time
from .forms import MyModelForm

import urllib.request
import urllib.parse
from lxml import etree

def index(request):
    context = {
        'title': admin_title,
        'index_info': index_info
    }

    return render(request, 'index.html', context=context)

def check(request):
    return render(request, 'check.html')


def upload_img(request):
    # 图片上传
    file = request.FILES.get('file')
    print(file.name)
    file_name = file.name
    file_name = '{}.{}'.format(int(time.time()), str(file_name).rsplit('.')[-1])
    with open(os.path.join(settings.MEDIA_ROOT, file_name), 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)
    upload_url = request.build_absolute_uri(settings.MEDIA_URL + file_name)
    ImageCheck.objects.create(file_name=file_name, file_url=upload_url)
    return restful.ok(data={'url': upload_url})


def check_img(request):
    # 图片检测
    image_url = request.POST.get('img_url')
    if not image_url:
        return restful.params_error(message='缺少必要的参数image_url')
    image_name = image_url.rsplit('/')[-1]
    image_path = os.path.join(settings.MEDIA_ROOT, image_name)
    pred_name = check_handle(image_path)
    know = query(pred_name)

    # 保存预测结果到数据库模型中
    obj = ImageCheck.objects.filter(file_name=image_name).last()
    obj.check_result = pred_name
    obj.save()
    return restful.ok(data={'pred_name': pred_name, 'query': know})

def query(content):
    # 请求地址
    url = 'https://baike.baidu.com/item/' + urllib.parse.quote(content)
    # 请求头部
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
    }
    # 利用请求地址和请求头部构造请求对象
    req = urllib.request.Request(url=url, headers=headers, method='GET')
    # 发送请求，获得响应
    response = urllib.request.urlopen(req)
    # 读取响应，获得文本
    text = response.read().decode('utf-8')
    # 构造 _Element 对象
    html = etree.HTML(text)
    # 使用 xpath 匹配数据，得到匹配字符串列表
    sen_list = html.xpath('//div[contains(@class,"J-summary")] /div /span/text()')
    # 过滤数据，去掉空白
    sen_list_after_filter = [item.strip('\n') for item in sen_list]
    # 将字符串列表连成字符串并返回
    return ''.join(sen_list_after_filter)

