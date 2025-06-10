import os
import time

#
#参数:
#   img_path: 要识别的图片的路径
#
#返回:
#   返回结果为各赛题中要求的识别结果，具体格式可参考提供压缩包中的 “图片对应输出结果.txt” 中一张图片对应的结果
#
def process_img(img_path):
    pass

#
#以下代码仅作为选手测试代码时使用，仅供参考，可以随意修改
#但是最终提交代码后，process.py文件是作为模块进行调用，而非作为主程序运行
#因此提交时请根据情况删除不必要的额外代码
#
if __name__=='__main__':
    imgs_folder = './imgs/'
    img_paths = os.listdir(imgs_folder)
    def now():
        return int(time.time()*1000)
    last_time = 0
    count_time = 0
    max_time = 0
    min_time = now()
    for img_path in img_paths:
        print(img_path,':')
        last_time = now()
        result = process_img(imgs_folder+img_path)
        run_time = now() - last_time
        print('result:\n',result)
        print('run time: ', run_time, 'ms')
        print()
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    print('\n')
    print('avg time: ',int(count_time/len(img_paths)),'ms')
    print('max time: ',max_time,'ms')
    print('min time: ',min_time,'ms')
