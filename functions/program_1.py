# -*- coding: utf-8 -*-
# @Time : 2022/11/21 23:27
# @Author : Zph
# @FileName: program_1.py
# @Software: PyCharm
# @Github ：https://github.com/hzz12138

"""
    program 1:函数运行计时器的装饰器实现
"""

import time


def timer(func):
    def wrapper(*args, **kwargs):  # *args代表任意多个无名参数 **关键词参数
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        finish_time = time.perf_counter()
        cost_time = finish_time - start_time
        if cost_time < 60:
            print(f"""func:{func.__name__}, took time:{'%.4f' % cost_time} seconds""")
        elif cost_time < 3600:
            minutes = (cost_time % 3600) // 60
            seconds = (cost_time % 60)
            print(f"""func:{func.__name__}, took time:{'%d' % minutes} minutes,{'%.1f' % seconds} seconds""")
        else:
            hours = cost_time // 3600
            minutes = (cost_time % 3600) // 60
            seconds = (cost_time % 60)
            print(
                f"""func:{func.__name__}, took time:{'%d' % hours} hours,{'%d' % minutes} minutes,{'%.1f' % seconds} seconds""")
        return result

    return wrapper


@timer
def waste1(num):
    @timer
    def waste_some_time(num):
        my_list = []
        for i in range(num):
            my_list.append(i)
        return my_list

    arr = waste_some_time(num)
    return arr


if __name__ == '__main__':
    # arr = waste_some_time(1000000)
    # print(arr)
    arr1 = waste1(1000000)
