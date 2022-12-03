import math
import pandas as pd
import geopandas as gpd
import json
import collections
import sys
import numpy as np
import networkx as nx
from pyqubo import Array
import neal

sys.setrecursionlimit(100000) #这里设置为十万


# path = 'F:/data_lu_0510_practice/arcgis10.6_0531/CUG_lu_data_merge_triangle/CUG_BIG/CUG_big_small/'
path = "./"
'''
1、所有小三角形的相邻三角形，类（三角形id (相邻三角形id1，2，3) 边长大小）
   相邻三角形的判定逻辑，如果有一个三角形的两个点坐标都和当前三角形一致，就是相邻三角形
2、冲突集合单独列出，新的三角形的三个顶点（顶点是不重复的那个点）
（a：b,c,d）
'''
def calculate_triangle_xy(path_select):
    arr_boundary = []
    f = gpd.read_file(path_select)      # 读取来的坐标是四个点，但是最后一个点和第一个点是一样的
    arr_triangle = []
    k = f.shape[0]
    polygon = f.geometry.to_json()
    polygon_dict = json.loads(polygon)
    # 转换为json提取坐标
    for i in range(0,k):
        line_1 = polygon_dict["features"][i]["geometry"]["coordinates"][0]
        arr_triangle.append(line_1)
    '''
        for line in boundary:
        arr_boundary.append(str(line).split(','))
    '''
    return arr_triangle

def intersection(arr_input1,arr_input2):
    # 传个二维数组进来
    count = 0
    for i in range(0, len(arr_input1)-1):
        for j in range(0, len(arr_input2)-1):
             if arr_input1[i] == arr_input2[j]:
                count += 1
    if count == 2:
        return True
    else:
        return False


class Triangle:
    # 三角形类构造方法
    def __init__(self, neighbor,id,r):
        # 三角形具有基本的属性，即id和半径
        self.id = id
        self.neighbor = neighbor
        self.r = r

# 实例化这个类,我为什么要创建一个类呢，不是很懂
b = Triangle(1,(2,3,4),7.9)
class Solution:
    def removeDuplicates(self, nums) :
        # 对列表进行循环修改时要使用nums[:]而不是nums
        for n in nums:
            if nums.count(n) > 1:
                nums.remove(n)
        if len(nums) > 6:
            Solution.removeDuplicates(self,nums)
        else:
            return nums


def calculate_neighbor(name_shp,r):
    arr_triangle = []           # 先验证一下逻辑,三角形重心坐标的距离计算是否相邻
    arr_triangle_1 = {}
    path_triangle_mix = path + name_shp
    df1 = gpd.GeoDataFrame.from_file(path_triangle_mix)

    arr_triangle = calculate_triangle_xy(path_triangle_mix)               # 传一个完整的路径进来，获得三角形的坐标数据
    for i in range(len(arr_triangle)):
        a = df1.area[i]
        b = round(math.sqrt(4*a/math.sqrt(3)), 2)
        if b == r:                                                 # 如果是小三角形的话，这个R属性不确定存在还是不存在，一会读数据的时候看看1123
            for j in range(len(arr_triangle)):
                if intersection(arr_triangle[i], arr_triangle[j]):         # 大三角形不可能和小三角形两个点重复,并且两个相同的点也不可能只有两个坐标重复
                    arr_triangle_1.setdefault(i,[]).append(j)
    return arr_triangle_1


def calculate_conflict(dict_neighbor):
    conflict_dict = {}                                           # 计算冲突数组的数据格式：字典：键值对，创建一个空字典
    for k1, value1 in dict_neighbor.items():
        for k2, value2 in dict_neighbor.items():
            a = len(list(set(value1).intersection(set(value2))))
            if a == 1 or a == 2:                                # 1个或2个相交，三个相交的话就是本身了
                conflict_dict.setdefault(k1, []).append(k2)
    return conflict_dict


'''
    # 判断是否还有构成更大三角形的可能，首先，周围要有三个三角形
    # 判断三角形是正立还是倒立三角形
    # 其次Y=**下一条直线，并且x 在范围内的点有5个
'''
def calculate_merge(dict_neighbor, name_shp):
    arr_triangle = []
    path_triangle_mix = path + name_shp
    arr_triangle = calculate_triangle_xy(path_triangle_mix)
    # 针对一组的进行合并进行处理
    result = []
    for k, value in dict_neighbor.items():
        result_value = []
        arr_triangle_big = []
        arr_triangle_big_new = []
        big_xy = []
        if len(value) == 3:    # 有三个相邻的三角形再去计算扩展可能性
           arr_triangle_big_new1 = []
           for z in value:              # 判断是正三角形还是倒三角形,把三角形的所有坐标加进来
               arr_triangle_big.append(arr_triangle[k])
               arr_triangle_big.append(arr_triangle[z])   # 这个地方的逻辑是加入上述编号的三角形
           for h in arr_triangle_big:
               for g in h:
                   arr_triangle_big_new.append(g)
           s = Solution()   # 去除重复值[x,y]完全相同的值被去除了
           s.removeDuplicates(arr_triangle_big_new)
           x = []
           y = []
           for i in arr_triangle_big_new:
               x.append(i[0])
               y.append(i[1])
           max_x = max(x)
           min_x = min(x)
           max_y = max(y)
           min_y = min(y)
            # 在这个地方可以获得大三角形的坐标
           a = collections.Counter(x)
           b = a.most_common(1)  # 获取出现次数最多的元素
           c = collections.Counter(y)
           d = c.most_common(1)[0][0]  # 获取出现次数最多的元素
           e = c.most_common(3)[2][0]
           line_bigbig = []
           if d == min_y:  # 正三角形，去遍历Polygon，向下去找至少有一个点能带入y = * 的方程并且数量==5
                big_xy.append([min_x,min_y])
                big_xy.append([max_x,min_y])
                big_xy.append([(min_x+max_x)/2,max_y])
               # result_value.append(big_xy)
                # result.append(big_xy)
                flag = 0
                # 这个地方改一下判断逻辑
                for mm in range(len(arr_triangle)):
                    count = 0
                    merge_tri_x1 = []
                    merge_tri_y1 = []
                    for nn in range(len(arr_triangle[mm])-1):
                        merge_tri_x1.append(arr_triangle[mm][nn][0])
                        merge_tri_y1.append(arr_triangle[mm][nn][1])
                    max_merge_x = max(merge_tri_x)
                    min_merge_x = min(merge_tri_x)
                    max_merge_y = max(merge_tri_y)
                    min_merge_y = min(merge_tri_y)
                    if max_merge_y == min_y and min_x <= (max_merge_x + min_merge_x) / 2 <= max_x:  # 三个点作为一组去考虑
                        count += 1
                    if count > 0:  # 统计有多少个至少有一个点在那条线上的数量
                        line_bigbig.append(mm)
                        flag += 1
                if flag == 5:
                    result.append(k)
                    result.append(big_xy)
                    result.append(line_bigbig)
                   # result_value.append(line_bigbig)
                    # result.append(line_bigbig)
                   # result.setdefault(k,[]).append(result_value)


           elif e == min_y:  # 倒三角形
                big_xy.append([min_x, max_y])
                big_xy.append([max_x, max_y])
                big_xy.append([(max_x+min_x)/2, min_y])
                # result_value.append(big_xy)
                flag = 0
                line_bigbig = []
                for mm in range(len(arr_triangle)):
                    merge_tri_x = []
                    merge_tri_y = []
                    for nn in range(len(arr_triangle[mm])-1):
                        merge_tri_x.append(arr_triangle[mm][nn][0])
                        merge_tri_y.append(arr_triangle[mm][nn][1])
                    max_merge_x = max(merge_tri_x)
                    min_merge_x = min(merge_tri_x)
                    max_merge_y = max(merge_tri_y)
                    min_merge_y = min(merge_tri_y)
                    count = 0
                    if min_merge_y == max_y and min_x <= (max_merge_x+min_merge_x)/2 <= max_x:    #三个点作为一组去考虑
                        count += 1
                    if count > 0:  # 统计有多少个至少有一个点在那条线上的数量
                        line_bigbig.append(mm)
                        flag += 1
                if flag == 5:
                   # result_value.append(line_bigbig)
                    result.append(k)
                    result.append(big_xy)
                    result.append(line_bigbig)

                    # result.append(line_bigbig)
                    '''
                    result.setdefault(k,[]).append(np.array(result_value))     # 生成一个目前大的三角形的编号
                    '''
    return result

'''
基于冲突集的字典，计算最大团，最大团中的每一个节点（元素）就是最终被选择用于合并的三角形
'''
def solve_the_maximum_clique(conflict_dict):
    conflict_graph = nx.Graph()
    for key in conflict_dict:
        for value in conflict_dict[key]:
            conflict_graph.add_edge(key, value)
    graph_nodes = list(set(conflict_graph.nodes))
    variable_dict = {}
    variable_dict_reverse = {}
    for i in range(len(graph_nodes)):
        variable_dict[graph_nodes[i]] = i
        variable_dict_reverse[i] = graph_nodes[i]
    variables = Array.create("t_cell", shape=len(graph_nodes), vartype="BINARY")
    Hamiltonian = 0
    for node in conflict_graph.nodes:
        Hamiltonian -= variables[variable_dict[node]]*variables[variable_dict[node]]
    for (u,v) in conflict_graph.edges:
        Hamiltonian += 2*variables[variable_dict[u]]*variables[variable_dict[v]]
    model = Hamiltonian.compile()
    bqm = model.to_bqm()

    solver = neal.SimulatedAnnealingSampler()
    sampleset = solver.sample(bqm,num_reads=100)
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample = min(decoded_samples, key=lambda s: s.energy)
    # print(best_sample)
    # return conflict_graph, variable_dict, variable_dict_reverse, best_sample
    clique = decode_clique_result(best_sample, variable_dict_reverse)
    return clique

'''
支撑上一个函数的解码
'''
def decode_clique_result(result, reverse_dict):
    clique = []
    for key in result.sample.keys():
        if result.sample[key] == 1:
            index = int(key.split("[")[1].split("]")[0])
            clique.append(reverse_dict[index])
    return clique
            
        
if __name__ == "__main__":
    dict_neighbor = {}
    dict_conflict = {}
    result = {}
    dict_neighbor = calculate_neighbor('ArcGISresult.shp', 12.99)   # r 是小三角形的边长

    dict_conflict = calculate_conflict(dict_neighbor)
    clique = solve_the_maximum_clique(dict_conflict)
    
    # result = calculate_merge(dict_neighbor,'ArcGISresult.shp')

    # print(result)










