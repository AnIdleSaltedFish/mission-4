import numpy as np
# np.set_printoptions(threshold=np.inf)
import pandas as pd
import netCDF4 as nc
import xarray as xr
import unicodeit

import math

import matplotlib
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import cnmaps
from cnmaps import get_adm_maps, draw_map
from cnmaps.sample import load_dem

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader

import shapefile
import geopandas as gpd
from shapely import geometry
import os
from scipy.stats import pearsonr

# 将一维列表按对应数值分割成不等长的子列表
def split_list_by_sizes(lst, sizes):
    start = 0
    sublists = []
    for size in sizes:
        end = start + size
        sublist = lst[start:end]
        sublists.append(sublist)
        start = end
    return sublists

# scatter数据处理
file2 = r'D:\huankepeixun\mission 2\mission 2 data-DailyConc_China_CNEMC_EachStation_2013to2021_nc\Daily_OBS_Concs4EachStation_Y2013toY2021.nc'
dataset2 = nc.Dataset(file2)
# dict_keys(['char_year', 'char_species', 'char_std_index', 'std_lat', 'std_lon', 'daily_conc'])9，366，8，1674
lats2 = dataset2.variables['std_lat'][:]
lons2 = dataset2.variables['std_lon'][:]
concs2 = dataset2.variables['daily_conc'][1:7, :, 0, :]
lats_data2 = np.array(lats2)
lons_data2 = np.array(lons2)
concs_data2 = np.array(concs2)
concs_data2 = np.where(concs_data2 == -999, np.nan, concs_data2)
sub_arrays = np.split(concs_data2, 6, axis=0)
all_data = []
reshaped_subarrays = [sub_array.reshape(366, 1674) for sub_array in sub_arrays]  # 将8个(1,366,1674)转换成9个(366,1674)
for i, sub_array in enumerate(reshaped_subarrays):
    last_list = [np.nanmean(x) for x in zip(*sub_array)]  # 在每一个（366,1674）跑的过程中，计算每个站点的年平均，即二维按列求和
    all_data.append(last_list)
all_data = np.array(all_data)
# print(all_data.shape)  (6, 1674)
# 6*1674是形状规整的数组，包含nan值

# 统计每年有效站点个数，统计nan个数，判断并统计nan有专门的函数，不需要自己写条件
nan_counts = np.isnan(all_data).sum(axis=1)
print(nan_counts)
num = 1674 - nan_counts
print(num)

# 将站点数据二维展开成一维，去掉nan值，再按每年的有效站点个数分割成不等长的子列表（二维np数组只能整行整列删除，无法删除单个值;并且np数组形状是规则的，列表形状才可以不规则）
data_2 = all_data.ravel()
data_2 = data_2[~np.isnan(data_2)]  # 展开成一维并删去nan值需要以np数组形式
data_2 = data_2.tolist()  # 这两行可以合并data_2 = data_2[~np.isnan(data_2)].tolist()
# 8275
sublists = split_list_by_sizes(data_2, num)  # 不等长分割需要以List形式，因此需要上步tolist
for sublist in sublists:
    print(sublist)
    print(len(sublist))

# contourf数据处理
file3 = r'D:\huankepeixun\mission 3\TAP_Daily_PM25_Y2001toY2020\TAP_Daily_PM25_Y2001toY2020.nc'
dataset3 = nc.Dataset(file3)
# dict_keys(['latitude', 'longitude', 'Daily_PM25'])
# latitude(450),Latitude from 15.05 to 59.95 by 0.1
# longitude(700),Longitude from 70.05 to 139.95 by 0.1
# float32 Daily_PM25(year, day, lat, lon)(20, 366, 450, 700)
lats3 = dataset3.variables['latitude'][:]
lons3 = dataset3.variables['longitude'][:]
lats_data3 = np.array(lats3)
lons_data3 = np.array(lons3)
concs3 = dataset3.variables['Daily_PM25'][13:19, :, :, :]
concs3 = np.array(concs3)
concs3 = np.where(concs3 == -999, np.nan, concs3)
data3 = np.nanmean(concs3, axis=1)

# 提取对应最近格点的数据
m = 0  # 年份的累加
y = []
for data in data3:
    yi = []
    for i in range(1674):
        if np.isnan(all_data[m, i]):
            continue
        else:
            lat0 = lats2[i]
            lon0 = lons2[i]

            delta_lat = []
            delta_lon = []

            # 使用掩码来获取最小值的索引，不使用掩码的话，nan值会跳过，索引值并不是真正的值，而是删去nan后新序列的索引。
            for lat in lats_data3:
                deltalat = np.fabs(lat - lat0)
                delta_lat.append(deltalat)
            delta_lat = np.array(delta_lat)
            deltalat_mask = ~np.isnan(delta_lat)
            filtered_lat = delta_lat[deltalat_mask]
            minlat_value_filtered = np.min(filtered_lat)
            lat_loc = np.where(delta_lat == minlat_value_filtered)

            for lon in lons_data3:
                deltalon = np.fabs(lon - lon0)
                delta_lon.append(deltalon)
            delta_lon = np.array(delta_lon)
            deltalon_mask = ~np.isnan(delta_lon)
            filtered_lon = delta_lon[deltalon_mask]
            minlon_value_filtered = np.min(filtered_lon)
            lon_loc = np.where(delta_lon == minlon_value_filtered)

            ds = data3[m, lat_loc, lon_loc]
            yi.append(ds)
    # print(len(yi)) 935, 1480, 1464, 1426, 1492, 1478
    yi = [item[0][0] for item in yi]  # 列表的元素输出后显示[array([[80.92473]], dtype=float32)]转化为纯数值
    m = m + 1
    y.append(yi)
for x in y:
    print(x)
    print(len(x))

# 计算nmb和ioa
nmb_list = []
ioa_list = []
for i in range(6):
    # list没有求和函数，基本上不存在可以直接进行数学计算的工具，常用的功能基本上都在np数组上
    y_np = np.array(y[i])
    sublist_np = np.array(sublists[i])
    num_np = np.array(num)
    nmb = (np.nansum(y_np) - np.nansum(sublist_np)) / num_np[i] / np.nanmean(sublist_np)
    nmb_list.append(nmb)
    # nmb = (y_np.sum()-sublist_np.sum())/num_np[i]/sublist_np.mean()  显示nan，虽然不知道为啥，所以都改用nansum,nanmean函数了

    y_no_nan = np.nan_to_num(y_np)  # np.nan_to_num()函数，将nan转换成0
    sub_no_nan = np.nan_to_num(sublist_np)  # 这两步换成0是为了解决，这个错误：array must not contain infs or NaNs
    r, p = pearsonr(y_no_nan, sub_no_nan)
    ioa_list.append(r)
print(nmb_list)
print(ioa_list)

# 站点数据是sublists，，格点数据是y
# 画图部分在代码试验
#'Arial Unicode MS','Microsoft MHei'
#  画图
# 设置标题和字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.family'] = 'Arial Unicode MS'  # 控制没有特殊说明的位置的字体，如刻度标签，下同
plt.rcParams['font.size'] = 13
plt.rcParams['font.weight'] = 'bold'
fig = plt.figure(figsize=(33, 20))
axes_main = fig.subplots(2, 3, sharex=True, sharey=True)  # 共享刻度坐标
'''fig.suptitle('Space Distribution of PM2.5', fontsize=20, weight='bold', y=0.94)'''  # 设置大图标题,y用来控制大标题的相对位置
font2 = {'size': 12, 'family': 'SimHei', 'weight': 'normal'}
label_font = {'size': 12, 'family': 'Arial Unicode MS', 'weight': 'bold'}
year = 2014
k = 0  # 图中文字1-6
zimu = ['a','b','c','d','e','f']
x1 = np.linspace(0,100,100)
x2 = np.linspace(0,200,100)
x3 = np.linspace(0,200,100)
y1 = 2*x1
y2 = 0.5*x2
y3 = x3
for i in range(2):
    for j in range(3):
        # 绘制散点图
        sc = axes_main[i, j].scatter(sublists[k], y[k], c='red', s=8, edgecolor='k', linewidths=0.1,  zorder=1)
        # 三条蓝线
        axes_main[i, j].scatter(x1, y1, c='blue', s=0.3, zorder=2)
        axes_main[i, j].scatter(x2, y2, c='blue', s=0.3, zorder=2)
        axes_main[i, j].scatter(x3, y3, c='blue', s=0.3, zorder=2)
        # 坐标轴及刻度设置
        axes_main[i, j].tick_params(top=False, bottom=True, left=True, right=False)
        axes_main[i, j].tick_params(axis='both', which='major', direction='out', width=1, length=6, labelsize=10)
        axes_main[i, j].set_xticks([0, 50, 100, 150, 200])  # 哪些值标刻度
        axes_main[i, j].set_yticks([0, 50, 100, 150, 200])
        axes_main[i, j].set_xlim(0, 200)  # 在原点处开始标刻度值
        axes_main[i, j].set_ylim(0, 200)
        # 坐标轴名称
        if j == 0:
            axes_main[i, j].set_ylabel('TAP PM2.5 (μg m$^{-3}$)', fontdict=label_font, style='italic')
            # axes_main[i, j].set_ylabel('CNEMC MDA8 O$_3$ (μg m$^{-3}$)', fontdict=label_font, style='italic')
        if i == 1:
            axes_main[i, j].set_xlabel('CNEMC PM2.5 (μg m$^{-3}$)', fontdict=label_font, style='italic')
            # axes_main[i, j].set_xlabel('TAP MDA8 O$_3$ (μg m$^{-3}$)', fontdict=label_font, style='italic')
        # 左上角右下角文字信息
        axes_main[i, j].text(0.03, 0.94, f'year : {year}', fontsize=12, style='italic',weight='bold', transform=axes_main[i, j].transAxes)
        axes_main[i, j].text(0.03, 0.88, f'station : {num[k]}', fontsize=12, style='italic', weight='bold', transform=axes_main[i, j].transAxes)
        axes_main[i, j].text(0.03, 0.82, f'NMB : {nmb_list[k]:.2f}', fontsize=12, style='italic', weight='bold', transform=axes_main[i, j].transAxes)
        axes_main[i, j].text(0.03, 0.76, f'IOA : {ioa_list[k]:.2f}', fontsize=12, style='italic', weight='bold', transform=axes_main[i, j].transAxes)
        axes_main[i, j].text(0.90, 0.05, f'({zimu[k]})', fontsize=12, style='italic', weight='bold', transform=axes_main[i, j].transAxes)
        # axes_main[i, j].text(0, 1.1, f'{year}', bbox=dict(facecolor='k', alpha=1, pad=20),  ha='center', va='center', fontsize=8, color='white', transform=axes_main[i, j].transAxes)
        k = k + 1
        year = year + 1
plt.show()





































































