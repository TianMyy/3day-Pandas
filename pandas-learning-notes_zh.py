Pandas

1.1 pandas基本介绍
	pandas基本数据结构:
		1. series:一维数组，能保存不同种数据类型，类似Numpy中的array和基本结构中的List。
		2. dataframe:二维的表格型数据结构，类似R中的data.frame，可理解为series的容器。
	1.1.1 Series类型
		一维series可以用一维列表初始化
			s=pd.Series([1, 3, 5, np.nan, 6, 8])
			print(s)
		默认情况下series的下标都是数字(也可以使用额外参数指定)，类型统一
			s=pd.Series([1, 3, 5, np.nan, 6, 8], index=['a', 'b', 'c', 'd', 'e', 'f'])
		索引--数据的行标签
			s.index
		值
			s.values/ s[0]
		切片操作
			s[2: 5] - 不包含索引5
			s[::2] - 从头取到尾，每次索引+2
		索引赋值
			s.index.name='索引' - '索引'成为index列的列名
			s.index=list('abcdef') - 也是将索引变为abcdef
			依然可以切片操作 s['a' : 'c'] - 两边都包含取到
	1.1.2 DataFrame类型
		1）
		dataframe是个二维结构，这里首先构造一维时间序列，作为第一维的下标:
			date = pd.date_range('20200101', periods=6) - 2020010 ~ 20200106
		创建一个dataframe：
			df=pd.DataFrame(np.random.randn(6, 4), index=date, column=list('ABCD')) - 定义了索引名和列名
		除了向dataframe中传递二维数组，也可以使用字典传入数据：
			字典的每个key代表一列，其value可以是各种能够转化为series的对象
			与series要求所有类型都一致不同，dataframe只要求每一列数据的格式相同
			df=pd.DataFrame({'A':1., 'B':pd.Timestamp('20200101'), 'C':pd.Series(1, index=list(range(4)), dtype=float), 
							 'D':np.array([3] * 4, dtype=int), 'E':pd.Categorical(['test', 'train', 'test', 'train']), 'F':'abc'})
		2）
		查看数据
			df.head()/ df.tail()/ df.head(3)/ df.tail(4)
		查看各列的数据类型
			df.dtypes
		查看各列的列标
			df.columns
		查看下标
			df.index
		查看数据值
			df.values

1.2 pandas读取数据及数据操作
	--将以豆瓣的电影数据作为示例(codes在'pandas-1.ipynb')
	读取数据
		df=pd.read_excel('...')/ df=pd.read_csv('...')
	行操作
		df.iloc[0]
		df.iloc[0: 5] - 左闭右开
		df.loc[0: 5] -非左闭右开
		添加一行	 -	dit={....}
					s=pd.Series(dit)
					s.name=38738
					df=df.append(s)
		删除一行	 -	df=df.drop([38738]) -> 38738是该行的索引
	列操作
		df.columns
		df['名字'][:5] - 看到'名字'这一列前五个
		df[['名字', '类型']] - 看到多列的数据
		增加一列	 -	df['序号']=range(1, len(df)+1) -> range()是左开右闭
		删除一列	 -	df=df.drop('序号', axis=1) -> axis=0指的行，axis=1指的列 
	通过标签选择数据
		df.loc[1, '名字'] - 取第二行的'名字'列
		df.loc[[1, 3, 5, 7, 9], ['名字', '评分']] - 取1，3，5，7，9行的'名字'列和'评分'列
	条件选择
		获取产地是美国的电影的数据  - 	df[df['产地']=='美国']
								   	df[df['产地']=='美国'][:5] - 前五行
		获取产地为美国，并且评分大于9的电影	  -	df[(df.产地=='美国') & (df.评分>9)]
		获取产地为美国或者中国，并且评分大于9的电影  -	df[((df.产地=='美国') | (df.产地=='中国大陆')) & (df.评分>9)]

1.3 缺失值和异常值处理
	1.3.1 缺失值处理方法
		 	 方法						说明
			dropna			根据标签中的缺失值进行过滤，删除缺失值	
			fillna			对缺失值进行填充
			isnull			返回一个布尔值对象，判断哪些值是缺失值
			notnull			isnull的否定形式
	1.3.2 缺失值处理
		判断缺失值	
			df.isnull()
			df[df['名字'].isnull()] - 所有'名字'是NaN的数据构成的dataframe
		填充缺失值
			df['评分'].fillna(0) -> 以0填充缺失值
			df['评分'].fillna(np.mean(df['评分']), inplace=True） -> 以'评分'这一列的平均值来填充缺失值
		删除缺失值
			df.dropna()常用参数  -> 	how='all' 删除全为空值的行或列
									inplace=True 覆盖之前的数据
									axis=0 选择行或列
	1.3.3 异常值处理
		异常值，即在数据集中存在不合理的值，又称离群点。
			eg:年龄-1, 笔记本重量1吨，身高18米，人数为12.8位...
		对于异常值来说数量都会很少，在不影响整体数据分布的情况下，直接删除即可。
		其他属性的异常值处理，会在格式转换部分进行讨论。
	1.3.4 数据保存
		数据处理之后，将数据重新保存到源文件
		df.to_excel('...')/ df.to_csv('...') -> 可以是新的文件url

1.4 作业1笔记整理
	1.4.1 列表(List)、集合(Set)和字典(Dic)的转换
		1 List -> Set
			setsample = set(listsample)
		2 Set -> List
			listsample = list(setsample)
		3 List -> Dictionary 
			列表不能直接使用dict转换成字典
			将两个列表内的元素两两组合成键值对，列表长度不一致时无匹配的元素就不展示
			a = ['a1','a2','a3','a4']
			b = ['b1','b2','b3']
			d = zip(a,b)
			print(dict(d))  # {'a1': 'b1', 'a2': 'b2', 'a3': 'b3'}
		4 Dictionary -> List
			dit = {'name':'zxf', 'age':'22', 'gender':'male', 'address':'shanghai'}
			1）将字典的key转换成列表
				lst = list(dit) 
				# ['name', 'age', 'gender', 'address']
			2）将字典的value转换成列表
				lst2 = list(dit.values()) 
				# ['zxf', '22', 'male', 'shanghai']
	1.4.2 Set(集合)
		1 集合特性
			- 集合可以实现去重的功能
			- 集合可以实现关系测试：交集, 差集, 并集, 是否子集, 是否没有交集
			- 集合是无序的, 不重复的数据类型
			- 不支持索引, 不支持切片取值, 不支持重复, 不支持连接
			- 支持成员操作符, 支持for loop
		2 定义集合
			s = set([1,2,3,1,2,3])  
				# set([1, 2, 3])
			s = set("hello") 
				# set(['h', 'e', 'l', 'o'])
			s = set({'a':1, 'b':2, 'c':3})  
				# set(['a', 'c', 'b'])
		3 集合的增删查改
			增加
				1）setsample.add(x) --> x就是要增加的元素值
				2）s = {1,2,3,2,1,5,6}
				   s1 = {'a', 'b', 'c'}
				   s.update(s1)
						# set(['a', 1, 2, 3, 5, 6, 'c', 'b'])
			查找
					s1 = {1, 2, 3, 4}
					s2 = {1, 2, 3, 5}
				1）数学关系
					-交集
						print s1 & s2 # set([1, 2, 3])
					- 并集
						print s1 | s2 # set([1, 2, 3, 4, 5])
					- 差集
						print s1 - s2 # set([4])
						print s2 - s1 # set([5])
					- 对等差分
						print s1 ^ s2 # set([4, 5])
				2）集合方法
					- 交集
						print s1.intersection(s2) # set([1, 2, 3])
					- 并集
						print s1.union(s2) # set([1, 2, 3, 5])
					- 差集
						print s1.difference(s2) # set([4])
						print s2.difference(s1) # set([5])
					- 对等差分
						print s1.symmetric_difference(s2) # set([4, 5])
			删除
				1）pop()随机删除
					s.pop()
				2）remove()删除集合指定元素，如果不存在则报错
					s.remove(x) --> x就是要删除的元素值
				3）discard()删除集合指定元素，如果不存在则do nothing
					s.discard(x) --> x就是要删除的元素值
				4）clear()清空集合元素
					s.clear()
	1.4.2 List(列表)
		1 List的增删查改
			判断一个list是否包含某个元素
				theList = ['a', 'b', 'c']
					if 'a'in theList:
						print 'a in the list'
					if 'd' not in theList:
						print 'd is not in the list'
			增加
				listsample.append(x) --> x就是要增加的元素值
				添加方式是最后面加入
			删除
				del listsample[index]
					 -列表名-  -索引-
			修改
				找到对应的下标然后通过索引直接赋值即可
				listsample[0] = 5
			倒序
				listsample=list1[::-1]
		2 List常见和常用的函数
				  函数名称				作用
				cmp(list1,list2)	比较两个list的元素, 返回值等于0时表示两个list相等
				len(list)			求取list的长度
				max(list)			求取list的长度
				min(list)			求取list的长度
				list(seq)			将元组转化为列表
		3 List相关的方法
				  方法名称						作用
				list.append(obj)		尾部插入新对象
				list.count(obj)			统计对象在列表中出现的次数
				list.extend(list1)		用新列表扩展现有列表(list末尾一次性追加一个list1)
				list.insert(index,obj)	将对象obj插入到列表的index位置中去
				list.pop()				删除列表中最后一个位置的元素, 并返回删除的元素值（尾删）
				list.remove(obj)		删除列表中第一次出现的对象obj
										(相同元素，删除最前面的那一个, 如果只有一个就直接删除此obj)
				list.reverse()			逆序操作（反向列表中的所有元素）
				list.sort()				默认升序, list.sort(reverse=True)降序
	1.4.3 Dictionary(字典)
		1 添加键值对
			直接给不存在的 key 赋值即可
			dict[key] = value
		2 修改键值对
			key不能被修改, 只能修改value
			若新添加元素的key与已存在元素的key相同, 那么对应的值就会被新的值替换掉，达到修改value的目的
			a = {'数学': 95, '语文': 89, '英语': 90}
			print(a)  # {'数学': 95, '语文': 89, '英语': 90}
			a['语文'] = 100
			print(a)  # {'数学': 95, '语文': 100, '英语': 90}
		3 删除键值对			
			1）clear()方法 -> 删除字典内所有元素
				dict.clear()
			2）pop()方法 -> 删除字典给定key所对应的value, 返回值为被删除的值
				pop_object = dict.pop(元素)
			3）popitem()方法 -> 随即返回并删除字典中的一对键值
				pop_object = site.popitem()
			4）del()方法 -> 能删单一的元素也能清空
				-单一元素-
					a = {'数学': 95, '语文': 89, '英语': 90}
					del a['语文']
					del a['数学']
					print(a)  # {'英语': 90}
				-清空-
					del dict
		4 判断字典中是否存在指定键值对
			首先判断字典中是否有对应的key, 可以使用in或not in运算符
			a = {'数学': 95, '语文': 89, '英语': 90}
			# 判断 a 中是否包含名为'数学'的key
			print('数学' in a) # True
			# 判断 a 是否包含名为'物理'的key
			print('物理' in a) # False
			其次获取对应key的value, 即可判断是否包含指定键值对
		5 update()将新字典所有键值对添加到旧字典上, 如果key有重复, 则直接覆盖
			a={'age': 18, 'name': 'gaoqi', 'job': 'techer'}
			b={'top': 173, 'name': 'Vince', 'tel': 123456}
			a.update(b)
			print(a)  #result：{'tel': 123456, 'name': 'Vince', 'top': 173, 'age': 18, 'job': 'techer'}



2.1 pandas数据处理
	使用part1中处理过的数据'movie_data.xlsx'作为示例(codes在'pandas-2.ipynb')
	2.1.1 数据格式转换(包含异常值处理)
		dtype:
			查看'投票人数'列的数据类型 - df['投票人数'].dtype
		astype:
			将'投票人数'列的数据类型转换为int - df['投票人数']=df['投票人数'].astype('int') 
		--1--
			将年份转化为整数格式
			1）转换
				df['年代']=df['年代'].astype('int')
					异常值(Error:invalid literal for int() with base 10: '2008\u200e')
			2）找到异常值所在行
				df[df['年代']=='2008\u200e']
			3）为异常值赋正常值
				df.loc[14934, '年代']=2008
				df.loc[14934]
			4）转换 并查看前五行数据及类型
				df['年代']=df['年代'].astype('int')
				df['年代'][:5]
		--2--
			将时长转化为整数格式
			1）转换
				df['时长']=df['时长'].astype('int')
					异常值(Error:invalid literal for int() with base 10: '8U')
			2）找到异常值所在行
				df[df['时长']=='8U']
			3）删掉异常值
				df.drop([31175], inplace=True)
			4）转换
				df['时长']=df['时长'].astype('int')
					异常值(Error:invalid literal for int() with base 10: '12J')
			5）找到异常值所在行
				df[df['时长']=='12J']
			6）删掉异常值
				df.drop([32464], inplace=True)
			7）转换 并查看前五行数据及类型
				df['时长']=df['时长'].astype('int')
				df['时长'][:5]
	2.1.2 排序
		--1--
			按照投票人数排序
				默认升序 - df.sort_values(by='投票人数')
				降序 - df.sort_values(by='投票人数', ascending=False)[:5]
		--2--
			多个值排序，先按照评分，再按照投票人数
				df.sort_values(by=['评分', '投票人数'], ascending=False)
	2.1.3 基本统计分析
		1 描述性统计
			dataframe.describe():对dataframe中的数值型数据进行描述性统计
			通过描述性统计可以发现一些异常值
			 - 找到异常值 df[df['年代']>2018]
			 - 删除异常值 df.drop(df[df['年代']>2018].index, inplace=True)
			 - 重新定义索引 df.index=range(len(df))
		2 最值
			1）最大值 --> .max()
				df['评分'].max()
			2）最小值 --> .min()
				df['评分'].min()
		3 均值和中值
			1）均值 --> .mean()
				df['评分'].mean()
			2）中值 --> .median()
				df['评分'].median()
		4 方差和标准差
			1）方差 --> .var()
				df['评分'].var()
			2）标准差 --> .std()
				df['评分'].std()
		5 求和
			--> .sum()
				df['投票人数'].sum()
		6 相关系数, 协方差
			1）相关系数 --> .corr()
				df[['投票人数', '评分']].corr()
			2）协方差 --> .cov()
				df[['投票人数', '评分']].cov()
		7 计数
			- 电影总数	len(df) # 38170
			- 电影来自多少国家	df['产地'].unique() # 包含所有国家的array
								len(df['产地'].unique()) # 28
			- 合并重复数据(USA和美国, 德国和西德, 俄罗斯和苏联...)
				单个替换 - df['产地'].replace('USA', '美国', inplace=True)
				列表替换 - df['产地'].replace(['西德', '苏联'], ['德国', '俄罗斯'], inplace=True)
			- 计算每一年电影的数量
				即是计数每个年份出现的次数
				df.年代.value_counts()
			- 电影产出前五的国家或地区
				df.产地.value_counts()[:5]

2.2 数据透视
	excel中数据透视表使用广泛, Pandas也有类似功能 --> pivot_table
	使用pandas中的pivot_table需要确保对数据的理解，清楚希望通过透视表解决什么问题
	2.2.1 透视表常见操作
		1 基础形式
			- 看到所有数据而非省略号
				pd.set_option('max_columns', 100)
				pd.set_option('max_rows', 500)
			- 以年代为索引查看信息
				pd.pivot_table(df, index=['年代'])
		2 可以有多个索引
			- 以年代和产地为索引
				pd.pivot_table(df, index=['年代', '产地']) -> 先以年代归类再以产地归类
		3 可以指定需要统计汇总的数据
			- 以年代和产地为索引, 只查看评分数据
				pd.pivot_table(df, index=['年代', '产地'], values=['评分'])
		4 可以指定函数来统计不同的统计值
			- 以年代和产地为索引分类, 对投票人数进行求和
				pd.pivot_table(df, index=['年代', '产地'], values=['投票人数'], aggfunc=np.sum)
			- 以产地为索引分类，对'投票人数', '评分'进行求和/求平均值
				pd.pivot_table(df, index=['产地'], values=['投票人数', '评分'], aggfunc=[np.sum, np.mean])
		5 NaN(非数值)处理
			- 想移除它们可以使用'fill_value'将其设置为0
				pd.pivot_table(df, index=['产地'], aggfunc=[np.sum, np.mean], fill_value=0)
		6 加入margins=True
			- 可以在下方显示一些总和数据, 最后一行 ALL
				pd.pivot_table(df, index=['产地'], aggfunc=[np.sum, np.mean], fill_value=0, margins=True)
		7 对不同值执行不同函数
			- 向aggfunc传递一个字典, 但其副作用是须将标签做得简洁
			- 对各个地区投票人数求和, 对评分求均值
				pd.pivot_table(df, index=['产地'], values=['投票人数', '评分'], aggfunc={'投票人数':np.sum, '评分':np.mean}, fill_value=0)
			- 对各个年份的投票人数求和, 对评分求均值
				pd.pivot_table(df, index=['年代'], values=['投票人数', '评分'], aggfunc={'投票人数':np.sum, '评分':np.mean}, fill_value=0)
	2.2.2 透视表过滤
		- table=pd.pivot_table(df, index=['年代'], values=['投票人数', '评分'], aggfunc={'投票人数':np.sum, '评分':np.mean}, fill_value=0)
		- type(table) # dataframe
			透视的结果就是一个dataframe
		- table[table.index==1994]
			查看1994年的'投票人数'和'评分'信息
		- table.sort_values('评分', ascending=False)
			对各个'年代'按照'评分'进行倒序排列, 看到评分均值第一为1924年
			但早期电影数量少, 均分相对都较高
		- pd.pivot_table(df, index=['产地', '年代'], values=['投票人数', '评分'], aggfunc={'投票人数':np.sum, '评分':np.mean}, fill_value=0)
			按照多个索引汇总过滤

2.3 作业2笔记整理
	2.3.1



3.1 数据重塑和轴向转换
	使用part2中处理过的数据'movie_data2.xlsx'作为示例(codes在'pandas-3.ipynb')
	3.1.1 层次化索引
		层次化索引是pandas一项主要功能，使我们在一个轴上拥有多个索引
		1 Series的层次化索引
			s = pd.Series(np.arange(1, 10), index=[['a','a','a','b','b','c','c','d','d'], [1,2,3,1,2,3,1,2,3]])
			s['a'] - 取外层索引
			s['a':'c'] - 切片操作
			s[:, 1] - 取内层索引
			s['a', 1] - 以内外层索引取到具体的值
		2 通过unstack方法可以将Series变成一个DataFrame
			外层索引做行索引, 内层索引做列名
			s.unstack()
			s.unstack().stack() - dataframe转回series
		3 DataFrame的层次化索引
			对于dataframe, 行和列都可以层次化索引
			data=pd.DataFrame(np.arange(12).reshape(4,3))
				没有索引定义操作时, 默认行列索引都为0,1,2,3......
			data=pd.DataFrame(np.arange(12).reshape(4,3), index=[['a','a','b','b'], [1,2,1,2]], columns=[['A','A','B'],['Z','X','C']])
				现在的dataframe中每一个值都是通过四个索引来决定的, 行的内外索引和列的内外索引
			data['A'] - 选取列
			data.index.names=['row1','row2']
				对行索引级别设置名称
			data.columns.names=['column1','column2']
				对列索引级别设置名称
			data.swaplevel('row1','row2')
				调整行的位置
	3.1.2 使用层次化索引处理'movie_data2.xlsx'
		---set_index 把列变成索引---
		---reset_index 把索引变成列---
		1 把'产地'和'年代'同时设成索引, '产地'是外层索引, '年代'为内层索引
			df=df.set_index(['产地', '年代'])
		2 每一个索引都是一个元组
			df.index[0]  # ('美国', 1994)
		3 获取所有美国电影，'产地'已经变成索引 
			--> .loc
			df.loc['美国']
		  获取1994年所有电影, 先交换内外索引的位置
		  	df=df.swaplevel('产地', '年代')
		  	df.loc[1994]
		4 这样做最大的好处是可以简化很多的筛选环节
	3.1.3 取消层次化索引
		df=df.reset_index() --> 回到原始格式
	3.1.4 数据旋转
		1 行列转换, 以前五部电影为例
			.T 可以直接让数据的行列进行交换
				data=df[:5] - 去前五列
				data.T - 行列转换
		2 dataframe也可以使用stack和unstack, 转化为层次索引的Series
			data.stack() - 转到层次化索引的Series
			data.stack().unstack() - 转回dataframe

3.2 数据分组和分组运算
	GroupBy技术：实现数据分组 / 分组运算, 作用类似于数据透视表
	3.2.1 按照电影的产地进行分组
		group=df.groupby(df['产地'])
			定义一个分组变量group
		group.mean()
		group.sum()
			计算分组后的各个统计量
		df['评分'].groupby(df['年代']).mean()
			计算每年的平均评分
	--只对数值变量进行分组运算, 若将某列类型转为非数值的则不会对其列数据进行运算
	3.2.2 传入多个分组变量
		df.groupby([df['产地'], df['年代']]).mean()
		- 获得每个地区, 每年的电影评分的均值
			group=df['评分'].groupby([df['产地'], df['年代']]).mean()
	3.2.3 Series会通过unstack方法转化为dataframe
		means.unstack - 会产生缺失值

3.3 离散化处理
	3.3.1 基础概念
		1 实际数据分析中, 对有的数据类型不关注数据的绝对取值, 只关注它所处的区间或者等级
			eg:评分>9的电影定义为A, 7~9定义为B, 5~7定义为C, 3~5定义为D, <3定义为E
		2 离散化也可以称为分组、区间化
			pandas提供了方便的函数 cut()
			pd.cut(x, bins, right=True, labels=None, rebits=False, precision=3, innclude_lowest=False)参数解释：
					x:需要离散化的数据、Series、DataFrame对象
					bins:分组的依据
					right=True：默认包括右边端点, 右闭
					innclude_lowest=False：默认不包括左端点, 左开
					labels=None：是否要用标签替换分出来的分组
					rebits=False：返回x当中的一个值
					precision=3
	3.3.2 离散化数据
		1 评分>9的电影定义为A, 7~9定义为B, 5~7定义为C, 3~5定义为D, <3定义为E
			df['评分等级']=pd.cut(df['评分'], [0,3,5,7,9,10], labels=['E','D','C','B','A'])
		2 根据投票人数来显示电影的热门程度
			投票越多越热门
			bins=np.percentile(df['投票人数'], [0,20,40,60,80,100])
				定义分组的依据和比例
			df['热门程度']=pd.cut(df['投票人数'], bins, labels=['E','D','C','B','A'])
				新建'热门程度'列, 按照A,B,C,D,E分类
		3 烂片集合
			投票人数多且评分很低
			df[(df.热门程度=='A') & (df.评分等级=='E')]
		4 冷门高分电影
			df[(df.热门程度=='E') & (df.评分等级=='A')]

3.4 合并数据集
	3.4.1 append
		把数据集拆分为多个
			df_usa=df[df.产地=='美国']
			df_china=df[df.产地=='中国大陆']
		合并
			df_china.append(df_usa)
	3.4.2 merge
		pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
				 left_index=False, right_index=False, sort=True, suffixes=('_x','_y'),
				 copt=True, indicator=False)
						left:对象
						right:另一个对象
						on:要加入的列名
						left_on:左边的综合使用作为键列
						right_on:右边的综合使用作为键列
						left_index:若为True, 则使用行索引从左综合作为联接键
						right_index:同上用法
						how:默认为内部(左 右 内部 外部)
						sort:默认值为True
						suffixes:字符串后缀不适用于重叠列的元组
		选取六部电影做示例
			-定义df1
				df1=df.loc[:5]
			-定义df2
				df2=df.loc[:5][['名字', '产地']]
				df2['票房']=[123344,23454,55556,333,6666,444]
			-打乱df2
				df2=df2.sample(frac=1)
				df2.index=range(len(df2))
			-合并
				pd.merge(df1, df2, how='inner', on='名字')
	3.4.3 concat
		将多个数据进行批量合并
			-定义df1, df2, df3
				df1=df[:10]
				df2=df[100:110]
				df3=df[200:210]
			-批量合并
				dff=pd.concat([df1,df2,df3], axis=0) - axis=0默认增加行

