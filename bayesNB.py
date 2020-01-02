dataset=[      ['青绿','蜷缩','浊响','清晰','凹陷','硬滑','是'],
                ['乌黑','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
                ['乌黑','蜷缩','浊响','清晰','凹陷','硬滑','是'],
                ['青绿','蜷缩','沉闷','清晰','凹陷','硬滑','是'],
                ['浅白','蜷缩','浊响','清晰','凹陷','硬滑','是'],
                ['青绿','稍蜷','浊响','清晰','稍凹','软粘','是'],
                ['乌黑','稍蜷','浊响','稍糊','稍凹','软粘','是'],
                ['乌黑','稍蜷','浊响','清晰','稍凹','硬滑','是'],
                ['乌黑','稍蜷','沉闷','稍糊','稍凹','硬滑','否'],
                ['青绿','硬挺','清脆','清晰','平坦','软粘','否'],
                ['浅白','硬挺','清脆','模糊','平坦','硬滑','否'],
                ['浅白','蜷缩','浊响','模糊','平坦','软粘','否'],
                ['青绿','稍蜷','浊响','稍糊','凹陷','硬滑','否'],
                ['浅白','稍蜷','沉闷','稍糊','凹陷','硬滑','否'],
                ['乌黑','稍蜷','沉闷','稍糊','稍凹','软粘','否'],
                ['浅白','蜷缩','浊响','模糊','平坦','硬滑','否'],
                ['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑','否']]

def trainNB(dataset,tezhenglist):
    
    p0vec={};p1vec={}
    for tezheng in tezhenglist:
        p0vec=dict(p0vec,**{tezheng:{}})        #字典合并
        p1vec=dict(p1vec,**{tezheng:{}})
    #p1vec=dict(p0vec)                     #p1vec\p0vec分别存储每个分类下各个特征值的占比
    
    p1_num=0    #好瓜占比初始化
    p0_num=0    #
    for sample in dataset:
        #print(p0vec)
        
        i=0
        for tezheng in sample[:-1]:
            
           if tezheng not in p0vec[tezhenglist[i]]:
                p0vec[tezhenglist[i]][tezheng]=0
                p1vec[tezhenglist[i]][tezheng]=0          #先记录所有属性值
            
           if sample[-1]=="是":             #好瓜
                p1vec[tezhenglist[i]][tezheng]+=1
           else:
                p0vec[tezhenglist[i]][tezheng]+=1
           i+=1
        if sample[-1]=="是":       #各分类统计
            p1_num+=1
        else:
            p0_num+=1
    #return p0vec,p1vec,p1_num
    #拉普拉斯修正计算
    for key,value in p0vec.items():
         for itemkey,itemvalue in value.items():
             p0vec[key][itemkey]=round((itemvalue+1)/(p0_num+len(p0vec[key])),3)
            
    for key,value in p1vec.items():
         for itemkey,itemvalue in value.items():
             p1vec[key][itemkey]=round((itemvalue+1)/(p1_num+len(p1vec[key])),3)
                
    return p0vec,p1vec,round((p1_num+1)/(p1_num+p0_num+2),3)

def classifyNB(newdata, p0vec, p1vec, pcgood):
    '''newdata:新的待分类特征，p0vec:样本中坏瓜条件下各特征值的条件概率，pcgood:样本中好瓜类别占比'''
    i=0
    p1=0
    p0=0
    for charactervalue in newdata:
        
        p1+=math.log(p1vec[tezhenglist[i]][charactervalue])
        p0+=math.log(p0vec[tezhenglist[i]][charactervalue])
        i+=1
        
    p1 = p1+ math.log(pcgood)
    p0 = p0+ math.log(1-pcgood)
    if p1 > p0:
        return 1
    else:
        return 0
        
>>> newdata=['青绿','稍蜷','浊响','清晰','凹陷','硬滑']
>>>
>>> classifyNB(newdata, p0vec, p1vec, pcgood)
1
>>> newdata=['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑']
>>> classifyNB(newdata, p0vec, p1vec, pcgood)
0
