# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:14:51 2019

@author: arschneider
"""
import pandas as pd
import os

os.chdir('C:/Users/arschneider/Documents/Projetos/Nike/Hard Launch/SAC reports')


df = pd.read_csv('Copy of ROD_ Nike_09_2019_INFRA_emails_30_09.csv', sep=';',encoding='latin-1')
df = df.dropna(thresh=4)

def remove_comma(data):
    return data.replace(',', '')  

names = list(df.columns)

df[names[2]] = df[names[2]].map(lambda x: str(x)[:-1]).astype(float)
df[names[3]] = df[names[3]].map(lambda x: str(x)[:-1]).astype(float)
df[names[4]] = df[names[4]].map(lambda x: str(x)[:-1]).astype(float)
df[names[8]]= df[names[8]].apply(remove_comma).astype(int)
df[names[11]] = df[names[11]].apply(remove_comma).astype(int)

Total = df.iloc[0]

df = df[27:]
#crete empty list
day_list = [None] * (df.shape[0])
for i in range (df.shape[0]):
    day_temp = { 'day' :  df['Month'].iloc[i],
            'mail_less_24': df[names[2]].iloc[i],
            'mail_more_24': df[names[3]].iloc[i],
            'backlog': df[names[4]].iloc[i],
            'received': df[names[8]].iloc[i],
            'answered': df[names[11]].iloc[i]
              }
    
    day_temp['vol_less_24'] = round(day_temp['received']*(day_temp['mail_less_24']/100))
    day_temp['vol_more_24'] = round(day_temp['received']*(day_temp['mail_more_24']/100))
    day_temp['vol_backlog'] = round(day_temp['received']*(day_temp['backlog']/100))
    
    day_list[i] = day_temp
    
volume_total_received = 0;
volume_total_answered_less24 = 0;
volume_total_answered_more24 = 0;
volume_total_backlog = 0;
    
for i in range (len(day_list)):
    volume_total_received = volume_total_received + day_list[i]['received']
    volume_total_answered_less24 = volume_total_answered_less24 + day_list[i]['vol_less_24']
    volume_total_answered_more24 = volume_total_answered_more24 + day_list[i]['vol_more_24']
    volume_total_backlog = volume_total_backlog + day_list[i]['vol_backlog']
    
    
    
percentage_less_24 = volume_total_answered_less24*100/volume_total_received
percentage_more_24 = volume_total_answered_more24*100/volume_total_received
percentage_backlog = volume_total_backlog*100/volume_total_received

dataT = {'Dia':['E-mail < 24h','E-mail>24h','Backlog','Contact Received','Contact Answered']}
dados = pd.DataFrame(dataT)
for i in range(len(day_list)):    
    dataT = [str(day_list[i]['mail_less_24'])+'%',
         str(day_list[i]['mail_more_24'])+'%',str(day_list[i]['backlog'])+'%',
         str(day_list[i]['received']),str(day_list[i]['answered'])]
    dados[day_list[i]['day']]  = dataT
        
dataT =[str(percentage_less_24)+'%',str(percentage_more_24)+'%',str(percentage_backlog)+'%',str(volume_total_received),
               str(volume_total_answered_less24+volume_total_answered_more24)] 

dados['WTD'] = dataT


dataT = [str(Total[names[2]])+'%',str(Total[names[3]])+'%',str(Total[names[4]])+'%',str(Total[names[8]]),str(Total[names[11]])]
dados['MTD'] = dataT
dados.to_csv('dados_semanais_emails_30_09.csv',sep = ';',index=False)


