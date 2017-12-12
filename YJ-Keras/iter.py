import os
import subprocess
import numpy as np
import datetime
import random
import warnings
import ROOT as rt
import math
from array import array

class wkiter(object):
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0, fstart=0, friend=0, zboson=0,varbs=0,w=0):
    self.istrain=istrain
    #if(batch_size<100):
    #  print("batch_size is small it might cause error")
    self.friend=friend
    self.zboson=zboson
    self.w=w
    print(self.friend,istrain)
    if(friend!=0):  
      if(self.zboson==0):
        data_path=["root/cutb/mg5_pp_qq_balanced_pt_100_500_","root/cutb/mg5_pp_gg_balanced_pt_100_500_"]
      else:
        data_path=["root/cut/mg5_pp_zq_passed_pt_100_500_","root/cut/mg5_pp_zg_passed_pt_100_500_"]
      self.gf=0
      self.qf=0
      self.gq=int(rat*friend)
      self.qg=int(rat*friend)
      self.frat=int(friend*rat)
      self.qfile=[]
      self.gfile=[]
      self.qjet=[]
      self.gjet=[]
      self.qim=[]
      self.gim=[]
      self.qlabel=[]
      self.glabel=[]
      self.qEntries=[]
      self.gEntries=[]
      self.qBegin=[]
      self.gBegin=[]
      self.qEnd=[]
      self.gEnd=[]
      self.qnum=0
      self.gnum=0
      for i in range(friend):
        dataname1=data_path[0]+str(fstart+i+1)+"_img.root"
        dataname2=data_path[1]+str(fstart+i+1)+"_img.root"
        self.qfile.append(rt.TFile(dataname1,'read'))
        self.gfile.append(rt.TFile(dataname2,'read'))
        self.qjet.append(self.qfile[i].Get("image"))
        self.gjet.append(self.gfile[i].Get("image"))
        self.qim.append(array('B', [0]*(3*(arnum*2+1)*(arnum*2+1))))
        self.gim.append(array('B', [0]*(3*(arnum*2+1)*(arnum*2+1))))
        #self.qlabel.append(array('B', [0]))
        #self.glabel.append(array('B', [0]))
        self.qjet[i].SetBranchAddress("image", self.qim[i])
        self.gjet[i].SetBranchAddress("image", self.gim[i])
        #self.qjet[i].SetBranchAddress("label", self.qlabel[i])
        #self.gjet[i].SetBranchAddress("label", self.glabel[i])
        self.qEntries.append(self.qjet[i].GetEntriesFast())
        self.gEntries.append(self.gjet[i].GetEntriesFast())
        self.qBegin.append(int(begin*self.qEntries[i]))
        self.gBegin.append(int(begin*self.gEntries[i]))
        self.qEnd.append(int(self.qEntries[i]*end))
        self.gEnd.append(int(self.gEntries[i]*end))
        self.qnum+=self.qEnd[i]-self.qBegin[i]
        self.gnum+=self.gEnd[i]-self.gBegin[i]

      self.a=self.gBegin[0]
      self.b=self.qBegin[0]
      self.aq=self.gBegin[self.frat]
      self.bg=self.qBegin[self.frat]
    else:
      #self.file=rt.TFile(data_path,'read')
      dataname1=data_path[0]
      dataname2=data_path[1]
      self.qfile=rt.TFile(dataname1,'read')
      self.gfile=rt.TFile(dataname2,'read')
      self.qjet=self.qfile.Get("image")
      self.gjet=self.gfile.Get("image")
      self.qim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
      self.gim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
      self.qjet.SetBranchAddress("image", self.qim)
      self.gjet.SetBranchAddress("image", self.gim)
      #self.qlabel = array('B', [0])
      #self.glabel = array('B', [0])
      #self.qjet.SetBranchAddress("label", self.qlabel)
      #self.gjet.SetBranchAddress("label", self.glabel)
      self.qEntries=self.qjet.GetEntriesFast()
      self.gEntries=self.gjet.GetEntriesFast()
      self.qBegin=int(begin*self.qEntries)
      self.gBegin=int(begin*self.gEntries)
      self.qEnd=int(self.qEntries*end)
      self.gEnd=int(self.gEntries*end)
      self.a=self.gBegin
      self.b=self.qBegin
    self.ratt=rat
    self.rat=sorted([1-rat,rat])
    self.batch_size = batch_size
    if(varbs==0):
      self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33)])
    else:
      data_names=['images','variables']
      self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33),(self.batch_size,5)])
    self.varbs=varbs
    self._provide_label = zip(label_names, [(self.batch_size,)])
    self.arnum=arnum
    self.maxx=maxx
    self.maxy=maxy
    self.endfile=0
    self.endcut=endcut
  def __iter__(self):
    return self

  def reset(self):
    if(self.friend!=0):
      #print("@@",self.istrain,"g",self.gf,"q",self.qf,"@@")
      for i in range(self.friend):
        self.qjet[i].GetEntry(self.qBegin[i])
        self.gjet[i].GetEntry(self.gBegin[i])
      self.a=self.gBegin[0]
      self.b=self.qBegin[0]
      self.aq=self.gBegin[self.frat]
      self.bg=self.qBegin[self.frat]
      self.gf=0
      self.qf=0
      self.gq=self.frat
      self.qg=self.frat
      self.endfile=0
    else:
      self.qjet.GetEntry(self.qBegin)
      self.gjet.GetEntry(self.gBegin)
      self.a=self.gBegin
      self.b=self.qBegin
      self.endfile = 0

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def close(self):
    self.file.Close()
  def sampleallnum(self):
    return self.Entries
  def trainnum(self):
    return self.End-self.Begin
  def totalnum(self):
    if(self.friend!=0):  
      #a=0
      #for i in range(self.friend):
      #  a+=self.qEnd[i]-self.qBegin[i]+self.gEnd[i]-self.qBegin[i]
      return int((self.qnum+self.gnum)/self.batch_size*0.80)
    else:
      return int((self.qEnd-self.qBegin+self.gEnd-self.qBegin)/self.batch_size*0.80)

  def next(self):
    while self.endfile==0:
      arnum=self.arnum
      jetset=[]
      variables=[]
      labels=[]
      rand=random.choice(self.rat)
      if(self.friend!=0 and self.zboson==0 and self.istrain==1):
        rand=0.4
      if(self.friend!=0 and self.zboson==0 and self.istrain==0):
        rand=0.31354286
      if(self.friend!=0 and self.zboson==1):
        rand=0.526
      if(self.friend!=0 and self.zboson==0 and self.istrain==1 and self.w==1):
        rand=0.37
      if(self.friend!=0):
        rand=self.gnum/1./(self.qnum+self.gnum)
      for i in range(self.batch_size):
        if(self.friend!=0):
          if(self.w==0):
            if(random.random()<rand):
              self.gjet[self.gf].GetEntry(self.a)
              self.a+=1
              jetset.append(np.array(self.gim[self.gf]).reshape((3,2*arnum+1,2*arnum+1)))
              labels.append([1,0])
              if(self.varbs==1):
                variables.append([self.gjet[self.gf].ptD,self.gjet[self.gf].axis1,self.gjet[self.gf].axis2,self.gjet[self.gf].nmult,self.gjet[self.gf].cmult])
              if(self.a>=self.gEnd[self.gf]):
                self.gf+=1
                if(self.gf==self.friend):
                  self.endfile=1
                  self.gf=0
                self.a=self.gBegin[self.gf]
            else:
              self.qjet[self.qf].GetEntry(self.b)
              self.b+=1
              jetset.append(np.array(self.qim[self.qf]).reshape((3,2*arnum+1,2*arnum+1)))
              labels.append([0,1])
              if(self.varbs==1):
                variables.append([self.qjet[self.qf].ptD,self.qjet[self.qf].axis1,self.qjet[self.qf].axis2,self.qjet[self.qf].nmult,self.qjet[self.qf].cmult])
              if(self.b>=self.qEnd[self.qf]):
                self.qf+=1
                if(self.qf==self.friend):
                  self.endfile=1
                  self.qf=0
                self.b=self.gBegin[self.qf]
          ####-----------
	  else:
            if(random.random()<rand):
              if(random.random()<self.ratt):
                self.gjet[self.gf].GetEntry(self.a)
                self.a+=1
                jetset.append(np.array(self.gim[self.gf]).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append([1,0])
                if(self.varbs==1):
                  variables.append([self.gjet[self.gf].ptD,self.gjet[self.gf].axis1,self.gjet[self.gf].axis2,self.gjet[self.gf].nmult,self.gjet[self.gf].cmult])
                if(self.a>=self.gEnd[self.gf]):
                  self.gf+=1
                  if(self.gf==self.frat):
                    self.endfile=1
                    self.gf=0
                  self.a=self.gBegin[self.gf]
              else:
                self.qjet[self.gq].GetEntry(self.aq)
                self.aq+=1
                jetset.append(np.array(self.gim[self.gq]).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append([1,0])
                if(self.varbs==1):
                  variables.append([self.qjet[self.gq].ptD,self.qjet[self.gq].axis1,self.qjet[self.gq].axis2,self.qjet[self.gq].nmult,self.qjet[self.gq].cmult])
                if(self.aq>=self.gEnd[self.gq]):
                  self.gq+=1
                  if(self.gq==self.friend):
                    self.endfile=1
                    self.gq=self.frat
                  self.aq=self.gBegin[self.gq]
            else:
              if(random.random()<self.ratt):
                self.qjet[self.qf].GetEntry(self.b)
                self.b+=1
                jetset.append(np.array(self.qim[self.qf]).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append([0,1])
                if(self.varbs==1):
                  variables.append([self.qjet[self.qf].ptD,self.qjet[self.qf].axis1,self.qjet[self.qf].axis2,self.qjet[self.qf].nmult,self.qjet[self.qf].cmult])
                if(self.b>=self.qEnd[self.qf]):
                  self.qf+=1
                  if(self.qf==self.frat):
                    self.endfile=1
                    self.qf=0
                  self.b=self.gBegin[self.qf]
              else:
                self.gjet[self.qg].GetEntry(self.bg)
                self.bg+=1
                jetset.append(np.array(self.qim[self.qg]).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append([0,1])
                if(self.varbs==1):
                  variables.append([self.gjet[self.qg].ptD,self.gjet[self.qg].axis1,self.gjet[self.qg].axis2,self.gjet[self.qg].nmult,self.gjet[self.qg].cmult])
                if(self.bg>=self.qEnd[self.qg]):
                  self.qg+=1
                  if(self.qg==self.friend):
                    self.endfile=1
                    self.qg=self.frat
                  self.bg=self.gBegin[self.qg]


        else:
          if(random.random()<0.5):
          #if(random.random()<rand):
            self.gjet.GetEntry(self.a)
            self.a+=1
            jetset.append(np.array(self.gim).reshape((3,2*arnum+1,2*arnum+1)))
            labels.append([1,0])
            if(self.a>=self.gEnd):
              self.a=self.gBegin
              self.endfile=1
          else:
            self.qjet.GetEntry(self.b)
            self.b+=1
            jetset.append(np.array(self.qim).reshape((3,2*arnum+1,2*arnum+1)))
            labels.append([0,1])
            if(self.b>=self.qEnd):
              self.b=self.gBegin
              self.endfile=1
          #if(rand<0.5):
          #    labels.append(0)
          #else:
          #    labels.append(1)
          #if(self.endcut==0 and self.ent>=self.End):
              #self.ent=self.Begin
              #self.endfile=1
      if(self.varbs==1):
        data=[np.array(jetset),np.array(variables)]
      else:
        data=np.array(jetset)
      label=np.array(labels)
      yield data, label
    #else:
      #if(self.istrain==1):
      #  print "\n",datetime.datetime.now()  
      #raise StopIteration

  def test(self):
    while self.endfile==0:
      arnum=self.arnum
      jetset=[]
      variables=[]
      labels=[]
      rand=random.choice(self.rat)
      if(self.friend!=0 and self.zboson==0 and self.istrain==1):
        rand=0.4
      if(self.friend!=0 and self.zboson==0 and self.istrain==0):
        rand=0.31354286
      if(self.friend!=0 and self.zboson==1):
        rand=0.526
      if(self.friend!=0 and self.zboson==0 and self.istrain==1 and self.w==1):
        rand=0.37
      if(self.friend!=0):
        rand=self.gnum/1./(self.qnum+self.gnum)
      for i in range(self.batch_size):
        if(self.friend!=0):
          if(self.w==0):
            if(random.random()<rand):
              self.gjet[self.gf].GetEntry(self.a)
              self.a+=1
              jetset.append(np.array(self.gim[self.gf]).reshape((3,2*arnum+1,2*arnum+1)))
              labels.append([1,0])
              if(self.varbs==1):
                variables.append([self.gjet[self.gf].ptD,self.gjet[self.gf].axis1,self.gjet[self.gf].axis2,self.gjet[self.gf].nmult,self.gjet[self.gf].cmult])
              if(self.a>=self.gEnd[self.gf]):
                self.gf+=1
                if(self.gf==self.friend):
                  self.endfile=1
                  self.gf=0
                self.a=self.gBegin[self.gf]
            else:
              self.qjet[self.qf].GetEntry(self.b)
              self.b+=1
              jetset.append(np.array(self.qim[self.qf]).reshape((3,2*arnum+1,2*arnum+1)))
              labels.append([0,1])
              if(self.varbs==1):
                variables.append([self.qjet[self.qf].ptD,self.qjet[self.qf].axis1,self.qjet[self.qf].axis2,self.qjet[self.qf].nmult,self.qjet[self.qf].cmult])
              if(self.b>=self.qEnd[self.qf]):
                self.qf+=1
                if(self.qf==self.friend):
                  self.endfile=1
                  self.qf=0
                self.b=self.gBegin[self.qf]
          ####-----------
	  else:
            if(random.random()<rand):
              if(random.random()<self.ratt):
                self.gjet[self.gf].GetEntry(self.a)
                self.a+=1
                jetset.append(np.array(self.gim[self.gf]).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append([1,0])
                if(self.varbs==1):
                  variables.append([self.gjet[self.gf].ptD,self.gjet[self.gf].axis1,self.gjet[self.gf].axis2,self.gjet[self.gf].nmult,self.gjet[self.gf].cmult])
                if(self.a>=self.gEnd[self.gf]):
                  self.gf+=1
                  if(self.gf==self.frat):
                    self.endfile=1
                    self.gf=0
                  self.a=self.gBegin[self.gf]
              else:
                self.qjet[self.gq].GetEntry(self.aq)
                self.aq+=1
                jetset.append(np.array(self.gim[self.gq]).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append([1,0])
                if(self.varbs==1):
                  variables.append([self.qjet[self.gq].ptD,self.qjet[self.gq].axis1,self.qjet[self.gq].axis2,self.qjet[self.gq].nmult,self.qjet[self.gq].cmult])
                if(self.aq>=self.gEnd[self.gq]):
                  self.gq+=1
                  if(self.gq==self.friend):
                    self.endfile=1
                    self.gq=self.frat
                  self.aq=self.gBegin[self.gq]
            else:
              if(random.random()<self.ratt):
                self.qjet[self.qf].GetEntry(self.b)
                self.b+=1
                jetset.append(np.array(self.qim[self.qf]).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append([0,1])
                if(self.varbs==1):
                  variables.append([self.qjet[self.qf].ptD,self.qjet[self.qf].axis1,self.qjet[self.qf].axis2,self.qjet[self.qf].nmult,self.qjet[self.qf].cmult])
                if(self.b>=self.qEnd[self.qf]):
                  self.qf+=1
                  if(self.qf==self.frat):
                    self.endfile=1
                    self.qf=0
                  self.b=self.gBegin[self.qf]
              else:
                self.gjet[self.qg].GetEntry(self.bg)
                self.bg+=1
                jetset.append(np.array(self.qim[self.qg]).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append([0,1])
                if(self.varbs==1):
                  variables.append([self.gjet[self.qg].ptD,self.gjet[self.qg].axis1,self.gjet[self.qg].axis2,self.gjet[self.qg].nmult,self.gjet[self.qg].cmult])
                if(self.bg>=self.qEnd[self.qg]):
                  self.qg+=1
                  if(self.qg==self.friend):
                    self.endfile=1
                    self.qg=self.frat
                  self.bg=self.gBegin[self.qg]


        else:
          if(random.random()<0.5):
          #if(random.random()<rand):
            self.gjet.GetEntry(self.a)
            self.a+=1
            jetset.append(np.array(self.gim).reshape((3,2*arnum+1,2*arnum+1)))
            labels.append([1,0])
            if(self.a>=self.gEnd):
              self.a=self.gBegin
              self.endfile=1
          else:
            self.qjet.GetEntry(self.b)
            self.b+=1
            jetset.append(np.array(self.qim).reshape((3,2*arnum+1,2*arnum+1)))
            labels.append([0,1])
            if(self.b>=self.qEnd):
              self.b=self.gBegin
              self.endfile=1
          #if(rand<0.5):
          #    labels.append(0)
          #else:
          #    labels.append(1)
          #if(self.endcut==0 and self.ent>=self.End):
              #self.ent=self.Begin
              #self.endfile=1
      if(self.varbs==1):
        data=[np.array(jetset),np.array(variables)]
      else:
        data=np.array(jetset)
      label=np.array(labels)
      return [data,label]
    #else:
      #if(self.istrain==1):
      #  print "\n",datetime.datetime.now()  
      #raise StopIteration

