#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import os
from PIL import Image
import PIL
from config import opt,bMirror,nDep
import sys

##normal coordinate grid
def getCanonic(x):
    theta= torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(1,2,3)
    flow= F.affine_grid(theta, x.size())#output Tensor of size (N×H×W×2), here N=1
    return flow

## when exceeding grid -- start from beginning of grid
def wrap(f):
    for i in range(f.shape[1]):##dim3=1 is const
        if f[0,i,0,1]>1:           
            delta =   int((f[0,i,0,1]-1)/2)+1
            f[0,i,:,1]-=2*delta
        if f[0,i,0,1]<-1:
            delta =   int((f[0,i,0,1]+1)/2)-1
            f[0,i,:,1]-=2*delta
    
    for i in range(f.shape[2]):##dim3=0 is const
        if f[0,0,i,0]>1:           
            delta =   int((f[0,0,i,0]-1)/2)+1
            f[0,:,i,0]-=2*delta  
        if f[0,0,i,0]<-1:
            delta =   int((f[0,0,i,0]+1)/2)-1
            f[0,:,i,0]-=2*delta
    return f
    
## when exceeding grid -- mirror and go back
def reflect_mirror(f):
    for i in range(f.shape[1]):##dim3=1 is const
        if f[0,i,0,1]>3:           
            old = f[0,i,0,1]*1.0 
            delta =   int((f[0,i,0,1]-3)/4)+1
            f[0,i,:,1]-=4*delta
            if f[0,i,0,1]>3 or f[0,i,0,1]<-1:   
                print (old,f[0,i,0,1],delta)
        if f[0,i,0,1]<-1:
            old = f[0,i,0,1]*1.0
            delta =   int((f[0,i,0,1]+1)/4)-1
            f[0,i,:,1]-=4*delta
            if f[0,i,0,1]>3 or f[0,i,0,1]<-1:   
                print (old,f[0,i,0,1],delta)

    for i in range(f.shape[2]):##dim3=0 is const
        if f[0,0,i,0]>3:           
            old=f[0,0,i,0]*1.0 
            delta =   int((f[0,0,i,0]-3)/4)+1
            f[0,:,i,0]-=4*delta  
            if f[0,0,i,0]>3 or f[0,0,i,0]<-1: 
                print (old,f[0,0,i,0],delta)
        if f[0,0,i,0]<-1:
            old=f[0,0,i,0]*1.0
            delta =   int((f[0,0,i,0]+1)/4)-1
            f[0,:,i,0]-=4*delta
            if f[0,0,i,0]>3 or f[0,0,i,0]<-1: 
               print (old,f[0,0,i,0],delta)

    #    print ("Max",f.max(),f.min())
    assert(f[:,:,:,1].max().item()<=3)
    assert(f[:,:,:,1].min().item()>=-1)
    assert(f[:,:,:,0].max().item()<=3)
    assert(f[:,:,:,0].min().item()>=-1)
    f=torch.where(torch.ByteTensor(f>1), 2-f, f) #so 1:3 maps to 1:-1 - -reflection
    return f 

##interpolation mode when output size is bigger than input image size
## @param x coordinates in any range
## @output coordinates scaled with some logic in -1,1
def reflect(x):
    if bMirror:
        return reflect_mirror(x)
    else:
        return wrap(x)

# z is 1x3xHxW
def randomTile(flow,z):
    f=flow*1.0
    ratioW=flow.shape[1]/float(z.shape[2])
    ratioH=flow.shape[1+1]/float(z.shape[2+1])
    #print (ratioH,ratioW)
    f[:,:,:,0]*= ratioH 
    f[:,:,:,0+1]*= ratioW
    f[:,:,:,0]+=np.random.rand()*40
    f[:,:,:,1]+=np.random.rand()*40
    f=reflect(f)
    out = F.grid_sample(z, f)#,padding_mode="reflection"
    sys.stdout.flush()
    print ("crop ready",out.shape)
    return out

##read an image file and return pytorch tensor in range -1 to 1
## @param bDel -- if true crop to some power of 2
def getImage(name, bDel=False):
    img = Image.open(name)
    if not bDel:##so texture, may rescale
            if opt.textureScale != 1:
                img = img.resize((int(img.size[0] * opt.textureScale), int(img.size[1] * opt.textureScale)), PIL.Image.LANCZOS)
    else:
            if opt.contentScale != 1:
                img = img.resize((int(img.size[0] * opt.contentScale), int(img.size[1] * opt.contentScale)), PIL.Image.LANCZOS)


    img = np.array(img)/255.0*2-1

    if bDel:##put to some power of 2 size, due to split routines
        delW = img.shape[1]%2**(nDep+1)
        delH = img.shape[0]%2**(nDep+1)
        if delH >0:
            img = img[:-delH]
        if delW >0:
            img = img[:,:-delW]
    img = img.swapaxes(2,1).swapaxes(1,0)
    img= torch.FloatTensor(img[np.newaxis])
    print ("image input",img.shape)
    return img

def getTemplates(opt,N,vis=True,path=str(bMirror)):
    x=getImage(opt.contentPath + os.listdir(opt.contentPath)[0], True)
    if N ==0:
        return x

    flow=getCanonic(x)
    nTex = len(os.listdir(opt.texturePath))
    out = torch.FloatTensor(N,3,x.shape[2],x.shape[3]).half()
    files=os.listdir(opt.texturePath)
    for n in range(N):
        z=getImage(opt.texturePath + files[n % nTex])
        out[n:n+1] = randomTile(flow,z)
    if vis:
        vutils.save_image(out[:8].float(),path+'templates.jpg', normalize=True,nrow=4,padding=10)##limit to 25 the shown templates
    return torch.cat([x,flow.permute(0,3,1,2)],1),out

##@param target is always (B,3,H,W). RGB image. The canonical coordinates of the used templates will  be added to last 2 channels
##@param template is always (N,3,H,W)
def randCrop(target,template,npx,canonicC):
    N = template.shape[0]
    nbatch=target.shape[0]
    te =torch.FloatTensor(nbatch,N,3,npx,npx)
    target=torch.cat([target,target[:,:2]*0],1)##0 coords, to be filled later
    for i in range(nbatch):
        r = np.random.randint(template.shape[2] - npx)
        r2 = np.random.randint(template.shape[3] - npx)
        te[i:i+1] = template[:,:,r:r+npx,r2:r2+npx].float()##all N textures set

        coords = canonicC[:,3:5,r:r+npx,r2:r2+npx]##so they belong to template, for RAW mode
        target[i,3:5]=coords##hack -- get coordinates here, belonging to templates
    return target,te


##only crop from single image!!
##@param target is always (1,3,H,W)
##@param template is always (N,3,H,W)
def randCropOverfit(dummyBatch,template,npx,target):
    nbatch = dummyBatch.shape[0]
    N = template.shape[0]
    te =torch.FloatTensor(nbatch,N,3,npx,npx)
    ba =torch.FloatTensor(nbatch,5,npx,npx)##cont + coord
    for i in range(nbatch):
        r= np.random.randint(target.shape[2]-npx)
        r2= np.random.randint(target.shape[3]-npx)
        ba[i:i+1] = target[:,:,r:r+npx,r2:r2+npx]

        te[i:i+1] = template[:,:,r:r+npx,r2:r2+npx].float()##all N textures set

        coords = target[:,3:5,r:r+npx,r2:r2+npx]##so they belong to template, for RAW mode
        ba[i,3:5]=coords##hack -- get coordinates here, belonging to templates
    return ba,te

#BxNx3xHxW  templates
#BxNxHxW mix
def getTemplateMixImage(mix, templates, mode='bilinear'):
    if type(mix) is list:
            out=[]
            for xx in mix:
                out.append(getTemplateMixImage(xx, templates, mode))
            return out

    nFT = templates.shape[4] // mix.shape[3]
    if nFT > 1:#if  mix diff size-- upsample
        mix = F.upsample(mix, scale_factor=nFT, mode=mode)
    N = mix.shape[1]
    B=mix.shape[0]
    H=mix.shape[2]
    W=mix.shape[3]
    C=templates.shape[2]##usually 3, unless attention on another level
    mix = mix.permute(0,2,3,1).contiguous().view(-1,1,N)
    
    templates = templates.permute(0,3,4,1,2).contiguous().view(-1,N,C)
    #print ("mix templ",mix.shape,templates.shape)
    prod = torch.bmm(mix,templates)##should be BHWx3
    return prod.view(B,H,W,C).permute(0,3,1,2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contentPath', required=True, help='path to dataset')
    parser.add_argument('--texturePath', required=True, help='path to dataset')
    parser.add_argument('--N', type=int, default=4)
    opt = parser.parse_args()
    print(opt)    
    
    getTemplates(opt,opt.N,vis=True)
    
   
    
