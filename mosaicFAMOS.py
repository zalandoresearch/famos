from __future__ import print_function
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from utils import TextureDataset,contentLoss,plotStats,blend,total_variation,rgb_channels,gramMatrix,invblend,tvArray,setNoise,learnedWN
import torchvision.transforms as transforms
import torchvision.utils as vutils
import itertools
import numpy as np
import sys
from network import weights_init,NetUskip, Discriminator,calc_gradient_penalty,NetU_MultiScale
from prepareTemplates import getTemplates,getTemplateMixImage,getImage

from config import opt,bMirror,nz,nDep,criterion
if opt.trainOverfit:
    from prepareTemplates import randCropOverfit as randCrop
else:
    from prepareTemplates import randCrop
import datetime
from splitInference import splitW
import time

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

canonicT=[transforms.RandomCrop(opt.imageSize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
mirrorT= []
if bMirror:
    mirrorT += [transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip()]
transformTex=transforms.Compose(mirrorT+canonicT)
dataset = TextureDataset(opt.texturePath,transformTex,opt.textureScale)
transformCon=transforms.Compose(canonicT)
cdataset = TextureDataset(opt.contentPath,transformCon,opt.contentScale)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
cdataloader = torch.utils.data.DataLoader(cdataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

N=opt.N
assert (N>0)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

desc="fc"+str(opt.fContent)+ "_fAlpha" + \
     str(opt.fAlpha)+"_fTV"+str(opt.fTV)+"_fEntropy"+str(opt.fEntropy)+ "_fDiversity"+str(opt.fDiversity)+\
     "_ngf"+str(ngf)+"_N"+str(N)+"_dep"+str(nDep)

if opt.WGAN:
    desc +='_WGAN'
if opt.LS:
        desc += '_LS'
if bMirror:
    desc += '_mirror'
if opt.contentScale !=1 or opt.textureScale !=1:
    desc +="_scale"+str(opt.contentScale)+";"+str(opt.textureScale)
desc += '_cLoss'+str(opt.cLoss)

if not opt.coordCopy:
    desc += "no coord copy"

targetMosaic,templates=getTemplates(opt,N,vis=True,path=opt.outputFolder+desc)
fixnoise = torch.FloatTensor(1, nz, targetMosaic.shape[2]//2**nDep,targetMosaic.shape[3]//2**nDep)
print("fixed variables",fixnoise.data.shape,targetMosaic.data.shape) 
netD = Discriminator(ndf, opt.nDepD, bSigm=not opt.LS and not opt.WGAN)

##################################
if opt.multiScale:
    netMix = NetU_MultiScale(ngf, nDep, nz, bSkip=opt.skipConnections, nc=N + 5, ncIn=5, bTanh=False, bCopyIn=opt.coordCopy, Ubottleneck=opt.Ubottleneck)
else:
    netMix =NetUskip(ngf, nDep, nz, bSkip=opt.skipConnections, nc=N + 5, ncIn=5, bTanh=False, bCopyIn=opt.coordCopy, Ubottleneck=opt.Ubottleneck)##copy coords more often
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device",device)

Gnets=[netMix]

if opt.refine:
    netRefine=NetUskip(ngf, nDep, nz, bSkip=True, nc=5, ncIn=4 * 3 + 2 + 2, bTanh=False)
    Gnets +=[netRefine]
if opt.cLoss>=100:
    from network import ColorReconstruction
    netR = ColorReconstruction(50, 1)#
    Gnets+=[netR]
elif opt.cLoss==10:
    from network import PerceptualF
    netR=PerceptualF()
else:
    netR = None
if opt.zPeriodic:
    Gnets += [learnedWN]

for net in [netD] + Gnets:
    try:
        net.apply(weights_init)
    except Exception as e:
        print (e,"weightinit")
    pass
    net=net.to(device)
    print(net)

NZ = opt.imageSize//2**nDep
noise = torch.FloatTensor(opt.batchSize, nz, NZ,NZ)

real_label = 1
fake_label = 0

noise=noise.to(device)
fixnoise=fixnoise.to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#netD.parameters()
optimizerU = optim.Adam([param for net in Gnets for param in list(net.parameters())], lr=opt.lr, betas=(opt.beta1, 0.999))

def famosGeneration(content, noise, templatePatch, bVis = False):
    if opt.multiScale >0:
        x = netMix(content, noise,templatePatch)
    else:
        x = netMix(content,noise)
    a5=x[:,-5:]
    A =4*nn.functional.tanh(x[:,:-5])##smooths probs somehow
    A = nn.functional.softmax(1*(A - A.detach().max()), dim=1)
    mixed = getTemplateMixImage(A, templatePatch)
    alpha = nn.functional.sigmoid(a5[:,3:4])
    beta = nn.functional.sigmoid(a5[:, 4:5])
    fake = blend(nn.functional.tanh(a5[:,:3]),mixed,alpha,beta)

    ##call second Unet to refine further
    if opt.refine:
        a5=netRefine(torch.cat([content,mixed,fake,a5[:,:3],tvArray(A)],1),noise)
        alpha = nn.functional.sigmoid(a5[:, 3:4])
        beta = nn.functional.sigmoid(a5[:, 4:5])
        fake = blend(nn.functional.tanh(a5[:, :3]), mixed, alpha, beta)

    if bVis:
        return fake,torch.cat([alpha,beta,(alpha+beta)*0.5],1),A,mixed#alpha
    return fake
        
buf=[]
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        t0 = time.time()
        sys.stdout.flush()
        content = next(iter(cdataloader))[0]
        content = content.to(device)

        content,templatePatch = randCrop(content,templates,opt.imageSize,targetMosaic)
        templatePatch =templatePatch.to(device)##needed -- I create new float Tensor in randCrop
        if opt.trainOverfit:
            content = content.to(device)

        if epoch==0 and i==0:
            print ("template size",templatePatch.shape)
        # train with real
        netD.zero_grad()
        text, _ = data
        batch_size = content.size(0)##if we use texture and content of diff size may have issue -- just trim
        text=text.to(device) 
        output = netD(text)##used to find correct size for label
        errD_real = criterion(output, output.detach()*0+real_label)
        errD_real.backward()
        D_x = output.mean()

        # train with fake
        noise=setNoise(noise)
        fake, alpha, A, mixedI = famosGeneration(content, noise, templatePatch, True)
        output = netD(fake.detach())#???why detach
        errD_fake = criterion(output, output.detach()*0+fake_label)
        errD_fake.backward()

        if opt.fAdvM > 0:
            loss_adv_mixed = criterion(netD(mixedI.detach()), output.detach() * 0 + fake_label)
            loss_adv_mixed.backward()

        D_G_z1 = output.mean()
        errD = errD_real + errD_fake
        if opt.WGAN:
            gradient_penalty = calc_gradient_penalty(netD, text, fake[:text.shape[0]])##for case fewer text images
            gradient_penalty.backward()

        optimizerD.step()

        content = next(iter(cdataloader))[0]
        content = content.to(device)
        content, templatePatch = randCrop(content,templates,opt.imageSize,targetMosaic)
        templatePatch = templatePatch.to(device)  ##needed -- I create new float Tensor in randCrop
        content = content.to(device)

        for net in Gnets:
            net.zero_grad()

        # train with fake -- create again
        noise=setNoise(noise)
        fake, alpha, A, mixedI  = famosGeneration(content, noise, templatePatch, True)

        output = netD(fake)
        loss_adv = criterion(output, output.detach()*0+real_label)
        D_G_z2 = output.mean()

        if opt.fAdvM>0:
            outputM = netD(mixedI)
            loss_adv_mixed = criterion(outputM, outputM.detach() * 0 + real_label)
            D_G_z2m = outputM.mean()
        else:
            D_G_z2m =D_G_z2*0
            loss_adv_mixed = loss_adv*0

        cLoss= contentLoss(fake,content[:,:3],netR,opt)
        cLoss2= contentLoss(mixedI,content[:,:3],netR,opt)

        entropy = (-A * (1e-8 + A).log()).mean()##entropy
        tv= total_variation(A)

        alpha_loss = alpha.mean()##large means more conv content; small is focus on mixedtemplate result
        atv= total_variation(alpha)
        diversity= gramMatrix(A.mean(3).mean(2).unsqueeze(2).unsqueeze(1)).mean()    #.std(1).mean()*-1  #force various templates -- across batch variance
        errG = loss_adv + opt.fAdvM * loss_adv_mixed + opt.fContent * cLoss + opt.fContentM * cLoss2 + opt.fAlpha * alpha_loss + opt.fEntropy * entropy + opt.fTV * tv + \
               +0.02 * atv + opt.fDiversity * diversity
        errG.backward()
        optimizerU.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f mixP %.4f content %4f template %.4f alphareg %.4f entropy %.4f tv %.4f atv %.4f diversity %.4f time %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, D_G_z2m, cLoss.item(), cLoss2.item(), alpha_loss.item(), entropy.item(), tv.item(), atv.item(), diversity.item(), time.time() - t0))

        buf += [[D_x.item(), D_G_z1.item(), D_G_z2.item(), cLoss.item(), cLoss2.item(), alpha_loss.item(), entropy.item(), tv.item(), atv.item(), diversity.item()]]

        ### RUN INFERENCE AND SAVE LARGE OUTPUT MOSAICS
        if i % 100 == 0:
            a=np.array(buf)
            plotStats(a,opt.outputFolder+desc)
            vutils.save_image(text,    '%s/real_textures.jpg' % opt.outputFolder,  normalize=True)
            IG=invblend(fake,mixedI,alpha[:,:1],alpha[:,1:2])
            vutils.save_image(torch.cat([content[:,:3], mixedI, IG,fake, alpha, rgb_channels(A)], 2), '%s/mosaic_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch, desc), normalize=True)

            fixnoise=setNoise(fixnoise)
            for net in Gnets:
                net.eval()
            with torch.no_grad():
                if False:##do whole mosaic in 1 pass -- warning, takes a lot of memory, do not use unless you have a good reason
                    templates=templates.to(device)
                    fakebig, alpha, A, mixedbig = famosGeneration(targetMosaic, fixnoise, templates.unsqueeze(0), True)
                else:
                    fakebig, alpha, A, mixedbig = splitW(targetMosaic, fixnoise, templates.unsqueeze(0), famosGeneration)
            vutils.save_image(mixedbig,'%s/mixed_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch,desc), normalize =True)
            if True:#
                vutils.save_image(alpha,'%s/alpha_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch,desc), normalize=False)
                vutils.save_image(rgb_channels(A), '%s/blenda_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch, desc), normalize=False)##already 01
                v=nn.functional.avg_pool2d(A.view(-1, 1, A.shape[2], A.shape[3]), int(16))
                vutils.save_image(v,'%s/blendaBW_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch,desc), normalize=False)

            vutils.save_image(fakebig,'%s/mosaicBig_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch,desc),normalize=True)

            ##RUN OUT-OF-SAMPLE
            with torch.no_grad():
                try:
                    im=getImage(opt.testImage, bDel=True)
                    if im.shape[2]>targetMosaic.shape[2] or im.shape[3]>targetMosaic.shape[3]:
                        print ("cropping to original mosaic size")
                        im=im[:,:,:targetMosaic.shape[2],:targetMosaic.shape[3]]
                    im=torch.cat([im,targetMosaic[:,3:5,:im.shape[2],:im.shape[3]]],1)##coords
                    print ("test image size",im.shape)
                    fixnoise2 = torch.FloatTensor(1, nz, im.shape[2] // 2 ** nDep,im.shape[3] // 2 ** nDep)
                    fixnoise2 = fixnoise2.to(device)
                    fixnoise2 =setNoise(fixnoise2)
                    fakebig,_,_,_= splitW(im, fixnoise2, templates.unsqueeze(0), famosGeneration)
                    vutils.save_image(fakebig, '%s/mosaicTransfer_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch, desc), normalize=True)
                except Exception as e:
                    print ("test image error",e)

            for net in Gnets:
                net.train()