from __future__ import print_function
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from utils import TextureDataset, contentLoss, plotStats, setNoise, learnedWN
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import sys
from network import weights_init,NetED, Discriminator,calc_gradient_penalty
from prepareTemplates import getTemplates,getImage

from config import opt,bMirror,nz,nDep,criterion
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

N=0
ngf = int(opt.ngf)
ndf = int(opt.ndf)

desc="fc"+str(opt.fContent)+"_ngf"+str(ngf)+"_ndf"+str(ndf)+"_dep"+str(nDep)+"-"+str(opt.nDepD)

if opt.WGAN:
    desc +='_WGAN'
if opt.LS:
        desc += '_LS'
if bMirror:
    desc += '_mirror'
if opt.contentScale !=1 or opt.textureScale !=1:
    desc +="_scale"+str(opt.contentScale)+";"+str(opt.textureScale)
desc += '_cLoss'+str(opt.cLoss)

targetMosaic=getTemplates(opt,N)
fixnoise = torch.FloatTensor(1, nz, targetMosaic.shape[2]//2**nDep,targetMosaic.shape[3]//2**nDep)
print("fixed variables",fixnoise.data.shape,targetMosaic.data.shape) 
netD = Discriminator(ndf, opt.nDepD, bSigm=not opt.LS and not opt.WGAN)

##################################
netMix =NetED(ngf, nDep, nz, nc=3, ncIn=3, bTanh=True, Ubottleneck=opt.zGL)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device",device)

Gnets=[netMix]
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
targetMosaic=targetMosaic.to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#netD.parameters()
optimizerU = optim.Adam([param for net in Gnets for param in list(net.parameters())], lr=opt.lr, betas=(opt.beta1, 0.999))

def ganGeneration(content, noise,templates=None, bVis = False):
    x = netMix.d(noise)
    if bVis:
        return x,0,0,0
    return x
        
buf=[]

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        t0 = time.time()
        sys.stdout.flush()
        content = next(iter(cdataloader))[0]
        content = content.to(device)

        # train with real
        netD.zero_grad()
        text, _ = data
        batch_size = content.size(0)##if we use texture and content of diff size may have issue -- just trim
        text=text.to(device) 
        output = netD(text)
        errD_real = criterion(output, output.detach()*0+real_label)
        errD_real.backward()
        D_x = output.mean()

        # train with fake
        noise=setNoise(noise)
        fake = ganGeneration(content, noise)
        output = netD(fake.detach())
        errD_fake = criterion(output, output.detach()*0+fake_label)
        errD_fake.backward()

        D_G_z1 = output.mean()
        errD = errD_real + errD_fake
        if opt.WGAN:
            gradient_penalty = calc_gradient_penalty(netD, text, fake[:text.shape[0]])##for case fewer text images
            gradient_penalty.backward()

        optimizerD.step()
        if i >0 and opt.WGAN and i%opt.dIter!=0:
            continue ##critic steps to 1 GEN steps

        for net in Gnets:
            net.zero_grad()

        content = next(iter(cdataloader))[0]
        content = content.to(device)
        # train with fake -- create again
        noise=setNoise(noise)
        fake = ganGeneration(content, noise)
        output = netD(fake)
        loss_adv = criterion(output, output.detach()*0+real_label)
        D_G_z2 = output.mean()

        noise[:,:opt.zGL]= netMix.e(content)
        fake2 = ganGeneration(content, noise)##TODO freeze gradient of decode?
        cLoss= contentLoss(fake2,content[:,:3],netR,opt)

        errG = loss_adv +opt.fContent*cLoss
        errG.backward()
        optimizerU.step()

        print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f content %4f time %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 D_x, D_G_z1, D_G_z2,cLoss.item(),time.time()-t0))

        buf += [[D_x.item(), D_G_z1.item(), D_G_z2.item(),cLoss.item()]]

        ### RUN INFERENCE AND SAVE LARGE OUTPUT MOSAICS
        if i % 100 == 0:
            a=np.array(buf)
            plotStats(a,opt.outputFolder+desc)
            vutils.save_image(text,    '%s/real_textures.jpg' % opt.outputFolder,  normalize=True)
            vutils.save_image(torch.cat([content,fake,fake2],2),'%s/mosaic_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch,desc),normalize=True)

            fixnoise=setNoise(fixnoise)
            fixnoise[:,:opt.zGL]=netMix.e(targetMosaic)

            vutils.save_image(fixnoise.view(-1,1,fixnoise.shape[2],fixnoise.shape[3]), '%s/noiseBig_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch, desc),normalize=True)

            netMix.eval()
            with torch.no_grad():
                if False:##if desired use this to make prediction in 1 pass, may be good for real-time video
                    fakebig = ganGeneration(targetMosaic, fixnoise)
                else:
                    fakebig,_,_,_ = splitW(targetMosaic, fixnoise, None, ganGeneration)

            vutils.save_image(fakebig,'%s/mosaicBig_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch,desc),normalize=True)


            ##RUN OUT-OF-SAMPLE
            with torch.no_grad():
                try:
                    im=getImage(opt.testImage, bDel=True)
                    im = im.to(device)
                    print("test image size", im.shape)
                    fixnoise2 = torch.FloatTensor(1, nz, im.shape[2] // 2 ** nDep, im.shape[3] // 2 ** nDep)
                    fixnoise2 = fixnoise2.to(device)
                    noise=setNoise(fixnoise2)
                    fakebig,_,_,_= splitW(im, fixnoise2, None, ganGeneration)
                    vutils.save_image(fakebig, '%s/mosaicTransfer_epoch_%03d_%s.jpg' % (opt.outputFolder, epoch, desc), normalize=True)
                except Exception as e:
                    print ("test image error",e)
            netMix.train()

            ##OPTIONAL
            ##save/load model for later use if desired
            #outModelName = '%s/netG_epoch_%d_%s.pth' % (opt.outputFolder, epoch*0,desc)
            #torch.save(netU.state_dict(),outModelName )
            #netU.load_state_dict(torch.load(outModelName))
